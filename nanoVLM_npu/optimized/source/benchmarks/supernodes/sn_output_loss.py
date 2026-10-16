#!/usr/bin/env python3
"""SN-Output-Loss Supernode Lab.

GPU evidence:
  model__61_forward_178.61 receives hidden [2,4096,960] FP32, tied head
  [49218,960] FP32 and targets [2,4096]. Autocast makes GEMM BF16 and emits
  logits [8192,49218]. The vocabulary GEMM remains one external kernel; CE
  mask, log-softmax and reductions are compiler-scheduled into two fused
  pointwise groups. model__61_backward_180.62 similarly keeps large GEMMs
  external while fusing pointwise gradient work.
GPU post-compile target:
  1 external head GEMM plus fused CE pointwise/reduction scheduling, BF16
  logits and FP32-stable CE accumulation; full logits are still materialized.
NPU current:
  models/vision_language_model.py:75-78 applies the tied head to every 8192
  row, reshapes, then calls F.cross_entropy(ignore_index=-100). Training drops
  returned logits. About 4/7 rows carry non-ignored labels in benchmark data.
Identified gaps:
  avoid head GEMM and logits storage for ignored rows; avoid redundant 3D/2D
  views; test fused npu_cross_entropy_loss; test equivalent projection layouts;
  try four backend=npu scopes. Filtering must preserve mean reduction, tied
  weights, FP32 master gradients, and exactly zero ignored-row hidden grads.

Real regimes: B=2,S={1024,4096}, H=960, V=49218, activation BF16 under
autocast and FP32 master hidden/weight. Timings include forward and backward.
L0: audited CE knobs; reduction/ignore/smoothing are training semantics, so the
    four candidates only spell the same required defaults explicitly.
L1: flatten before head; boolean filtering; nonzero+index_select; masked_select.
L2: official fused CE on all rows and on filtered rows through public/op APIs.
L3: linear/matmul/addmm/einsum projection spellings on the cumulative filter.
L4: CE-only, head-only, full head+CE, and cumulative filtered full scope.
L5: native fused loss already owns backward; custom backward only if it fails.
L6: skipped unless all official/eager/compile paths miss the overall target.
"""
from __future__ import annotations

import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import torch_npu

DEVICE = "npu:0"
HIDDEN = 960
VOCAB = 49218
ACTIVATION_DTYPE = torch.bfloat16


@dataclass
class State:
    hidden: torch.Tensor
    weight: torch.Tensor
    targets: torch.Tensor

    @property
    def grad_tensors(self):
        return self.hidden, self.weight


def make_state(batch: int, seq: int) -> State:
    generator = torch.Generator(device="cpu").manual_seed(20260831 + seq)
    hidden = torch.randn(batch, seq, HIDDEN, generator=generator).to(DEVICE).requires_grad_()
    weight = (torch.randn(VOCAB, HIDDEN, generator=generator) * HIDDEN**-0.5).to(DEVICE).requires_grad_()
    targets = torch.randint(0, VOCAB, (batch * seq,), generator=generator)
    # Deterministic 4/7 valid ratio approximates the observed training batches.
    targets[torch.arange(batch * seq) % 7 >= 4] = -100
    return State(hidden, weight, targets.reshape(batch, seq).to(DEVICE))


def native_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss, _, _, _ = torch_npu.npu_cross_entropy_loss(logits, targets)
    return loss.reshape(())


def native_loss_ops(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss, _, _, _ = torch.ops.npu.npu_cross_entropy_loss.default(
        logits, targets, None, "mean", -100, 0.0, 0.0, False)
    return loss.reshape(())


def valid_boolean(state: State):
    hidden = state.hidden.reshape(-1, HIDDEN)
    targets = state.targets.reshape(-1)
    valid = targets != -100
    return hidden[valid], targets[valid]


def valid_index(state: State):
    hidden = state.hidden.reshape(-1, HIDDEN)
    targets = state.targets.reshape(-1)
    index = torch.nonzero(targets != -100, as_tuple=False).squeeze(1)
    return hidden.index_select(0, index), targets.index_select(0, index)


def B0_full_3d(state: State):
    logits = F.linear(state.hidden, state.weight)
    return F.cross_entropy(logits.reshape(-1, VOCAB), state.targets.reshape(-1), ignore_index=-100)


# L0: required semantics, explicitly spelled four ways.
def L0a_explicit_defaults(state: State):
    logits = F.linear(state.hidden, state.weight).reshape(-1, VOCAB)
    return F.cross_entropy(logits, state.targets.reshape(-1), weight=None,
                           reduction="mean", ignore_index=-100, label_smoothing=0.0)


def L0b_torch_api(state: State):
    logits = F.linear(state.hidden, state.weight).reshape(-1, VOCAB)
    return torch.nn.functional.cross_entropy(logits, state.targets.reshape(-1), ignore_index=-100)


def L0c_nll_equivalent(state: State):
    logits = F.linear(state.hidden, state.weight).reshape(-1, VOCAB)
    return F.nll_loss(F.log_softmax(logits, dim=-1), state.targets.reshape(-1), ignore_index=-100)


def L0d_loss_module(state: State):
    logits = F.linear(state.hidden, state.weight).reshape(-1, VOCAB)
    return torch.nn.CrossEntropyLoss(ignore_index=-100)(logits, state.targets.reshape(-1))


# L1: eliminate views/work/materialization. Later levels accumulate on filtering.
def L1a_flatten_before_head(state: State):
    hidden = state.hidden.reshape(-1, HIDDEN)
    logits = F.linear(hidden, state.weight)
    return F.cross_entropy(logits, state.targets.reshape(-1), ignore_index=-100)


def L1b_filter_boolean(state: State):
    hidden, targets = valid_boolean(state)
    return F.cross_entropy(F.linear(hidden, state.weight), targets)


def L1c_filter_index_select(state: State):
    hidden, targets = valid_index(state)
    return F.cross_entropy(F.linear(hidden, state.weight), targets)


def L1d_filter_masked_select(state: State):
    flat = state.hidden.reshape(-1, HIDDEN)
    targets = state.targets.reshape(-1)
    valid = targets != -100
    selected = torch.masked_select(flat, valid[:, None]).reshape(-1, HIDDEN)
    return F.cross_entropy(F.linear(selected, state.weight), targets.masked_select(valid))


# L2: official fused loss. Its defaults are mandatory when requires_grad=True.
def L2a_native_full(state: State):
    logits = F.linear(state.hidden.reshape(-1, HIDDEN), state.weight)
    return native_loss(logits, state.targets.reshape(-1))


def L2b_native_filtered(state: State):
    hidden, targets = valid_boolean(state)
    return native_loss(F.linear(hidden, state.weight), targets)


def L2c_native_ops_full(state: State):
    logits = F.linear(state.hidden.reshape(-1, HIDDEN), state.weight)
    return native_loss_ops(logits, state.targets.reshape(-1))


def L2d_native_ops_filtered(state: State):
    hidden, targets = valid_boolean(state)
    return native_loss_ops(F.linear(hidden, state.weight), targets)


def L2e_native_full_fp32(state: State):
    logits = F.linear(state.hidden.reshape(-1, HIDDEN), state.weight).float()
    return native_loss(logits, state.targets.reshape(-1))


def L2f_native_filtered_fp32(state: State):
    hidden, targets = valid_index(state)
    return native_loss(F.linear(hidden, state.weight).float(), targets)


def pad_vocab_for_native(logits: torch.Tensor) -> torch.Tensor:
    # GPU Teacher also pads 49218 -> 49224 around the head GEMM. Fake classes
    # have -inf logits, so they contribute exactly zero softmax probability.
    padding = logits.new_full((logits.shape[0], 6), float("-inf"))
    return torch.cat((logits, padding), dim=-1)


def L2g_native_full_padded(state: State):
    logits = F.linear(state.hidden.reshape(-1, HIDDEN), state.weight)
    return native_loss(pad_vocab_for_native(logits), state.targets.reshape(-1))


def L2h_native_filtered_padded(state: State):
    hidden, targets = valid_index(state)
    return native_loss(pad_vocab_for_native(F.linear(hidden, state.weight)), targets)


def L2i_native_filtered_index(state: State):
    hidden, targets = valid_index(state)
    return native_loss(F.linear(hidden, state.weight), targets)


# L3: projection layouts, accumulated with the strongest exact L1 candidate.
def L3a_filtered_linear(state: State):
    hidden, targets = valid_boolean(state)
    return F.cross_entropy(F.linear(hidden, state.weight), targets)


def L3b_filtered_matmul(state: State):
    hidden, targets = valid_boolean(state)
    return F.cross_entropy(torch.matmul(hidden, state.weight.t()), targets)


def L3c_filtered_addmm(state: State):
    hidden, targets = valid_boolean(state)
    bias = torch.zeros(VOCAB, device=hidden.device, dtype=hidden.dtype)
    return F.cross_entropy(torch.addmm(bias, hidden, state.weight.t()), targets)


def L3d_filtered_einsum(state: State):
    hidden, targets = valid_boolean(state)
    return F.cross_entropy(torch.einsum("nh,vh->nv", hidden, state.weight), targets)


def ce_scope(logits, targets):
    return F.cross_entropy(logits, targets, ignore_index=-100)


def head_scope(hidden, weight):
    return F.linear(hidden, weight)


def full_scope(hidden, weight, targets):
    return F.cross_entropy(F.linear(hidden.reshape(-1, HIDDEN), weight), targets.reshape(-1), ignore_index=-100)


def filtered_scope(hidden, weight, targets):
    flat = hidden.reshape(-1, HIDDEN)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != -100
    return F.cross_entropy(F.linear(flat[valid], weight), flat_targets[valid])


compiled_ce = torch.compile(ce_scope, backend="npu", dynamic=False, fullgraph=True)
compiled_head = torch.compile(head_scope, backend="npu", dynamic=False, fullgraph=True)
compiled_full = torch.compile(full_scope, backend="npu", dynamic=False, fullgraph=True)
compiled_filtered = torch.compile(filtered_scope, backend="npu", dynamic=False, fullgraph=True)


def L4a_compile_ce(state: State):
    logits = F.linear(state.hidden.reshape(-1, HIDDEN), state.weight)
    return compiled_ce(logits, state.targets.reshape(-1))


def L4b_compile_head(state: State):
    logits = compiled_head(state.hidden.reshape(-1, HIDDEN), state.weight)
    return F.cross_entropy(logits, state.targets.reshape(-1), ignore_index=-100)


def L4c_compile_full(state: State):
    return compiled_full(state.hidden, state.weight, state.targets)


def L4d_compile_filtered(state: State):
    return compiled_filtered(state.hidden, state.weight, state.targets)


VARIANTS: dict[str, Callable[[State], torch.Tensor]] = {
    "B0": B0_full_3d,
    "L0a": L0a_explicit_defaults, "L0b": L0b_torch_api,
    "L0c": L0c_nll_equivalent, "L0d": L0d_loss_module,
    "L1a": L1a_flatten_before_head, "L1b": L1b_filter_boolean,
    "L1c": L1c_filter_index_select, "L1d": L1d_filter_masked_select,
    "L2a": L2a_native_full, "L2b": L2b_native_filtered,
    "L2c": L2c_native_ops_full, "L2d": L2d_native_ops_filtered,
    "L2e": L2e_native_full_fp32, "L2f": L2f_native_filtered_fp32,
    "L2g": L2g_native_full_padded, "L2h": L2h_native_filtered_padded,
    "L2i": L2i_native_filtered_index,
    "L3a": L3a_filtered_linear, "L3b": L3b_filtered_matmul,
    "L3c": L3c_filtered_addmm, "L3d": L3d_filtered_einsum,
    "L4a": L4a_compile_ce, "L4b": L4b_compile_head,
    "L4c": L4c_compile_full, "L4d": L4d_compile_filtered,
}


def clone_state(source: State) -> State:
    return State(source.hidden.detach().clone().requires_grad_(True),
                 source.weight.detach().clone().requires_grad_(True),
                 source.targets.detach().clone())


def evaluate(fn, state):
    with torch.autocast(device_type="npu", dtype=ACTIVATION_DTYPE):
        loss = fn(state)
    grads = torch.autograd.grad(loss, state.grad_tensors)
    torch.npu.synchronize()
    return loss, grads


def metrics(actual, expected):
    actual = actual.detach().float().reshape(-1)
    expected = expected.detach().float().reshape(-1)
    diff = (actual - expected).abs()
    cosine = 1.0 if actual.numel() == 1 else F.cosine_similarity(actual, expected, dim=0).item()
    return diff.max().item(), cosine, diff.norm().item() / max(expected.norm().item(), 1e-12)


def correctness(name: str, batch: int, seq: int):
    source = make_state(batch, seq)
    expected_loss, expected_grads = evaluate(B0_full_3d, clone_state(source))
    actual_loss, actual_grads = evaluate(VARIANTS[name], clone_state(source))
    worst = 1.0
    for kind, actuals, expecteds in (("loss", (actual_loss,), (expected_loss,)),
                                     ("grad", actual_grads, expected_grads)):
        for index, (actual, expected) in enumerate(zip(actuals, expecteds)):
            maximum, cosine, relative = metrics(actual, expected)
            worst = min(worst, cosine)
            print(f"CHECK name={name} B={batch} S={seq} {kind}{index} "
                  f"max_abs={maximum:.7g} cosine={cosine:.9f} rel={relative:.7g}", flush=True)
    if worst < 0.9999:
        raise RuntimeError(f"strict correctness gate failed: {worst}")


def benchmark(name: str, seq: int, warmup: int = 2, reps: int = 5):
    state = make_state(2, seq)
    for _ in range(warmup):
        evaluate(VARIANTS[name], state)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        evaluate(VARIANTS[name], state)
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    print(f"TIMING name={name} B=2 S={seq} valid_ratio=0.5714 f+b "
          f"median_ms={statistics.median(samples):.4f} p95_ms={p95:.4f}", flush=True)


def main():
    torch.npu.set_device(0)
    print("=== SN-Output-Loss Supernode Lab ===", flush=True)
    try:
        print(f"SCHEMA native_ce={torch.ops.npu.npu_cross_entropy_loss.default._schema}", flush=True)
    except Exception as error:
        print(f"SCHEMA native_ce unavailable: {error}", flush=True)
    requested = os.environ.get("SN_VARIANTS", "").strip()
    names = requested.split(",") if requested else list(VARIANTS)
    for name in names:
        print(f"--- {name} ---", flush=True)
        try:
            correctness(name, 1, 128)
            correctness(name, 2, 1024)
            benchmark(name, 1024)
            benchmark(name, 4096)
        except Exception as error:
            print(f"ERROR name={name} type={type(error).__name__} message={str(error)[:2000]}", flush=True)


if __name__ == "__main__":
    main()
