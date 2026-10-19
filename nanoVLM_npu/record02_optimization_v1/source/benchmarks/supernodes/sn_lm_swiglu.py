#!/usr/bin/env python3
"""SN-LM-SwiGLU Supernode Lab.

GPU evidence:
  model__18_forward_49.18/fx_graph_readable.py shows FP32 parameters/input
  autocast to BF16 around three external GEMMs. In ir_post_fusion.txt,
  sigmoid + gate multiply + up multiply + FP32/BF16 casts are one fused
  pointwise node (op6), while gate/up/down GEMMs remain external. Backward
  model__56_backward_185.67 similarly fuses the SiLU derivative chain.
NPU current:
  models/language_model.py uses two projections, F.silu, multiply, and down
  projection. Eager activation and multiply are separate launches.
Identified gaps:
  official npu_swiglu; concatenated gate/up projection; algebraic spellings;
  backend=npu scopes, while preserving FP32 master weights and checkpoint keys.

Real regimes: B=1,S=128 smoke and B=2,S={1024,4096}, H=960, I=2560.
All timings include forward and backward under BF16 autocast.
L0: dim=-1 is the only semantics-preserving API parameter; no local env knob.
L1: four equivalent eager SiLU/multiply spellings.
L2: four official npu_swiglu call/layout spellings plus grouped-matmul probe.
L3: four single-projection/manual-layout rewrites.
L4: four backend=npu scopes, isolated through SN_VARIANTS.
L5: only if official fused autograd fails. L6: only after L0-L5 fail.
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
INTERMEDIATE = 2560
ACTIVATION_DTYPE = torch.bfloat16


@dataclass
class State:
    x: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor

    @property
    def grad_tensors(self):
        return self.x, self.gate_weight, self.up_weight, self.down_weight


def make_state(batch: int, seq: int) -> State:
    generator = torch.Generator(device="cpu").manual_seed(20260831 + seq)
    x = torch.randn(batch, seq, HIDDEN, generator=generator).to(DEVICE).requires_grad_()
    gate = torch.randn(INTERMEDIATE, HIDDEN, generator=generator).mul_(HIDDEN**-0.5).to(DEVICE).requires_grad_()
    up = torch.randn(INTERMEDIATE, HIDDEN, generator=generator).mul_(HIDDEN**-0.5).to(DEVICE).requires_grad_()
    down = torch.randn(HIDDEN, INTERMEDIATE, generator=generator).mul_(INTERMEDIATE**-0.5).to(DEVICE).requires_grad_()
    return State(x, gate, up, down)


def projections(state: State):
    return F.linear(state.x, state.gate_weight), F.linear(state.x, state.up_weight)


def B0_eager(state: State):
    gate, up = projections(state)
    return F.linear(F.silu(gate) * up, state.down_weight)


def L1a_explicit_sigmoid(state: State):
    gate, up = projections(state)
    return F.linear(gate * torch.sigmoid(gate) * up, state.down_weight)


def L1b_method_sigmoid(state: State):
    gate, up = projections(state)
    return F.linear(gate * gate.sigmoid() * up, state.down_weight)


def L1c_torch_mul(state: State):
    gate, up = projections(state)
    return F.linear(torch.mul(F.silu(gate), up), state.down_weight)


def L1d_method_mul(state: State):
    gate, up = projections(state)
    return F.linear(F.silu(gate).mul(up), state.down_weight)


def L2a_native_cat(state: State):
    gate, up = projections(state)
    return F.linear(torch_npu.npu_swiglu(torch.cat((gate, up), -1), dim=-1), state.down_weight)


def L2b_native_ops(state: State):
    gate, up = projections(state)
    joined = torch.cat((gate, up), -1)
    return F.linear(torch.ops.npu.npu_swiglu.default(joined, -1), state.down_weight)


def L2c_native_positive_dim(state: State):
    gate, up = projections(state)
    return F.linear(torch_npu.npu_swiglu(torch.cat((gate, up), 2), dim=2), state.down_weight)


def L2d_native_concat_alias(state: State):
    gate, up = projections(state)
    return F.linear(torch_npu.npu_swiglu(torch.concat((gate, up), -1), dim=-1), state.down_weight)


def L2e_grouped_projection(state: State):
    # GMM is not autocast-aware: align x and weights explicitly for attempt 2.
    flat = state.x.reshape(-1, HIDDEN).to(ACTIVATION_DTYPE)
    gate, up = torch_npu.npu_grouped_matmul(
        [flat, flat],
        [state.gate_weight.t().to(ACTIVATION_DTYPE), state.up_weight.t().to(ACTIVATION_DTYPE)],
        split_item=0,
        group_type=-1,
    )
    shape = (*state.x.shape[:-1], INTERMEDIATE)
    return F.linear(F.silu(gate.reshape(shape)) * up.reshape(shape), state.down_weight)


class GroupedProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight):
        flat = x.reshape(-1, HIDDEN).to(ACTIVATION_DTYPE)
        gate_compute = gate_weight.to(ACTIVATION_DTYPE)
        up_compute = up_weight.to(ACTIVATION_DTYPE)
        gate, up = torch_npu.npu_grouped_matmul(
            [flat, flat], [gate_compute.t(), up_compute.t()],
            split_item=0, group_type=-1)
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.save_for_backward(flat, gate_compute, up_compute)
        shape = (*x.shape[:-1], INTERMEDIATE)
        return gate.reshape(shape), up.reshape(shape)

    @staticmethod
    def backward(ctx, grad_gate, grad_up):
        flat, gate_weight, up_weight = ctx.saved_tensors
        grad_gate = grad_gate.reshape(-1, INTERMEDIATE)
        grad_up = grad_up.reshape(-1, INTERMEDIATE)
        grad_x = grad_gate.matmul(gate_weight)
        grad_x.add_(grad_up.matmul(up_weight))
        grad_gate_weight = grad_gate.t().matmul(flat).float()
        grad_up_weight = grad_up.t().matmul(flat).float()
        return (grad_x.reshape(ctx.input_shape).to(ctx.input_dtype),
                grad_gate_weight, grad_up_weight)


def L5a_grouped_projection_custom_backward(state: State):
    gate, up = GroupedProjectionFunction.apply(
        state.x, state.gate_weight, state.up_weight)
    return F.linear(F.silu(gate) * up, state.down_weight)


def combined_projection(state: State):
    return F.linear(state.x, torch.cat((state.gate_weight, state.up_weight), dim=0))


def L3a_concat_projection_native(state: State):
    return F.linear(torch_npu.npu_swiglu(combined_projection(state), dim=-1), state.down_weight)


def L3b_concat_projection_manual(state: State):
    gate, up = combined_projection(state).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, state.down_weight)


def L3c_concat_projection_explicit(state: State):
    gate, up = combined_projection(state).split(INTERMEDIATE, dim=-1)
    return F.linear(gate * torch.sigmoid(gate) * up, state.down_weight)


def L3d_concat_projection_ops(state: State):
    joined = combined_projection(state)
    return F.linear(torch.ops.npu.npu_swiglu.default(joined, -1), state.down_weight)


def pointwise_silu(gate, up):
    return F.silu(gate) * up


def pointwise_explicit(gate, up):
    return gate * torch.sigmoid(gate) * up


compiled_pointwise_silu = torch.compile(pointwise_silu, backend="npu", dynamic=False, fullgraph=True)
compiled_pointwise_explicit = torch.compile(pointwise_explicit, backend="npu", dynamic=False, fullgraph=True)


def full_two_projection(x, gate_weight, up_weight, down_weight):
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)


def full_concat_projection(x, gate_weight, up_weight, down_weight):
    joined = F.linear(x, torch.cat((gate_weight, up_weight), dim=0))
    gate, up = joined.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down_weight)


compiled_full_two = torch.compile(full_two_projection, backend="npu", dynamic=False, fullgraph=True)
compiled_full_concat = torch.compile(full_concat_projection, backend="npu", dynamic=False, fullgraph=True)


def L4a_compile_pointwise_silu(state: State):
    gate, up = projections(state)
    return F.linear(compiled_pointwise_silu(gate, up), state.down_weight)


def L4b_compile_pointwise_explicit(state: State):
    gate, up = projections(state)
    return F.linear(compiled_pointwise_explicit(gate, up), state.down_weight)


def L4c_compile_full_two(state: State):
    return compiled_full_two(state.x, state.gate_weight, state.up_weight, state.down_weight)


def L4d_compile_full_concat(state: State):
    return compiled_full_concat(state.x, state.gate_weight, state.up_weight, state.down_weight)


VARIANTS: dict[str, Callable[[State], torch.Tensor]] = {
    "B0": B0_eager, "L1a": L1a_explicit_sigmoid,
    "L1b": L1b_method_sigmoid, "L1c": L1c_torch_mul,
    "L1d": L1d_method_mul, "L2a": L2a_native_cat,
    "L2b": L2b_native_ops, "L2c": L2c_native_positive_dim,
    "L2d": L2d_native_concat_alias, "L2e": L2e_grouped_projection,
    "L3a": L3a_concat_projection_native,
    "L3b": L3b_concat_projection_manual,
    "L3c": L3c_concat_projection_explicit,
    "L3d": L3d_concat_projection_ops,
    "L4a": L4a_compile_pointwise_silu,
    "L4b": L4b_compile_pointwise_explicit,
    "L4c": L4c_compile_full_two, "L4d": L4d_compile_full_concat,
    "L5a": L5a_grouped_projection_custom_backward,
}


def metrics(actual, expected):
    actual = actual.detach().float().reshape(-1)
    expected = expected.detach().float().reshape(-1)
    diff = (actual - expected).abs()
    return (diff.max().item(),
            F.cosine_similarity(actual, expected, dim=0).item(),
            diff.norm().item() / max(expected.norm().item(), 1e-12))


def evaluate(fn, state):
    with torch.autocast(device_type="npu", dtype=ACTIVATION_DTYPE):
        output = fn(state)
        loss = output.float().square().mean()
    grads = torch.autograd.grad(loss, state.grad_tensors)
    torch.npu.synchronize()
    return output, grads


def clone_state(source):
    return State(*(x.detach().clone().requires_grad_(True) for x in source.grad_tensors))


def correctness(name, batch, seq):
    source = make_state(batch, seq)
    expected_out, expected_grads = evaluate(B0_eager, clone_state(source))
    actual_out, actual_grads = evaluate(VARIANTS[name], clone_state(source))
    worst = 1.0
    for kind, actuals, expecteds in (("out", (actual_out,), (expected_out,)),
                                     ("grad", actual_grads, expected_grads)):
        for index, (actual, expected) in enumerate(zip(actuals, expecteds)):
            maximum, cosine, relative = metrics(actual, expected)
            worst = min(worst, cosine)
            print(f"CHECK name={name} B={batch} S={seq} {kind}{index} max_abs={maximum:.7g} cosine={cosine:.9f} rel={relative:.7g}", flush=True)
    if worst <= 0.9999:
        raise RuntimeError(f"strict correctness gate failed: {worst}")


def benchmark(name, seq, warmup=3, reps=7):
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
    print(f"TIMING name={name} B=2 S={seq} f+b median_ms={statistics.median(samples):.4f} p95_ms={p95:.4f}", flush=True)


def main():
    torch.npu.set_device(0)
    print("=== SN-LM-SwiGLU Supernode Lab ===", flush=True)
    print(f"SCHEMA swiglu={torch.ops.npu.npu_swiglu.default._schema}", flush=True)
    print(f"SCHEMA grouped_matmul={torch.ops.npu.npu_grouped_matmul.default._schema}", flush=True)
    requested = os.environ.get("SN_VARIANTS", "").strip()
    names = requested.split(",") if requested else list(VARIANTS)
    for name in names:
        print(f"--- {name} ---", flush=True)
        try:
            correctness(name, 1, 128)
            correctness(name, 2, 4096)
            benchmark(name, 1024)
            benchmark(name, 4096)
        except Exception as error:
            print(f"ERROR name={name} type={type(error).__name__} message={str(error)[:1600]}", flush=True)


if __name__ == "__main__":
    main()
