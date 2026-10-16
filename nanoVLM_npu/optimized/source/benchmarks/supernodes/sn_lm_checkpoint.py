#!/usr/bin/env python3
"""
SN-LM-Checkpoint Supernode Lab
==============================
GPU fusion groups: 32 compiled LM forward/backward graph pairs. GPU keeps
activations and its backward does not recompute every complete transformer
block; compiled pointwise groups surround external attention/GEMM kernels.
NPU current: models/language_model.py checkpoints all 32 complete LM blocks
with non-reentrant checkpointing, recomputing RMS-attention-residual and
RMS-SwiGLU-residual. This was introduced because batch_size=2 previously OOMed.
Identified gaps: lm_dropout=0 makes RNG-state preservation redundant; native
GQA and bf16 attention reduce the memory pressure that forced checkpointing;
checkpointing only attention or only MLP may retain enough memory while
avoiding part of the recomputation.

Real shape/dtype: B=2,S=4096,C=960,bf16, 15 Q heads/5 KV heads. Four real
blocks are used so partial-checkpoint policies are measurable without turning
the isolated Lab into a full training job. Every method runs forward+backward,
reports peak allocation and compares output/input-gradient to B0.

L0: preserve_rng_state True/False.
L1: no checkpoint; checkpoint alternating blocks; checkpoint 1 of 4 blocks.
L2: current block already uses official fused attention/RMS/SwiGLU APIs.
L3: attention-only checkpoint; MLP-only checkpoint.
L4: skipped for checkpoint control flow; component compile belongs to each
    compute Supernode and fused attention explicitly disallows graph mode.
L5/L6: skipped because native autograd is correct.
"""
import gc
import math
import statistics
import time

import torch
import torch_npu
from torch.utils.checkpoint import checkpoint

from models.config import VLMConfig
from models.language_model import LanguageModelBlock, RotaryEmbedding


B, S, C, DEPTH = 2, 4096, 960, 4
DTYPE, DEVICE = torch.bfloat16, "npu:0"


def make_inputs():
    gen = torch.Generator(device="cpu").manual_seed(1234)
    x = (torch.randn(B, S, C, generator=gen) * 0.02).to(DEVICE, DTYPE).requires_grad_()
    position_ids = torch.arange(S, device=DEVICE).unsqueeze(0).expand(B, -1)
    cfg = VLMConfig()
    rotary = RotaryEmbedding(cfg).to(DEVICE)
    cos, sin = rotary(position_ids)
    mask = torch.ones(B, S, dtype=torch.long, device=DEVICE)
    mask[1, :S // 8] = 0
    return x, cos, sin, mask


def checkpoint_block(block, x, cos, sin, mask, preserve_rng_state):
    out, _ = checkpoint(block, x, cos, sin, mask, None,
                        use_reentrant=False,
                        preserve_rng_state=preserve_rng_state)
    return out


def attention_part(block, x, cos, sin, mask):
    residual = x
    out = block.norm1(x)
    out, _ = block.attn(out, cos, sin, mask, None)
    return residual + out


def mlp_part(block, x):
    return x + block.mlp(block.norm2(x))


def run_stack(blocks, x, cos, sin, mask, method):
    for idx, block in enumerate(blocks):
        if method == "B0_all_checkpoint_rng":
            x = checkpoint_block(block, x, cos, sin, mask, True)
        elif method == "L0a_all_checkpoint_no_rng":
            x = checkpoint_block(block, x, cos, sin, mask, False)
        elif method == "L1a_no_checkpoint":
            x, _ = block(x, cos, sin, mask, None)
        elif method == "L1b_checkpoint_even_blocks":
            if idx % 2 == 0:
                x = checkpoint_block(block, x, cos, sin, mask, False)
            else:
                x, _ = block(x, cos, sin, mask, None)
        elif method == "L1c_checkpoint_one_of_four":
            if idx == 0:
                x = checkpoint_block(block, x, cos, sin, mask, False)
            else:
                x, _ = block(x, cos, sin, mask, None)
        elif method == "L3a_attention_only_checkpoint":
            x = checkpoint(lambda z, block=block: attention_part(block, z, cos, sin, mask),
                           x, use_reentrant=False, preserve_rng_state=False)
            x = mlp_part(block, x)
        elif method == "L3b_mlp_only_checkpoint":
            x = attention_part(block, x, cos, sin, mask)
            x = checkpoint(lambda z, block=block: mlp_part(block, z), x,
                           use_reentrant=False, preserve_rng_state=False)
        else:
            raise ValueError(method)
    return x


METHODS = (
    "B0_all_checkpoint_rng",
    "L0a_all_checkpoint_no_rng",
    "L1a_no_checkpoint",
    "L1b_checkpoint_even_blocks",
    "L1c_checkpoint_one_of_four",
    "L3a_attention_only_checkpoint",
    "L3b_mlp_only_checkpoint",
)


def clear_grads(blocks):
    for block in blocks:
        for param in block.parameters():
            param.grad = None


def execute(blocks, method, capture=False):
    x, cos, sin, mask = make_inputs()
    valid = mask.unsqueeze(-1)
    out = run_stack(blocks, x, cos, sin, mask, method)
    loss = (out.float().square() * valid).sum() / (valid.sum() * C)
    loss.backward()
    torch.npu.synchronize()
    if capture:
        return (out * valid).detach().float().cpu(), x.grad.detach().float().cpu()


def benchmark(blocks, method, warmup=1, reps=3):
    samples = []
    peak = 0
    for idx in range(warmup + reps):
        clear_grads(blocks)
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        torch.npu.synchronize()
        start = time.perf_counter()
        execute(blocks, method)
        elapsed = (time.perf_counter() - start) * 1000
        peak = max(peak, torch.npu.max_memory_allocated() / 2**30)
        if idx >= warmup:
            samples.append(elapsed)
    p95 = sorted(samples)[max(0, math.ceil(.95 * len(samples)) - 1)]
    return statistics.median(samples), p95, peak


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()


def main():
    torch.npu.set_device(0)
    cfg = VLMConfig()
    blocks = torch.nn.ModuleList([LanguageModelBlock(cfg) for _ in range(DEPTH)]).to(DEVICE, DTYPE)
    blocks.train()
    print("=== SN-LM-Checkpoint Supernode Lab ===", flush=True)
    print(f"shape B={B} S={S} C={C} depth={DEPTH} dtype={DTYPE}", flush=True)
    print("L2 skip: compute nodes already use official fused APIs", flush=True)
    print("L4 skip: checkpoint control flow is not a compile fusion scope", flush=True)
    ref_out, ref_grad = execute(blocks, METHODS[0], capture=True)
    for method in METHODS:
        try:
            clear_grads(blocks)
            out, grad = execute(blocks, method, capture=True)
            max_abs = (out - ref_out).abs().max().item()
            grad_cos = cosine(grad, ref_grad)
            median_ms, p95_ms, peak_gib = benchmark(blocks, method)
            print(f"RESULT method={method} median_ms={median_ms:.3f} p95_ms={p95_ms:.3f} peak_gib={peak_gib:.3f} max_abs={max_abs:.6g} grad_cos={grad_cos:.8f}", flush=True)
        except Exception as exc:
            print(f"ERROR method={method} type={type(exc).__name__} message={str(exc)[:1600]}", flush=True)
        finally:
            clear_grads(blocks)
            gc.collect()
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
