#!/usr/bin/env python3
"""
SN-LM-RMS-Residual Supernode Lab
================================
GPU fusion groups: repeated FusedSchedulerNode groups fuse RMS square/mean/
rsqrt/scale and adjacent residual work; backward fuses reductions/grad adds.
GPU target: one fused pointwise/reduction kernel per RMS chain and fused
residual+next-RMS where graph scheduling permits, bf16 input/output.
NPU current: models/language_model.py runs eager pow/mean/add/rsqrt/mul/mul
twice in each of 32 blocks and launches residual adds separately.
Identified gaps: official npu_rms_norm replaces the RMS chain; official
npu_add_rms_norm can fuse an attention residual add with the following norm;
pow versus square/multiply are alternative eager expressions. Epsilon is fixed
at 1e-5 to preserve training dynamics.

Real shapes: B=2,S={1024,4096},C=960. The Lab covers both all-bf16 and the
actual autocast boundary (fp32 residual/weight + bf16 attention output).
All methods include backward.
L0: fixed epsilon/API defaults (no legal dynamics-changing tuning).
L1: six pow/square/multiply eager spellings are tested independently.
L2: npu_rms_norm and npu_add_rms_norm.
L3: torch.add and Tensor.add residual spellings are tested; in-place add is
    skipped because the pre-attention residual remains live in autograd.
L4: four NPU compile scopes cover RMS pow/mul and residual+RMS pow/mul. They
    deliberately stop before attention, whose training API rejects graph mode.
L5/L6: skipped when native fused backward passes strict correctness.
"""
import gc
import math
import statistics
import time

import torch
import torch_npu


B, C, EPS = 2, 960, 1e-5
DTYPE, DEVICE = torch.bfloat16, "npu:0"
SHAPES = (1024, 4096)


def rms_pow(x, weight):
    return x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + EPS) * weight


def rms_mul(x, weight):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + EPS) * weight


def rms_square(x, weight):
    return x * torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + EPS) * weight


def rms_mul_method(x, weight):
    return x * torch.rsqrt(torch.mean(x.mul(x), dim=-1, keepdim=True) + EPS) * weight


def rms_square_method(x, weight):
    return x * torch.rsqrt(torch.mean(x.square(), dim=-1, keepdim=True) + EPS) * weight


def rms_pow_function(x, weight):
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + EPS) * weight


def rms_pow_method(x, weight):
    return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + EPS) * weight


def rms_native(x, weight):
    return torch_npu.npu_rms_norm(x, weight, epsilon=EPS)[0]


L4a_compile_rms_pow = torch.compile(
    rms_pow, backend="npu", dynamic=False, fullgraph=True)
L4b_compile_rms_mul = torch.compile(
    rms_mul, backend="npu", dynamic=False, fullgraph=True)


RMS_METHODS = {
    "B0_pow": rms_pow,
    "L1a_mul": rms_mul,
    "L1b_square": rms_square,
    "L1c_mul_method": rms_mul_method,
    "L1d_square_method": rms_square_method,
    "L1e_pow_function": rms_pow_function,
    "L1f_pow_method": rms_pow_method,
    "L2a_npu_rms_norm": rms_native,
    "L4a_compile_rms_pow": L4a_compile_rms_pow,
    "L4b_compile_rms_mul": L4b_compile_rms_mul,
}


def add_rms_separate(x1, x2, weight):
    summed = x1 + x2
    return rms_pow(summed, weight), summed


def add_rms_native(x1, x2, weight):
    y, _, summed = torch_npu.npu_add_rms_norm(x1, x2, weight, epsilon=EPS)
    return y, summed


def add_rms_torch_add(x1, x2, weight):
    summed = torch.add(x1, x2)
    return rms_pow(summed, weight), summed


def add_rms_method(x1, x2, weight):
    summed = x1.add(x2)
    return rms_pow(summed, weight), summed


def add_rms_mul(x1, x2, weight):
    summed = x1 + x2
    return rms_mul(summed, weight), summed


L4c_compile_add_rms_pow = torch.compile(
    add_rms_separate, backend="npu", dynamic=False, fullgraph=True)
L4d_compile_add_rms_mul = torch.compile(
    add_rms_mul, backend="npu", dynamic=False, fullgraph=True)


class AddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, weight):
        y, rstd, summed = torch_npu.npu_add_rms_norm(
            x1, x2, weight, epsilon=EPS)
        ctx.save_for_backward(summed, weight, rstd)
        return y, summed

    @staticmethod
    def backward(ctx, grad_y, grad_summed):
        summed, weight, rstd = ctx.saved_tensors
        grad_from_norm, grad_weight = torch_npu.npu_rms_norm_backward(
            grad_y, summed, weight, rstd)
        grad_total = grad_from_norm
        if grad_summed is not None:
            grad_total = grad_total + grad_summed
        return grad_total, grad_total, grad_weight


def add_rms_custom_autograd(x1, x2, weight):
    return AddRMSNormFunction.apply(x1, x2, weight)


def add_rms_custom_autograd_aligned(x1, x2, weight):
    return AddRMSNormFunction.apply(
        x1, x2.to(x1.dtype), weight.to(x1.dtype)
    )


ADD_METHODS = {
    "B0_add_then_pow_rms": add_rms_separate,
    "L3a_torch_add_then_pow_rms": add_rms_torch_add,
    "L3b_method_add_then_pow_rms": add_rms_method,
    "L4c_compile_add_rms_pow": L4c_compile_add_rms_pow,
    "L4d_compile_add_rms_mul": L4d_compile_add_rms_mul,
    "L2b_npu_add_rms_norm": add_rms_native,
    "L5a_npu_add_rms_norm_custom_backward": add_rms_custom_autograd,
    "L5b_custom_backward_dtype_aligned": add_rms_custom_autograd_aligned,
}


def make_inputs(s, residual=False, mixed=False):
    gen = torch.Generator(device="cpu").manual_seed(1234)
    x1_dtype = torch.float32 if mixed else DTYPE
    x1 = torch.randn(B, s, C, generator=gen).to(DEVICE, x1_dtype).requires_grad_()
    weight = torch.randn(C, generator=gen).to(DEVICE, x1_dtype).requires_grad_()
    if not residual:
        return (x1, weight)
    x2 = torch.randn(B, s, C, generator=gen).to(DEVICE, DTYPE).requires_grad_()
    return x1, x2, weight


def capture(fn, s, residual=False, mixed=False):
    inputs = make_inputs(s, residual, mixed)
    outputs = fn(*inputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    gen = torch.Generator(device="cpu").manual_seed(5678)
    loss = 0.0
    for out in outputs:
        grad = torch.randn(out.shape, generator=gen).to(DEVICE, out.dtype)
        loss = loss + (out * grad).float().sum()
    loss.backward()
    torch.npu.synchronize()
    return (tuple(out.detach().float().cpu() for out in outputs),
            tuple(inp.grad.detach().float().cpu() for inp in inputs))


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()


def benchmark(fn, s, residual=False, mixed=False, warmup=3, reps=7):
    samples = []
    peak = 0
    for idx in range(warmup + reps):
        inputs = make_inputs(s, residual, mixed)
        torch.npu.reset_peak_memory_stats()
        torch.npu.synchronize()
        start = time.perf_counter()
        outputs = fn(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        sum(out.float().square().mean() for out in outputs).backward()
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        peak = max(peak, torch.npu.max_memory_allocated() / 2**30)
        if idx >= warmup:
            samples.append(elapsed)
    p95 = sorted(samples)[max(0, math.ceil(.95 * len(samples)) - 1)]
    return statistics.median(samples), p95, peak


def run_family(title, methods, s, residual, mixed=False):
    print(f"--- {title} B={B} S={s} C={C} dtype={DTYPE} ---", flush=True)
    ref_outs, ref_grads = capture(next(iter(methods.values())), s, residual, mixed)
    for name, fn in methods.items():
        try:
            outs, grads = capture(fn, s, residual, mixed)
            max_abs = max((a - b).abs().max().item() for a, b in zip(outs, ref_outs))
            grad_cos = min(cosine(a, b) for a, b in zip(grads, ref_grads))
            median_ms, p95_ms, peak_gib = benchmark(fn, s, residual, mixed)
            print(f"RESULT family={title} method={name} S={s} median_ms={median_ms:.3f} p95_ms={p95_ms:.3f} peak_gib={peak_gib:.3f} max_abs={max_abs:.6g} grad_cos={grad_cos:.8f}", flush=True)
        except Exception as exc:
            print(f"ERROR family={title} method={name} S={s} type={type(exc).__name__} message={str(exc)[:1200]}", flush=True)
        finally:
            gc.collect()
            torch.npu.empty_cache()


def main():
    torch.npu.set_device(0)
    print("=== SN-LM-RMS-Residual Supernode Lab ===", flush=True)
    print("GPU target: fused RMS chain and fused residual+RMS; bf16 I/O", flush=True)
    print("L0 skip: epsilon changes training dynamics; no other semantic-free API knobs", flush=True)
    print("L4: four backend=npu scopes; attention remains outside graph", flush=True)
    for s in SHAPES:
        run_family("rms", RMS_METHODS, s, residual=False)
        run_family("add_rms", ADD_METHODS, s, residual=True)
        run_family("mixed_fp32_bf16_add_rms", ADD_METHODS, s,
                   residual=True, mixed=True)


if __name__ == "__main__":
    main()
