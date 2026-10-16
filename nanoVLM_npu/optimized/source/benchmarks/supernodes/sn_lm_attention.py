#!/usr/bin/env python3
"""
SN-LM-Attention Supernode Lab
=============================
GPU fusion groups: representative model__63_forward_243.123 has op3 addmm,
op4 aten._scaled_dot_product_flash_attention (one kernel), and fused surrounding
pointwise/reductions; matching backward graph has fused reductions and GEMMs.
GPU pre-compile: Q/K/V projections, reshape/transpose, GQA expansion, RoPE, SDPA.
GPU post-compile: one attention kernel and fused pointwise chains, bf16 I/O.
NPU current: language_model.py:216-300 materializes K/V 5->15 heads, builds a
Bx1x4096x4096 bool mask in every one of 32 blocks, then calls fused attention.
Identified gaps: native NPU API supports Nq/Nkv GQA directly; current token
bounds describe a full band despite a causal mask; BNSD creates transpose and
contiguous work; same causal mask and no-op dtype conversions are repeated.

Real regime: B=2,S=4096,Nq=15,Nkv=5,D=64 with left padding. The
projection outputs are bf16, but the current float32 RoPE cache promotes Q/K
to float32; B0 therefore casts the repeated V to float32 before attention.
S=1024 diagnoses API failures.  Both moderate and extreme left padding are
tested because causal masking makes the padded query rows fully masked.
All implemented candidates run forward+backward and compare output/gradients.
L0: four API parameter variants. L1: native GQA/cast/mask-reuse variants.
L2: B0 already is the only training fused attention API found.
L3: BSND, BSH, uint8 mask; compressed causal is skipped because per-sample
padding cannot be represented without changing semantics.
L4a-d skipped: installed fused-attention docs say graph mode is unsupported.
L5/L6 skipped unless native-GQA autograd fails.
"""
import gc
import math
import statistics
import time

import torch
import torch_npu
from torch.utils.checkpoint import checkpoint

B, NQ, NKV, D = 2, 15, 5, 64
DTYPE, DEVICE = torch.bfloat16, "npu:0"
SHAPES = {"diagnostic": 1024, "train": 4096}
PAD_PROFILES = ("left_1_8", "left_7_8")
MASK_CACHE = {}


def padding_mask(s, pad_profile):
    mask = torch.ones(B, s, dtype=torch.bool, device=DEVICE)
    if pad_profile == "left_1_8":
        mask[1, :s // 8] = False
    elif pad_profile == "left_7_8":
        mask[1, :(7 * s) // 8] = False
    else:
        raise ValueError(f"unknown padding profile: {pad_profile}")
    return mask


def full_mask(s, pad_profile, cached=False, dtype=torch.bool):
    key = (s, pad_profile, dtype)
    if cached and key in MASK_CACHE:
        return MASK_CACHE[key]
    valid = padding_mask(s, pad_profile)
    mask = ~torch.tril(torch.ones(s, s, dtype=torch.bool, device=DEVICE))
    mask = mask.view(1, 1, s, s).expand(B, -1, -1, -1)
    mask = (mask | (~valid).view(B, 1, 1, s)).contiguous().to(dtype)
    if cached:
        MASK_CACHE[key] = mask
    return mask


def fa(q, k, v, *, repeat=True, cached=False, sparse_mode=0,
       pre=None, next_=None, inner_precise=0, gen_mask_parallel=True,
       layout="BNSD", mask_dtype=torch.bool, do_cast=True,
       contiguous_kv=False, pad_profile="left_1_8"):
    s = q.shape[2]
    if repeat:
        k = k.repeat_interleave(NQ // NKV, dim=1)
        v = v.repeat_interleave(NQ // NKV, dim=1)
    if do_cast:
        k, v = k.to(q.dtype), v.to(q.dtype)
    if contiguous_kv:
        k, v = k.contiguous(), v.contiguous()
    mask = full_mask(s, pad_profile, cached=cached, dtype=mask_dtype)
    if layout == "BSND":
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    elif layout == "BSH":
        q = q.transpose(1, 2).reshape(B, s, NQ * D)
        k = k.transpose(1, 2).reshape(B, s, k.shape[1] * D)
        v = v.transpose(1, 2).reshape(B, s, v.shape[1] * D)
    out = torch_npu.npu_fusion_attention(
        q, k, v, head_num=NQ, input_layout=layout, atten_mask=mask,
        scale=1.0 / math.sqrt(D), keep_prob=1.0,
        pre_tockens=s if pre is None else pre,
        next_tockens=s if next_ is None else next_, inner_precise=inner_precise,
        sparse_mode=sparse_mode, gen_mask_parallel=gen_mask_parallel)[0]
    if layout == "BSND":
        out = out.transpose(1, 2)
    elif layout == "BSH":
        out = out.reshape(B, s, NQ, D).transpose(1, 2)
    return out


METHODS = {
    "B0_control": {},
    "L0a_causal_bounds": {"pre": 4096, "next_": 0},
    "L0b_allmask": {"sparse_mode": 1},
    "L0c_serial_maskgen": {"gen_mask_parallel": False},
    "L0d_invalid_row_precision": {"inner_precise": 2},
    "L0e_causal_inner2": {"pre": 4096, "next_": 0, "inner_precise": 2},
    "L1a_native_gqa": {"repeat": False},
    "L1b_remove_noop_cast": {"do_cast": False},
    "L1c_cached_mask": {"cached": True},
    "L1d_gqa_cached": {"repeat": False, "cached": True},
    "L1e_gqa_cached_causal": {"repeat": False, "cached": True, "pre": 4096, "next_": 0},
    "L1f_gqa_causal": {"repeat": False, "pre": 4096, "next_": 0},
    "L1g_gqa_contiguous": {"repeat": False, "contiguous_kv": True},
    "L1h_gqa_contiguous_causal": {"repeat": False, "contiguous_kv": True, "pre": 4096, "next_": 0},
    "L1i_gqa_inner2": {"repeat": False, "inner_precise": 2},
    "L1j_gqa_contiguous_inner2": {"repeat": False, "contiguous_kv": True, "inner_precise": 2},
    "L1k_gqa_contiguous_causal_inner2": {"repeat": False, "contiguous_kv": True, "pre": 4096, "next_": 0, "inner_precise": 2},
    "L1l_bf16_rope": {"rope_bf16": True, "do_cast": False},
    "L1m_bf16_rope_gqa": {"rope_bf16": True, "repeat": False, "do_cast": False},
    "L1n_bf16_rope_gqa_causal": {"rope_bf16": True, "repeat": False, "do_cast": False, "pre": 4096, "next_": 0},
    "L3a_bsnd_gqa": {"repeat": False, "cached": True, "layout": "BSND"},
    "L3b_bsh_gqa": {"repeat": False, "cached": True, "layout": "BSH"},
    "L3c_uint8_mask": {"repeat": False, "cached": True, "mask_dtype": torch.uint8},
    "L3d_uint8_gqa_causal": {"repeat": False, "cached": True, "mask_dtype": torch.uint8, "pre": 4096, "next_": 0},
}


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def fresh_inputs(s, rope_bf16=False):
    # Generate on CPU with an explicit generator. torch.manual_seed alone did
    # not reset this torch_npu build's device RNG reproducibly at S=4096.
    generator = torch.Generator(device="cpu").manual_seed(1234)
    # Model projections produce BSND views followed by transpose(1, 2), so all
    # three BNSD tensors are non-contiguous. Preserve that real layout here.
    q0 = (torch.randn(B, s, NQ, D, generator=generator) * .1).to(DEVICE, DTYPE).transpose(1, 2)
    k0 = (torch.randn(B, s, NKV, D, generator=generator) * .1).to(DEVICE, DTYPE).transpose(1, 2)
    v = (torch.randn(B, s, NKV, D, generator=generator) * .1).to(DEVICE, DTYPE).transpose(1, 2).requires_grad_()
    cos = torch.randn(B, s, D, generator=generator).to(DEVICE)
    sin = torch.randn(B, s, D, generator=generator).to(DEVICE)
    if rope_bf16:
        cos, sin = cos.to(DTYPE), sin.to(DTYPE)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q = ((q0 * cos) + (rotate_half(q0) * sin)).detach().requires_grad_()
    k = ((k0 * cos) + (rotate_half(k0) * sin)).detach().requires_grad_()
    return q, k, v


def apply_attention(q, k, v, kwargs, use_checkpoint=False):
    if use_checkpoint:
        return checkpoint(lambda q_, k_, v_: fa(q_, k_, v_, **kwargs),
                          q, k, v, use_reentrant=False)
    return fa(q, k, v, **kwargs)


def run_once(s, kwargs, use_checkpoint=False):
    call_kwargs = dict(kwargs)
    rope_bf16 = call_kwargs.pop("rope_bf16", False)
    q, k, v = fresh_inputs(s, rope_bf16=rope_bf16)
    out = apply_attention(q, k, v, call_kwargs, use_checkpoint)
    grad_gen = torch.Generator(device="cpu").manual_seed(5678)
    grad = torch.randn(out.shape, generator=grad_gen).to(DEVICE, DTYPE)
    # The collator labels left-padding positions with -100, so no loss gradient
    # reaches those query rows. Comparing random gradients on invalid rows gives
    # a false precision failure for kernels that intentionally special-case
    # fully masked rows (for example inner_precise=2).
    valid_q = padding_mask(s, call_kwargs["pad_profile"]).view(B, 1, s, 1)
    grad = grad * valid_q
    out.backward(grad)
    torch.npu.synchronize()
    # Clone to host so the NPU caching allocator cannot reuse and overwrite the
    # reference buffers while later candidates run.
    checked_out = (out * valid_q).detach().float().cpu()
    return checked_out, tuple(x.grad.detach().float().cpu() for x in (q, k, v))


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def benchmark(s, kwargs, warmup=2, reps=5, use_checkpoint=False):
    call_kwargs = dict(kwargs)
    rope_bf16 = call_kwargs.pop("rope_bf16", False)
    q, k, v = fresh_inputs(s, rope_bf16=rope_bf16)
    times = []
    torch.npu.reset_peak_memory_stats()
    for idx in range(warmup + reps):
        for x in (q, k, v):
            x.grad = None
        torch.npu.synchronize()
        start = time.perf_counter()
        out = apply_attention(q, k, v, call_kwargs, use_checkpoint)
        valid_q = padding_mask(s, call_kwargs["pad_profile"]).view(B, 1, s, 1)
        (out.float().square() * valid_q).mean().backward()
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if idx >= warmup:
            times.append(elapsed)
    p95 = sorted(times)[max(0, math.ceil(.95 * len(times)) - 1)]
    return statistics.median(times), p95, torch.npu.max_memory_allocated() / 2**30


def main():
    torch.npu.set_device(0)
    print("=== SN-LM-Attention Supernode Lab ===", flush=True)
    print("GPU compiled: 1 SDPA kernel, native GQA semantics, bf16", flush=True)
    print("NPU B0: float32 Q/K/V attention after RoPE promotion + repeat K/V 5->15 + full mask + 1 FA kernel", flush=True)
    print("L2 skip: B0 is official training fused API; inference APIs excluded", flush=True)
    print("L3d skip: compressed causal cannot encode per-sample padding", flush=True)
    print("L4a-d skip: fused-attention training API does not support graph mode", flush=True)
    for label, s in SHAPES.items():
        for pad_profile in PAD_PROFILES:
            print(f"--- shape={label} padding={pad_profile} B={B} S={s} Nq={NQ} Nkv={NKV} D={D} ---", flush=True)
            ref_kwargs = dict(METHODS["B0_control"], pad_profile=pad_profile)
            ref_out, ref_grads = run_once(s, ref_kwargs)
            for name, original in METHODS.items():
                kwargs = dict(original, pad_profile=pad_profile)
                if "pre" in kwargs:
                    kwargs["pre"] = s
                try:
                    out, grads = run_once(s, kwargs)
                    diff = (out.float() - ref_out.float()).abs().max().item()
                    grad_cosines = tuple(cosine(a, b) for a, b in zip(grads, ref_grads))
                    grad_cos = min(grad_cosines)
                    med, p95, peak = benchmark(s, kwargs)
                    print(f"RESULT method={name} S={s} padding={pad_profile} median_ms={med:.3f} p95_ms={p95:.3f} peak_gib={peak:.3f} max_abs={diff:.6g} grad_cos={grad_cos:.8f} grad_cos_qkv={grad_cosines}", flush=True)
                except Exception as exc:
                    print(f"ERROR method={name} S={s} padding={pad_profile} type={type(exc).__name__} message={str(exc)[:1200]}", flush=True)
                finally:
                    gc.collect()
                    torch.npu.empty_cache()

    # The real model checkpoints every LM block. Re-run the candidates that
    # can plausibly win or explain error 161002 under the exact non-reentrant
    # recomputation mode before any further 16-NPU submission.
    s = SHAPES["train"]
    checkpoint_methods = (
        "B0_control", "L0a_causal_bounds", "L0e_causal_inner2",
        "L1a_native_gqa", "L1g_gqa_contiguous",
        "L1h_gqa_contiguous_causal", "L1k_gqa_contiguous_causal_inner2",
        "L1l_bf16_rope", "L1m_bf16_rope_gqa", "L1n_bf16_rope_gqa_causal",
    )
    for pad_profile in PAD_PROFILES:
        print(f"--- checkpoint=True shape=train padding={pad_profile} S={s} ---", flush=True)
        ref_kwargs = dict(METHODS["B0_control"], pad_profile=pad_profile)
        ref_out, ref_grads = run_once(s, ref_kwargs, use_checkpoint=True)
        for name in checkpoint_methods:
            kwargs = dict(METHODS[name], pad_profile=pad_profile)
            if "pre" in kwargs:
                kwargs["pre"] = s
            try:
                out, grads = run_once(s, kwargs, use_checkpoint=True)
                diff = (out.float() - ref_out.float()).abs().max().item()
                grad_cosines = tuple(cosine(a, b) for a, b in zip(grads, ref_grads))
                med, p95, peak = benchmark(s, kwargs, warmup=1, reps=3, use_checkpoint=True)
                print(f"CHECKPOINT method={name} S={s} padding={pad_profile} median_ms={med:.3f} p95_ms={p95:.3f} peak_gib={peak:.3f} max_abs={diff:.6g} grad_cos={min(grad_cosines):.8f} grad_cos_qkv={grad_cosines}", flush=True)
            except Exception as exc:
                print(f"CHECKPOINT_ERROR method={name} S={s} padding={pad_profile} type={type(exc).__name__} message={str(exc)[:1200]}", flush=True)
            finally:
                gc.collect()
                torch.npu.empty_cache()


if __name__ == "__main__":
    main()
