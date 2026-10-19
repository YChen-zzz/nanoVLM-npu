# Stage 1 optimization progress

- Target: `BENCH_MEASURE elapsed_s` about 500s; `FINAL_VAL_LOSS` 1.08 +/- 0.01.
- Baseline: 923.607710s, val loss 1.08289274, commit `cc423df`.
- Current best 16-NPU short run: 70.168917s, train loss 1.20413574.
- Stage 1 cumulative full run: 563.833686s, final val loss 1.08506393.
- Stage 1 status: complete; all high-priority SN L0-L4 cells are closed.
- Next phase: Stage 2 profiling-driven optimization from commit `6d54f53`.
- GPU Teacher: `../gpu_teacher_data/nanovlm_compile_20260828_105147`.

| SN | Priority | L0 | L1 | L2 | L3 | L4 | L5 | L6 | Winner | Ablation | Status |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| LM-Attention | high | ✅ L0a causal bounds: 116.154515s→102.354690s (-11.88%) | ✅ GQA 111.811738s; bf16 RoPE 97.604360s ✅cond; cumulative L1q 90.397127s ✅cond | ✅skip: B0 is only training fused API; inference APIs change semantics | ✅ 4 variants: BSND/BSH/uint8/contiguous, no independent win | ✅skip: FA training API explicitly has no graph mode (4 scopes inapplicable) | ✅skip: native autograd passed | ✅skip: official fused kernel sufficient | L1q causal+bf16 RoPE+GQA | ✅cond 90.397127s (-22.17%) | done |
| LM-Checkpoint | high | ✅ RNG-off no gain; unsupported max_split pathological; expandable segments fixed OOM | ✅ no-checkpoint+expandable 75.570058s; every4 78.824542s; every8 77.173539s | ✅skip: compute nodes use official fused APIs | ✅ attention-only/MLP-only slower; partial fallback recorded | ✅skip: checkpoint control flow not a compile fusion scope | ✅skip: native autograd exact | ✅skip: no custom kernel needed | no checkpoint + expandable segments | ✅pass 75.570058s (-34.94% vs original) | done |
| LM-RMS-Residual | high | ✅skip: epsilon fixed; no semantic-free API knob | ✅ 6 exact expressions; `x*x` 75.202440s | ❌ `npu_rms_norm` 72.841718s but train loss 2.08481742; raw add-RMS has no autograd | ✅ torch.add/Tensor.add exact, no stable Lab gain | ❌ 4 `backend=npu` scopes slower than eager | ✅ L5b dtype-aligned add-RMS custom backward: 74.289084s | ✅skip: official fused kernel already used | L5c = L1a+L5b | ✅pass 73.682681s (-2.50% vs 75.570058s) | done |
| LM-SwiGLU | high | ✅skip: dim is semantic; no free API/env knob | ✅ 4 SiLU/mul spellings, no gain | ✅ native SwiGLU no gain; GMM custom backward 3.2532ms | ✅ L3a concat projection + native SwiGLU 2.8220ms | ❌ 4 `backend=npu` scopes slower | ✅ GMM custom backward exact but slower | ✅skip: official fused kernel available | L3e cumulative concat projection + native SwiGLU | ✅cond 72.429232s (-1.70%) | done |
| Output-Loss | medium | ✅ 4 equivalent CE parameter/API spellings; no gain | ✅ L1c valid-row index_select 16.9087ms vs 28.1740ms | ✅cond L2i filtered native CE 12.9400ms; full ignored-row API crashes | ✅ 4 layouts; linear fastest | ❌ 3 scopes 28.00-28.86ms; filtered scope dynamic-shape unsupported | ✅skip: official native backward passed | ✅skip: official fused loss sufficient | L2i index-select valid rows + native CE | ✅cond 70.168917s (-3.12%) | done |

GPU/NPU gap: GPU uses fused SDPA with native GQA semantics and fused surrounding
pointwise chains. Current NPU source expands K/V 3x, rebuilds a full mask in all
32 LM blocks, and checkpoints all blocks. The first active SN is LM-Attention.

## LM-Attention evidence

- Lab r8 (`logs/sn_lm_attention_r8.log`) uses the real fixed `S=4096`, bf16,
  non-contiguous BNSD inputs, left padding at 1/8 and 7/8, valid-token-only
  loss gradients, and `checkpoint(..., use_reentrant=False)`.
- Checkpoint B0: 8.55-8.62 ms; causal bounds: 5.92-6.02 ms (~30% faster),
  max valid-output diff 6.1e-5, minimum Q/K/V grad cosine >1.00005.
- Native GQA is bit-exact on valid outputs and reduces peak lab allocation.
  The earlier 161002 root cause was a mistakenly removed cast: real RoPE makes
  Q/K float32 while V stays bf16. Keeping the cast passed the exact model/data/
  checkpoint probe; corrected 16-NPU ablation 20849 is queued.
- `inner_precise=2` changes only invalid padded query rows when the training
  loss mask is respected, but is slower than the corresponding default mode.
- L0a causal bounds passed 16-NPU ablation: 116.154515s -> 102.354690s
  (-11.88%); train loss 1.20287657 vs baseline 1.20954577.
- GPU-aligned bf16 RoPE is conditional-pass in Lab: valid-output max diff
  9.66e-4, gradient cosine ~1.000, checkpoint attention 13.96 -> 8.25 ms.
- 16-NPU short results: native GQA 111.811738s; causal+GQA 98.068233s;
  bf16 RoPE 97.604360s; causal+bf16 RoPE 91.720968s. The last result is
  reproduced at 92.113266s with train loss 1.19912248, showing the first run's
  1.27214152 was run/data variation rather than a stable regression.
- Final cumulative L1q (causal + bf16 RoPE + native GQA): 90.397127s,
  train loss 1.20309860, -22.17% versus the 116.154515s short baseline. It is
  merged as `✅cond` because bf16 RoPE has max valid-output diff 9.66e-4; the
  final full run must satisfy `FINAL_VAL_LOSS=1.08+/-0.01` or this is reverted.

## LM-Checkpoint evidence

- Lab r1 (`logs/sn_lm_checkpoint_r1.log`) uses four complete optimized LM
  blocks at B=2,S=4096,C=960,bf16 and checks valid-token output/input gradient.
- All methods are numerically exact. Full checkpoint: 139.434ms, 0.590GiB;
  no checkpoint: 129.668ms, 1.620GiB; preserve_rng_state=False: 142.419ms.
- Alternating-block, one-of-four, attention-only, and MLP-only checkpointing
  were slower than no checkpoint and use intermediate memory amounts.
- Exact full-model 2-NPU DDP probe completed two batch_size=2 training steps.
- Plain no-checkpoint reached micro-step 62 on 16 NPU then OOMed from allocator
  fragmentation (56.87GiB allocated, 60.11GiB reserved, 314MiB free).
- `max_split_size_mb=128` is unsupported by this torch_npu allocator and caused
  3.8-21s micro-steps. The supported `expandable_segments:True` completed:
  75.570058s, train loss 1.19811538. Every-4 and every-8 checkpoint fallbacks
  completed at 78.824542s and 77.173539s respectively.

## LM-RMS-Residual evidence

- Lab r4/r5 covers B=2, S={1024,4096}, C=960 and the real autocast boundary:
  fp32 residual/weight plus bf16 attention output. Six exact RMS expressions,
  two residual-add spellings, two official fused APIs, four NPU compile scopes,
  and two custom-autograd variants were tested with forward and backward.
- Raw `npu_add_rms_norm` has no registered autograd. Its first model probe also
  exposed the real mixed-dtype constraint: x1/gamma fp32 and x2 bf16 are not a
  supported signature (161002). L5b explicitly performs the same promotion as
  eager add, then calls the fused API and uses `npu_rms_norm_backward`.
- On the real mixed boundary at S=4096, L5b has max output diff 1.91e-6 and
  gradient cosine 1.00000000. The two-NPU model probe completed two optimizer
  updates; 16-NPU ablation completed in 74.289084s with train loss 1.20020396.
- Standalone `npu_rms_norm` was faster (72.841718s) but failed the conditional
  accuracy observation with train loss 2.08481742 and is rejected. All four
  compile scopes were slower than eager at S=4096.
- Exact L1a (`x*x`) completed in 75.202440s. Its cumulative L5c combination
  with L5b completed in 73.682681s with train loss 1.22192336 and is the final
  RMS winner.

## LM-SwiGLU evidence

- GPU forward `model__18_forward_49.18` keeps the three GEMMs external and
  fuses SiLU, gate/up multiply, and casts into one pointwise kernel; backward
  `model__56_backward_185.67` similarly fuses the activation derivative chain.
- The Lab covers B=2, S={1024,4096}, H=960, I=2560 with forward+backward.
  Four eager spellings and native SwiGLU with two projections showed no gain;
  all four `backend=npu` scopes were slower. GMM required explicit BF16 and a
  custom backward, then measured 3.2532ms versus B0 3.2961ms.
- L3a concatenates the two checkpoint-compatible projection weights at runtime,
  issues one projection, then calls `npu_swiglu`: 2.8220ms (-14.4% in Lab).
  Output max diff is 0.015625 and worst grad cosine is 0.999984741, so it is
  conditional rather than strict. A two-update 2-NPU model probe passed.
- Independent L3a short run was 73.099600s with noisy train loss 1.25545527.
  The cumulative L3e rerun on RMS L5c completed at 72.429232s with train loss
  1.21468484, confirming speed and normal short-run loss. It is accepted as
  `✅cond`; final full loss must remain in 1.08 +/- 0.01.

## Output-Loss evidence

- GPU `model__61_forward_178.61` uses hidden `[2,4096,960]`, vocab 49218 and
  logits `[8192,49218]`: the head GEMM remains external while CE pointwise and
  reductions are compiler-scheduled. Training consumes only loss and ignores
  returned logits; about 57% of target rows are valid.
- The Lab covers B=2, S={1024,4096}, H=960, V=49218 with forward+backward.
  B0 is 28.1740ms at S=4096. Filtering ignored rows before the tied head with
  `nonzero+index_select` is 16.9087ms, with identical loss/hidden gradient and
  weight-grad cosine 1.0.
- Full-row `npu_cross_entropy_loss` triggers an AICore/MTE error when targets
  contain -100, reproduced in BF16, FP32, and padded-vocab repairs. After valid
  row filtering the public and op APIs pass; cumulative L2i is 12.9400ms.
  Against eager CE its loss relative difference is about 5e-4 and minimum grad
  cosine is 0.99999988, so it remains conditional.
- Four projection layouts were tested; linear is fastest. CE-only, head-only,
  and full `backend=npu` scopes are 28.00-28.86ms; filtered full compile is
  rejected because `nonzero` has data-dependent output shape. A two-update
  2-NPU model probe passed. The 16-NPU cumulative ablation completed in
  70.168917s with train loss 1.20413574, versus 72.429232s before this SN.
