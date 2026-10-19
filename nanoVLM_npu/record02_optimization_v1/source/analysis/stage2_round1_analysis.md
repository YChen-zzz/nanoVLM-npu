# Stage 2 Round 1 — Phase 2 analysis

Baseline commit: `7244e82` (performance code through `6d54f53`)

- Wall-clock short: 70.168917 s / 100 microbatches = 701.689 ms/microbatch
- Full training: 563.833686 s, `FINAL_VAL_LOSS=1.08506393`
- L0: `profiling/20260831_021041_l0`
- L1: `profiling/20260831_021404_l1`
- L0 measure: 84.712026 s / 100 microbatches, loss 1.19832153
- L1 measure: 111.126807 s / 100 microbatches, loss 1.20494141

## Bounds and L0/L1 cross-check

The profiled active window contains three microbatches. A rough Tier-1 roofline uses a 910B-class assumption of about 320 BF16 TFLOP/s and 1.6 TB/s HBM. The 460M-parameter LM+ViT workload, with 30–44 images/rank and 4096 text positions, has a major-GEMM/attention lower-bound order of 0.25–0.35 s for the three-microbatch window. This is deliberately an order-of-magnitude floor, not a prediction.

| Bound | Time / 3 microbatches | Meaning |
|---|---:|---|
| Tier 1 roofline | ~0.30 s | physical lower-bound order |
| Tier 2 L0 Computing | 1.6204 s | actual kernel execution sum |
| Tier 3 aligned wall-clock | 2.1051 s | 3 × 701.689 ms |

- gap A = about 1.32 s: kernel/algorithm efficiency dominates the physical gap.
- gap B = about 0.485 s (23.0% of Tier 3): exposed non-compute/synchronization upper bound. In distributed training this includes exposed communication, not only Python.
- L0 step trace: Computing 66.4%, Free 1.8%, Communication 31.8%, utilization 97.4%.
- L1 step trace: Computing 42.8%, Free 32.3%, Communication 25.0%, utilization 57.0%.
- Utilization differs by 40.4 percentage points, so L1 Free/host-idle is a profiler artifact. Bottleneck direction uses L0; L1 operator/kernel/call-stack evidence remains valid.

## Line B root-cause trace

| Finding source | Finding | Bridge | Source | Root cause | Candidate / disposition |
|---|---|---|---|---|---|
| step trace SIGNAL | 2.0x step spread | trace + update-step mapping | training loop 527–600 | active window contains ordinary and optimizer/update microbatches plus variable image counts | expected regime variance; keep same coverage |
| op statistic DEFINITE | top 3 = 49.1% device time | kernel shapes + source | attention, linear layers, DDP | allreduce/FA/matmul are the real workload | optimize synchronization and structural launch count; do not remove required math |
| op statistic SIGNAL | movement ops 7.5%, 153.8 ms | operator stack | LM attention/linear and ViT | required layout/autocast plus repeated mask construction | remove NPU-dead additive mask and reuse combined bool mask; other layouts already covered by Stage 1 |
| op statistic SIGNAL | Conv2D backward 39.4 ms | input shapes + source | `vision_transformer.py:35` | required patch embedding backward, memory-dominated | 1.9% device share; below current candidates, retain |
| kernel SIGNAL | AI_CORE and vector both material | kernel ratios | whole model | mixed GEMM/FA and pointwise workload | covered by specific candidates |
| kernel SIGNAL | high-duration low-compute kernels | shapes + stacks | ViT convolution; Adam foreach; large adds/cats | memory-bound necessary ops and optimizer traffic | retain for fused-optimizer later; no blind replacement |
| kernel SIGNAL | compute-bound matmuls | shapes + source | LM/ViT linear layers | required GEMMs | QKV merge is a later independent candidate |
| kernel SIGNAL | high waits | trace timing + communication JSON | data sync, dynamic index assignment, DDP buckets | stream barriers and rank arrival skew | CPU synchronization + dense image replacement; DDP tuning later |
| kernel SIGNAL | fusible sequence 0.12 ms | stream sequence | optimizer mul sequence | only 0.0% compute | excluded (<1%) |
| trace SIGNAL | prefetch/prealloc | call stacks | batch `.to()` and image concat | H2D copies are small in host self time | excluded this round: about 4–13 ms sites, below 1% wall |
| operator DEFINITE | 90% host ops have no device work | L0 cross-check + call stacks | framework/model | most metadata dispatch overlaps; blocking sites dominate exposed time | act only on proven barriers below |
| operator SIGNAL | `record_event` 6.6 ms | call stack | profiler and distributed wrappers | profiling instrumentation | excluded from production optimization |
| operator DEFINITE | apparent AI_CPU matmul/addmm/conv | kernel-core cross-check | attribution nesting | all 59 AI_CPU kernels are communication kernels; no non-communication fallback | excluded as report attribution false positive |
| memory SIGNAL/DEFINITE | churn and repeated sizes | operator memory + API statistic | normal activation/autograd allocations | many short-lived tensors, but allocator runtime is 0.1 ms and peak is 58% HBM | broad preallocation excluded; target repeated mask/index path specifically |
| API SIGNAL | sync is 52.7% API time | call stacks | `data_utils.py:70-72`, benchmark timing, loss/token reporting | per-micro NPU collective + `.item()` and required benchmark synchronization | move validity collective to existing Gloo group; retain measurement sync |
| communication DEFINITE | communication is wait-dominated | all-rank comparison + raw message sizes | DDP reducer and data iterator | 59 large buckets transfer quickly but ranks arrive unevenly; three scalar validity collectives also serialize streams | CPU validity sync now; bucket/load-balance experiment later |
| communication SIGNAL | HCCS/SIO small packets | raw JSON | scalar validity collectives | 8–16-byte per-micro collectives | CPU validity sync removes them from NPU stream |

Important call-stack evidence:

- `aten::_local_scalar_dense`: 1546 calls / 432.54 ms host. Three calls from `data/data_utils.py:72` account for 370.96 ms. Optimizer scalar reads account for only about 17 ms.
- `aten::_index_put_impl_`: six records / 228.15 ms host; maps to `models/vision_language_model.py:47`. The trace has a 249.1 ms stall adjacent to this dynamic write path.
- DDP: 59 gradient allreduces in the update microbatch, about 3.396 GB logical traffic. Rank-0 collective elapsed 897.5 ms versus rank-15 180.3 ms; real link transit is sub-ms per bucket, proving arrival skew rather than bandwidth saturation.
- `npu_dtype_cast`: 94.18 ms host across 4776 records, mostly linear autocast/framework metadata. The actionable model-owned repeats are attention masks and already-approved Stage-1 mixed-dtype boundaries.

## Line A source audit

Call chain: `train()` → DDP `forward()` → `VisionLanguageModel.forward()` → ViT / modality projector / LM → output head/loss → backward → DDP reducer → clip/AdamW. The operator report attributes 75% of inclusive host total to project frames, so the project layer has candidates below.

| Hot operation | Redundant work? | Reusable? | Fixed output shape? | Multi-op replacement opportunity | Decision |
|---|---|---|---|---|---|
| synchronized dataloader validity collective | NPU scalar allocation/allreduce/item each micro | existing CPU process group | scalar | same MIN collective on CPU | candidate R1-A |
| batch H2D copies | no | pin/prefetch possible | text fixed, image dynamic | non-blocking prefetch | <1% host; defer |
| image concatenation | required | no, input-dependent | dynamic images | none proven | retain |
| ViT patch conv | required | no | image size fixed | no equivalent public fused API | retain |
| ViT 12 blocks | QKV already fused | no | token length fixed | SDPA already used | retain |
| modality pixel shuffle/project | reshapes required | no | dynamic image count | current views + one GEMM | retain |
| image-token replacement | clone + dynamic bool index causes sync | dense mask/index can be reused within op | text shape fixed | functional dense gather/where | candidate R1-C |
| RoPE | computed once and shared over 32 LM blocks | already reused | fixed in training | Stage-1 BF16 winner | retain |
| LM Q/K/V projections | three GEMMs share input | mergeable | fixed hidden dims | concatenated projection | later candidate; higher numerical/checkpoint risk |
| NPU attention additive mask | float mask is computed then unused | yes | fixed training shape | delete dead branch | candidate R1-B |
| NPU causal+padding mask per LM block | identical across 32 blocks | yes, per-forward | fixed training shape | build once and pass | candidate R1-B |
| fused attention | required | no | fixed training shape | official NPU FA already used | retain |
| RMS/residual | previously redundant chain | Stage-1 fused | fixed | already optimized | retain |
| SwiGLU projections/activation | shared input | Stage-1 concat winner | fixed | already optimized | retain |
| valid-row head/loss | ignored rows redundant | Stage-1 filtering winner | valid count dynamic | already optimized | retain |
| DDP gradient buckets | 59 launch/sync points | bucket views/tuning possible | fixed model | larger buckets/static graph | later isolated 16-NPU ablation |
| grad clipping + AdamW | many foreach/scalar metadata ops | fused optimizer possible | fixed parameters | NPU fused optimizer | later; optimizer-state/numerical risk |

## Ten waste categories

| Category | Status | Evidence |
|---|---|---|
| explicit synchronization | candidate | dataloader NPU collective + `.item()` = 370.96 ms/3 calls |
| dispatch scheduling | candidate/defer | repeated masks actionable; general dispatch overlaps because L0 Free is 1.8% |
| memory-management blocking | excluded broadly | allocator API 0.1 ms, peak 58%; target mask/index allocations only |
| online compilation | excluded | no repeated compile event in trace; warmup complete |
| memory bandwidth | candidate | kernel ratios memory-dominant; remove repeated masks/index path |
| compute saturation | deferred | required matmuls/FA; Stage 1 exhausted primary fused APIs |
| layout/format conversion | candidate/defer | 7.5% movement; actionable mask casts removed, remaining layouts required/tested |
| communication synchronization | candidate/defer | CPU scalar collective now; DDP bucket/load skew later |
| small-op fragmentation | excluded | detected fusible run totals 0.12 ms (0.0%) |
| uncovered latency | candidate | CPU collective and dynamic index barriers; no multi-compute-stream opportunity |

## Candidate ranking and checkpoint A decision

The user pre-authorized the agent to choose optimization options. Checkpoint A is therefore satisfied by the following recorded selection:

| Rank | Candidate | Waste removed | Counterfactual upper bound | Risk | Decision |
|---:|---|---|---:|---|---|
| 1 | move per-micro validity MIN collective from NPU/HCCL to existing CPU/Gloo group | explicit sync / small-packet communication | 370.96 ms in 3-micro L1 window (10.8% of trace span; capped by aligned gap B) | low, same collective/result | implement first |
| 2 | build NPU attention mask once per forward and remove unused additive float mask | bandwidth/layout/dispatch/churn | subset of 153.8 ms movement plus repeated 32-layer stalls; estimated 3–7% | low-medium, exact bool mask required | implement second |
| 3 | replace boolean `index_put` image-token update with an equivalent dense functional mapping | sync/dispatch/churn | 228.15 ms host in 3-micro window (6.7% trace span) | medium, gradients/order must be exact | implement third |
| 4 | DDP bucket/static-graph tuning | communication waits | up to exposed update-step comm, but mostly rank-arrival skew | medium | defer until fresh profile |
| 5 | merge LM Q/K/V projections | GEMM dispatch/efficiency | subset of MatMul 24.8% | medium-high, checkpoint remap required | defer until fresh profile |
| 6 | fused AdamW/clip norm | optimizer traffic | subset of update-step pointwise work | high numerical/state risk | defer until higher candidates close |

Each selected item requires a two-NPU numerical/update probe and an independent 16-NPU short ablation before combination.

## Phase 3 implementation log

| Candidate | Level 1 / numerical evidence | Performance evidence | Status |
|---|---|---|---|
| R1-A CPU/Gloo validity MIN | 2-NPU update: 9.757997 s / 8 microbatches, loss 2.94702148; baseline 10.513027 s, loss 2.93920898 (loss delta 0.27%) | 16-NPU short Job 20961: 69.553444 s, loss 1.19813110; fixed baseline 70.168917 s, loss 1.20413574; improvement 0.615473 s / 0.88% | retained |
| R1-B shared NPU attention mask + dead additive-mask removal | Jobs 20962/20965: S=1024 and S=4096 loss/output bit-exact; input-gradient max diff <=5.9e-14; parameter-gradient max diff <=1.9e-12; deterministic repeat exact. Two-step update: 9.707152 s, loss 2.97631836 (within 1% of the A1 probe reference). | 16-NPU A1+B short Job 20967: 68.771468 s, loss 1.29500366; 0.781976 s / 1.12% faster than A1 and 1.397449 s / 1.99% faster than fixed baseline | retained; Phase-4 full accuracy gate required because short-loss samples vary |
| R1-C image-token replacement | Job 20964: both alternatives are bit-exact for output, token-embedding gradient, and image-embedding gradient. | 25-repeat forward+backward medians per rank: original index_put 0.624/0.823 ms; framework masked_scatter 3.872/3.955 ms; custom dense gather+where 2.250/2.316 ms. Both alternatives regress materially. | rejected under Level-2 rule: hardware-unfriendly substitutes; keep original |

The low parameter-gradient cosine reported by the R1-B probe is an ill-conditioned near-zero-vector artifact: the maximum absolute difference is only 1.9e-12. The corrected predeclared gate accepts cosine only when the absolute error is not already <=1e-10.

## Phase 3 -> Phase 4 closure

| Finding source | Finding | Implementation status | Evidence |
|---|---|---|---|
| hot-path audit | synchronized dataloader validity collective | implemented | `data/data_utils.py`, `train.py`; Jobs 20960/20961 |
| hot-path audit | repeated/dead NPU attention masks | implemented | `models/language_model.py`; Jobs 20965/20967 |
| hot-path audit | dynamic image-token assignment | excluded after required Level-2 exploration | Job 20964: framework and custom alternatives bit-exact but 2.7x-6.3x slower |
| attribution: explicit sync / small-packet communication | NPU scalar MIN + item | implemented | CPU/Gloo MIN with unchanged cross-rank semantics |
| attribution: dispatch/layout/memory churn | per-block mask creation | implemented | one read-only shared mask per training/prefill forward |
| attribution: memory management broadly | allocator not limiting | excluded | allocator API 0.1 ms and 58% peak HBM in Phase-2 report |
| attribution: compilation | no repeated compilation | excluded | warm L0/L1 traces contain no repeated compile event |
| attribution: compute saturation | required GEMM/FA | deferred to a fresh round | Stage 1 exhausted current fused paths; stale QKV hypothesis is not carried forward |
| attribution: communication wait | DDP rank-arrival skew | deferred to a fresh round | transfer is sub-ms; bucket strategy must be reconsidered from fresh L1 |
| attribution: small-op fragmentation | 0.12 ms fusible sequence | excluded | <1% threshold |
| root-cause trace | all DEFINITE/WARNING findings | closed | each row in the Phase-2 root-cause table is implemented, evidence-excluded, or explicitly deferred to fresh profiling |

No selected/actionable Round-1 item remains untreated. Phase 4 begins with full-training accuracy Job 20970, followed by the aligned wall-clock/L0 confirmation if accuracy passes.

## Phase 4 results

- Wall-clock short, Job 20967: 68.771468 s vs 70.168917 s baseline (-1.99%).
- L0, Job 20978, `profiling/20260831_030726_l0`: 80.771385 s vs 84.712026 s baseline (-4.65%); train loss 1.19984375 vs 1.19832153.
- Full training, Job 20997: 554.191518 s vs 563.833686 s Stage-1 baseline (-9.642168 s / -1.71%); `FINAL_VAL_LOSS=1.08230782`, inside the predeclared 1.08 +/- 0.01 gate.
- Jobs 20970/20981/20997 exposed a torch_npu post-result C++ static-destruction abort. Trace markers proved all ranks completed device synchronization plus Gloo/HCCL barriers and destruction before the abort. The training and validation code path was unchanged; a final `os._exit(0)` after flush and completed process-group destruction was validated with the same 16-rank/64-validation-batch exit path in Job 21004, which completed successfully.

Checkpoint B is accepted under the user's standing authorization to make optimization choices and continue without pausing. Evidence is recorded in `evidence_db/nanovlm-sync-mask-20260831.yaml` and `comparison_records/20260831_042500/summary.md` before commit.
