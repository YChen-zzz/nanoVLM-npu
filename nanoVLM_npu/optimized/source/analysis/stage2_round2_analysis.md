# Stage 2 Round 2 — Phase 2 analysis

Status: fresh Round-2 analysis completed from Job 21008 (`profiling/20260831_043053_l1`).
Cross-check baseline is the immediately preceding Phase-4 L0,
`profiling/20260831_030726_l0`. No Round-1 candidate conclusion is reused.

## Measurement bounds and cross-check

The active window is three microbatches. Tier 1 remains an order-of-magnitude
roofline floor for the unchanged 460M-parameter LM+ViT workload on a 910B-class
NPU; it is not a runtime prediction.

| Bound | Time / 3 microbatches | Evidence |
|---|---:|---|
| Tier 1 roofline | ~0.30 s | major GEMM/attention compute and HBM lower-bound estimate |
| Tier 2 L0 Computing | 1.5903 s | previous Phase-4 L0 rank-0 step trace |
| Tier 3 aligned wall clock | 2.0631 s | accepted short 68.771468 s / 100 microbatches × 3 |

- gap A = 1.2903 s: kernel/algorithm efficiency dominates the physical gap.
- gap B = 0.4728 s, 22.9% of Tier 3: exposed host/communication is still material.
- L0: Computing 1590.3 ms, Free 181.1 ms, Comm 494.8 ms, utilization 89.8%.
- L1: Computing 1594.5 ms, Free 1184.7 ms, Comm 484.9 ms, utilization 57.4%.
- L1 utilization is 32.4 percentage points below L0. This exceeds the 20 pp
  guard, so L1 Free/host-idle is a profiler artifact. Bottleneck classification
  uses L0; L1 operator/kernel attribution remains valid.

## Line B root-cause trace

| Finding source | Finding | Bridge | Source | Root cause | Candidate / disposition |
|---|---|---|---|---|---|
| op statistic DEFINITE | top 3 are 47.8% device time | kernel shapes + source | DDP reducer, fused attention, Linear | required workload; allreduce exposure is amplified by unequal per-rank image work | C4 rank balancing; otherwise do not remove required math |
| op statistic SIGNAL | movement is 133.7 ms / 6.9% | call stacks + kernel shapes | LM QKV layouts, RoPE, dtype boundaries, gate/up weight cat | most transforms are required by BNSD attention and mixed-precision boundaries | C5 later; identity `to()` sites are <1% and excluded |
| op/kernel SIGNAL | Conv2DBackpropFilter 39.4 ms / 2.0%, memory dominated | input shapes + source | `vision_transformer.py:34` patch convolution | required ViT patch-embedding gradient with 30/44/43 images | exclude this round: only 2.0% ceiling and no equivalent fused training op |
| kernel SIGNAL | AI core and vector core both significant | op cross-check | matmul/FA plus elementwise optimizer/backward | expected mixed transformer workload, not fallback | covered by C5; no standalone defect |
| kernel SIGNAL | long waits, including 143.7 ms before image concat and 17–33 ms around optimizer foreach | trace timing + all-rank comparison | data-dependent image work and optimizer transition | ranks enter DDP collectives/next phases at different times | C4 and C3 |
| trace SIGNAL | H2D/preallocation candidates | Call Stack | `train_profile_stage2.py:561-563`, `_process_images` | input copies are per-batch and data-dependent | excluded: project call sites total <1% and pinning already enabled |
| operator DEFINITE | 89.1% pure-host time | L0 cross-check + call stacks | framework metadata, DDP wrapper, model dispatch | L1 barriers inflate host exposure; true L0 Free ceiling is 181.1 ms / 8.8% | C2; broad flat-forward retained only as later fallback |
| operator SIGNAL | `record_event` 6.43 ms | Call Stack | fused norm/attention autograd and DDP | autograd bookkeeping | excluded: 0.31% of Tier 3 |
| operator DEFINITE | apparent AI_CPU Matmul/Addmm/Conv | accelerator-core cross-check | nested communication attribution | exactly 59 AI_CPU kernels are the 59 communication kernels; compute kernels run AI Core/vector | excluded as parser attribution false positive |
| memory SIGNAL | reserved +995 MB, churn 14,055, fragmentation +3261 MB | API statistic + peak attribution | normal activation/autograd lifetime | allocator memory-management API time is 0.0 ms; active peak 36.7 GB is 56% HBM | broad preallocation excluded |
| memory DEFINITE | repeated equal-size allocations and 10,987 short-lived tensors | operator memory + source | activations, autograd, fused attention intermediates | allocations are framework-owned and allocator-cached; no allocator stall | exclude broad buffers; C5 covers structural sources only |
| API SIGNAL | sync 743.3 ms / 45.3% of L1 API time | Call Stack + source | benchmark per-step `synchronize`, `.item()`, logging scalar collective | instrumentation repeatedly drains the async queue | C2, capped by L0 Free 181.1 ms |
| communication DEFINITE | 59 ops, 484.9 ms, reported 100% Wait | all-rank step/kernel comparison | DDP reducer plus one scalar logging allreduce | rank compute spans vary with 22–60 images/microbatch; scalar stats collective is unnecessary without W&B | C2 removes scalar collective; C4 addresses imbalance |
| `index_put` root trace | 3 outer calls take 66.5/70.9/75.5 ms | trace view + input shapes + source | `models/vision_language_model.py:47` | boolean advanced assignment needs dynamic indices and drains preceding asynchronous vision work | C1 |

## All-rank evidence

- Three-microbatch Computing ranges from 1380.4 ms (rank 12) to 1823.8 ms
  (rank 7), a 443.4 ms spread (21.5% of Tier 3).
- Conv-backward image counts range from 22 to 60 per microbatch. Ranks with
  greater image counts generally have greater compute and less communication wait.
- Communication time ranges from 223.8 ms to 497.3 ms. This inverse relation
  confirms straggler wait rather than a link-bandwidth bottleneck.

## Line A source audit

### Call-chain layer quantification

Inclusive host attribution totals are nesting-heavy, so percentages are shares
of summed inclusive time, not additive wall-clock shares.

| Layer / first project frame | Inclusive share | Required candidate |
|---|---:|---|
| framework / no project frame | 31.84% | C2; flat-forward is later fallback |
| `train_profile_stage2.py:583 train` (model call) | 18.42% | C1 and C5 |
| `train_profile_stage2.py:613 train` (optimizer) | 5.80% | below 10%; C2 removes unrelated telemetry barrier |
| LM Q/K/V projection lines 285–287 | 3.10–3.18% each | individually below 10%; required GEMMs |

Every layer above the 10% gate has a candidate or an evidence-based disposition.

### Hot-path operation audit

| Operation | Redundant work? | Output input-dependent? | Fixed shape? | Multi-step replacement? | Disposition |
|---|---|---|---|---|---|
| dataloader validity MIN | necessary termination check | yes | scalar | can be pipelined | C3 |
| input ids/labels/mask H2D | necessary | yes | yes | prefetch possible | <1% project host time; exclude |
| image list concat + H2D | necessary | yes | image count varies | batching already used | <1%; exclude |
| ViT patch conv | necessary | yes | image count varies | no supported fused training equivalent | 2.0%; exclude |
| ViT QKV projection | no | yes | per-image | already one combined Linear | accepted form |
| image-token replacement | dynamic-index overhead | yes | output `[2,4096,960]` fixed | masked-scatter equivalent | C1 |
| RoPE construction | repeats by shape | no for fixed `(start,T)` | yes | cacheable | device cost 0.006%; exclude |
| LM Q/K/V | required | yes | yes | structural packing possible | C5 later; risk/ceiling below C1/C2 |
| fused attention | required | yes | yes | already official NPU FA | accepted form |
| gate/up weight concat | repeats after each update | yes (weights update) | yes | packed parameter needs checkpoint remap | C5 later |
| Add+RMSNorm | no | yes | yes | already official fused op | accepted form |
| masked loss row selection | required to avoid full-vocab projection | yes | valid count varies | dynamic nonzero is necessary for current sparse head | retain; saved head compute exceeds sync cost |
| per-microbatch sync / `.item()` / loss gather | telemetry-only | no training dependency | scalar | defer to boundary | C2 |
| DDP gradient allreduce | required | yes | fixed buckets | overlap already active; imbalance leaves tail | C4 later |
| AdamW foreach math | required | yes | fixed params | framework foreach already present | no removable work found |

## Ten waste categories

| Waste category | Bottleneck type | Status | Evidence |
|---|---|---|---|
| explicit synchronization | Host | candidate C2 | L1 API sync 743.3 ms; capped by L0 Free 181.1 ms |
| dispatch scheduling | Host | candidate C2/C5 | 17,873 launches; L0 host ceiling 8.8% |
| memory-management blocking | Host/Allocator | excluded | memory-management API 0.0 ms; active peak 56% HBM |
| online compilation | Host | excluded | tiling/compile subtotal 4.1 ms / 0.3%; no persistent compile signal |
| memory bandwidth limited | Memory | candidate C5 / conv excluded | movement 133.7 ms; conv only 39.4 ms |
| compute saturation | Compute | excluded from removal | cube utilization 91%; matmul/FA are required model math |
| layout/format conversion | Memory | candidate C5 | Transpose+Cast 130.2 ms device, mostly required layout/dtype boundaries |
| communication synchronization wait | Comm | candidates C2/C4 | 223.8–497.3 ms by rank; compute spread 443.4 ms |
| small-operator fragmentation | General | candidate C5 | 49.2% kernels <20 us; only 5.01 ms <5 us, so structural fusion only |
| uncovered latency | General | candidates C1/C3/C4 | index write drains queue; validity collective and rank skew expose stalls |

## Candidate ranking and automatic Checkpoint A

The user explicitly pre-authorized optimization choices. Checkpoint A therefore
selects candidates automatically, in descending counterfactual ceiling, while
preserving training dynamics and checkpoint structure.

| Rank | Candidate | Dimension / waste removed | Counterfactual upper bound | Risk | Decision |
|---:|---|---|---:|---|---|
| 1 | C4 globally balance variable image encoder work while preserving the exact global batch | hide / communication wait | 443.4 ms / 21.5% | high; distributed autograd and sample-order constraints | defer until lower-risk candidates exhausted |
| 2 | C1 replace dynamic boolean `index_put` with bit-exact `masked_scatter` | replace / uncovered sync | 214.5 ms / 10.4% | medium; isolated kernel is slower but avoids queue drain | implement first |
| 3 | C2 defer benchmark telemetry sync, scalar extraction, and no-W&B scalar allreduce to the measurement boundary | eliminate+hide / explicit sync | L0 Free cap 181.1 ms / 8.8% | low-medium; reporting path only | implement second, separately |
| 4 | C3 pipeline next-batch CPU/Gloo validity reduction under current NPU compute | hide / communication+host wait | L0 Free cap 181.1 ms / 8.8% | medium; iterator lifecycle | implement if C1/C2 insufficient |
| 5 | C5 reduce structural layout/cast/weight-cat launches | eliminate+replace / movement+fragmentation | 133.7 ms / 6.5% | medium-high; autograd/checkpoint coupling | revisit with fresh L1 after C1/C2 |
| 6 | replace patch convolution | replace / memory bandwidth | 39.4 ms / 1.9% | high | exclude this round |

Automatic Checkpoint A decision: approve C1, then C2 as independent ablations.
C3 is approved only if fresh evidence after C1/C2 still shows exposed host wait.
C4/C5 require a fresh profile because C1/C2 can materially change queueing and
communication exposure.

## Phase 3 implementation and ablation results

All 16-NPU short runs used the same 10-step warmup, 25 measured optimizer
steps, 100 measured microbatches, batch size 2, gradient accumulation 4, data,
and environment. The accepted Round-1 short (`68.771468 s`) is the fixed
reference.

| Variant | Job | Elapsed | Delta vs reference | Result |
|---|---:|---:|---:|---|
| C1 framework `masked_scatter` | 21053 | 69.095788 s | +0.47% | rejected |
| C1 + C2a/C2b | 21054 | 68.554320 s | -0.32% | superseded |
| C2a/C2b only, original `index_put` | 21069 | 68.410410 s | -0.53% | superseded |
| C1 custom dense gather/where + C2a/C2b | 21070 | 68.047145 s | -1.05% | superseded |
| previous row + C2c deferred loss readback | 21080 | 62.977993 s | -8.43% | accepted |
| previous row + C3 pipelined Gloo validity | 21086 | 63.090434 s | -8.26% | rejected (+0.18% vs 21080) |

The first 256-GiB scheduling workaround (Job 21028) was killed by SIGKILL
during 16-rank model initialization before training. It produced no benchmark
sample and is excluded; all valid A/B runs used the original 512-GiB request.

### Accuracy / equivalence evidence

- C1 framework and custom implementations: output, token gradient, and image
  gradient max absolute difference were all exactly zero in Job 21014.
- C2a host-known RoPE maximum: cosine and sine max absolute difference were
  zero for `(start,length)` `(0,4096)`, `(113,1024)`, and `(4090,32)` in Job
  21031, including the dynamic-scaling boundary.
- C2b changes only no-W&B telemetry and logging collectives.
- C2c stores detached scalar losses, performs one batched D2H at epoch end,
  and converts values to Python in original microbatch order before applying
  the same accumulation factor and sum. It does not feed model or optimizer.
- C3 sequence semantics passed a two-rank Gloo test with invalid-batch skips
  and uneven iterator exhaustion in Job 21087, but the end-to-end regression
  caused rejection.

## Phase 3 to Phase 4 closure

| Finding source | Finding | Status | Evidence |
|---|---|---|---|
| hot-path audit: image-token replacement | dynamic boolean assignment drains the queue | implemented | custom dense gather/where is bit-exact; Job 21070 improved over C2-only by 0.363265 s |
| hot-path audit: telemetry sync | repeated synchronize, device token `.item()`, logging reductions | implemented | C2a/b plus C2c; Job 21080 is 8.43% faster than the fixed reference |
| hot-path audit: dataloader validity MIN | CPU/Gloo collective can be overlapped | implemented then rejected | Job 21087 semantics passed; Job 21086 regressed 0.18% vs 21080 |
| root trace: `index_put` | 214.5 ms host trace over three microbatches | implemented with custom alternative | framework alternative rejected; required custom alternative accepted |
| explicit synchronization | L0 Free ceiling 181.1 ms | implemented | deferred loss readback is the dominant measured gain |
| communication wait / rank imbalance | 443.4 ms compute spread | deferred to next fresh profile | current profile is invalidated by an 8.43% queueing change; C4 requires fresh L1 |
| layout/cast/small-op fragmentation | 133.7 ms movement subtotal | deferred to next fresh profile | current profile is invalidated; C5 requires fresh L1 |
| allocator / online compile / patch conv / required compute | excluded | profiling evidence in the tables above | no actionable unclosed item for this round |

Automatic Checkpoint-A closure selects Job 21080's source state for Phase 4.
C4 and C5 are not carried forward as current-profile conclusions; if the final
target is not met they must be reconsidered only after a new Round-3 L1.

## Phase 4 gates

### Full training and final validation

Job 21091 completed successfully using the original 210-step full-training
parameters from `run_npu_benchmark.sh` and the accepted Job-21080 source
snapshot.

- `BENCH_MEASURE`: `500.507781 s` for 200 measured steps / 800 microbatches.
- `FINAL_VAL_LOSS`: `1.08257675` over 64 batches per rank; required interval
  was `[1.07, 1.09]`, so the accuracy gate passed.
- Versus Round-1 accepted full `554.191518 s`: `53.683737 s` / `9.69%` faster.
- Versus the initial Stage-1 full `563.833686 s`: `63.325905 s` / `11.23%`
  faster.

### Phase-end L0

Job 21096 completed successfully at `profiling/20260831_064059_l0`. The L0
instrumented short reports `76.585270 s`; this includes profiler overhead and
is not the authoritative wall clock. Rank-0 step trace changed from Round-1
Phase-4 L0 Computing/Free/Communication `1590.3/181.1/494.8 ms` to
`1916.359/31.708/697.894 ms`. Computing utilization rose from `89.8%` to
`98.37%`, and exposed Free time fell by `149.392 ms` (`82.5%`). Across all 16
ranks, median Computing/Free/Communication were
`1991.272/31.595/619.892 ms`.

The higher Computing and Communication totals reflect the data-dependent image
mix of the newly sampled active window and the custom dense replacement, while
the collapse in Free time is directionally consistent with eliminating repeated
loss readback barriers. The no-profiler full result is the authoritative
end-to-end metric.

Both Phase-4 gates pass. Automatic Checkpoint B is therefore approved for the
accepted C1-custom + C2a/b/c changes after the evidence-db record is written.
The requested terminal condition is met (`500.507781 s`, approximately 500 s,
with validation loss `1.08257675`), so automatic Checkpoint C selects stop after
the engineering commit; no Round-3 conclusions are carried over from this
profile.
