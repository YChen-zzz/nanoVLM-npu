# NanoVLM Stage 2 Round 1 comparison record

Predeclared final accuracy gate: `FINAL_VAL_LOSS` must be within `1.08 +/- 0.01`.

| Measurement | Original baseline | Round-1 optimized | Result |
|---|---:|---:|---|
| 16-NPU short wall clock | 70.168917 s | 68.771468 s (Job 20967) | pass, -1.99% |
| 16-NPU L0 wall clock | 84.712026 s | 80.771385 s (Job 20978) | pass, -4.65% |
| 16-NPU full `BENCH_MEASURE` | 563.833686 s | 554.191518 s (Job 20997) | pass, -1.71% |
| Full validation loss | 1.08506393 | 1.08230782 (Job 20997) | pass |

Layer-1 evidence:

- A1 2-NPU update: loss delta 0.27% versus its baseline.
- B S=1024/4096: loss/output bit-exact, gradient max absolute difference <=1.9e-12, deterministic repeats exact.
- B real 2-NPU update: loss delta 0.40% versus A1 probe.
- C framework and custom replacements: bit-exact but 2.7x-6.3x slower, rejected.

Exit-path evidence:

- Full metrics were emitted before a torch_npu 2.7.1.post2 static-destructor SIGABRT.
- Job 21002 proved every rank reached successful Gloo and HCCL destruction before that abort.
- Job 21004 exercised 16 ranks and the same 64-batch validation path, then completed cleanly with the post-destroy, post-flush `_exit(0)` workaround.

Raw evidence references:

- `logs/probe_stage2_a1_cpu_sync.log`
- `logs/stage2_a1_cpu_sync_short.log`
- `logs/probe_stage2_b_mask_equiv.log`
- `logs/probe_stage2_b_mask_train.log`
- `logs/stage2_ab_shared_mask_short.log`
- `logs/probe_stage2_c_image_replace.log`
- `logs/probe_stage2_exit_cleanup_16_fullval.log`
- `benchmark_results_npu/manual_20260831_040405/eager.log`
- `profiling/20260831_030726_l0/run.log`
