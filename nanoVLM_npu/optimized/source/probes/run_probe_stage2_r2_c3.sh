#!/usr/bin/env bash
set -eo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ROOT=/models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning
SNAPSHOT="$ROOT/analysis/round2/c1custom_c2c_c3_source_snapshot"
export PYTHONPATH="$SNAPSHOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
torchrun --standalone --nproc_per_node=2 "$ROOT/probes/probe_stage2_r2_c3_pipeline.py" \
  2>&1 | tee "$ROOT/logs/probe_stage2_r2_c3_pipeline.log"
