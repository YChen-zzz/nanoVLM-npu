#!/usr/bin/env bash
set -eo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ROOT=/models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning
export PYTHONUNBUFFERED=1
export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1
mkdir -p "$ROOT/logs"
torchrun --standalone --nproc_per_node=2 "$ROOT/probes/probe_stage2_image_replace.py" \
  2>&1 | tee "$ROOT/logs/probe_stage2_c_image_replace.log"
