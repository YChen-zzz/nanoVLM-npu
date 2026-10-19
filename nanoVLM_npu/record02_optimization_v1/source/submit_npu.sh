#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/ktp.sh"

ktp submit \
    --name nanovlm-npu-bench \
    --image 10.1.30.201:31443/user-1-chenyupeng/codeserver-mindspeed:v1.0.5 \
    --framework PyTorch \
    --command "cd /models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning && bash run_npu_benchmark.sh" \
    --cpu 64 \
    --memory 512Gi \
    --npu 16 \
    --queue user-1-chenyupeng-compute \
    --job-type vcjob \
    --max-runtime 1440
