#!/usr/bin/env bash
set -eo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test

export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ROOT=/models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning
RESULTS="$ROOT/benchmark_results_npu"
RUN_ID="${SCO_JOB_ID:-${JOB_ID:-manual}}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RESULTS/$RUN_ID"
mkdir -p "$RUN_DIR" "$ROOT/.cache/huggingface"

export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

cd "$ROOT"

pip install --no-index --no-deps \
    "$ROOT/offline_assets/wheels/torchvision-0.22.1"*.whl \
    "$ROOT/offline_assets/wheels/einops-"*.whl \
    "$ROOT/offline_assets/wheels/datasets-"*.whl 2>&1 | tail -5 || true

echo "=============================================="
echo "  nanoVLM NPU Warmup Benchmark (eager only)"
echo "  16 NPUs (single node)"
echo "  Global batch = 16 x bs2 x grad_accum4 = 128"
echo "  (Stage-1 optimized attention, expandable allocator, no LM checkpointing)"
echo "=============================================="

python - <<'PY' | tee "$RUN_DIR/environment.txt"
import platform
import torch
try:
    import torch_npu
    npu_available = torch.npu.is_available()
    npu_count = torch.npu.device_count() if npu_available else 0
except ImportError:
    npu_available = False
    npu_count = 0
import transformers
print("python", platform.python_version())
print("torch", torch.__version__)
print("npu_available", npu_available)
print("npu_count", npu_count)
print("transformers", transformers.__version__)
if npu_available:
    for i in range(npu_count):
        print("npu", i, torch.npu.get_device_name(i), torch.npu.get_device_properties(i).total_memory)
PY

run_case() {
    local case_name=$1
    local compile_flag=$2
    local log_file="$RUN_DIR/${case_name}.log"
    export NANOVLM_BENCHMARK_CASE="$case_name"

    local start_s end_s rc
    start_s=$(date +%s)
    set +e
    torchrun --standalone --nproc_per_node=16 train.py \
        "$compile_flag" \
        --no_log_wandb \
        --no_lmms_eval \
        --no_eval \
        --eval_at_end \
        --benchmark_warmup_steps 10 \
        --max_training_steps 210 \
        --stats_log_interval 50 \
        --gradient_accumulation_steps 4 \
        --batch_size 2 \
        --train_dataset_files "$ROOT/offline_assets/datasets/FineVision_concat_shuffled_2/full/data-*.parquet" \
        --vit_model_type "$ROOT/offline_assets/models/siglip2-base-patch16-512" \
        --lm_model_type "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
        --lm_tokenizer "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
        2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    end_s=$(date +%s)
    printf 'BENCH_PROCESS case=%s elapsed_s=%s exit_code=%s\n' "$case_name" "$((end_s-start_s))" "$rc" | tee -a "$log_file"
    return "$rc"
}

# Only eager mode — torch.compile inductor backend not supported on Ascend NPU
run_case eager --no-compile

ln -sfn "$RUN_DIR" "$RESULTS/latest"
printf 'BENCHMARK_RUN_DIR=%s\n' "$RUN_DIR"

echo ""
echo "=== GPU Baseline (eager 8xH100) ==="
echo "  measured train_loss = 1.11979331"
echo "  final_val_loss      = 1.07837781"
echo "==================================="
