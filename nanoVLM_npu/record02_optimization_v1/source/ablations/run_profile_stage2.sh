#!/usr/bin/env bash
set -eo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ROOT=/models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning
LEVEL="${NANOVLM_NPU_PROFILE_LEVEL:-L1}"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ROOT/profiling/${STAMP}_${LEVEL,,}"
mkdir -p "$RUN_DIR" "$ROOT/.cache/huggingface"
ln -sfn "$(basename "$RUN_DIR")" "$ROOT/profiling/latest"
export NANOVLM_NPU_PROFILE_DIR="$RUN_DIR"
export NANOVLM_NPU_PROFILE_LEVEL="$LEVEL"
export NANOVLM_BENCHMARK_CASE="stage2_${LEVEL,,}"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python "$ROOT/probes/validate_profiling_env.py" \
  --device npu:0 --output-dir "$RUN_DIR" | tee "$RUN_DIR/environment_check.log"

cd "$ROOT"
pip install --no-index --no-deps \
  "$ROOT/offline_assets/wheels/torchvision-0.22.1"*.whl \
  "$ROOT/offline_assets/wheels/einops-"*.whl \
  "$ROOT/offline_assets/wheels/datasets-"*.whl 2>&1 | tail -5 || true
torchrun --standalone --nproc_per_node=16 ablations/train_profile_stage2.py \
  --no-compile --no_log_wandb --no_lmms_eval --no_eval \
  --benchmark_warmup_steps 10 --max_training_steps 35 --stats_log_interval 50 \
  --gradient_accumulation_steps 4 --batch_size 2 \
  --train_dataset_files "$ROOT/offline_assets/datasets/FineVision_concat_shuffled_2/full/data-*.parquet" \
  --vit_model_type "$ROOT/offline_assets/models/siglip2-base-patch16-512" \
  --lm_model_type "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  --lm_tokenizer "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  2>&1 | tee "$RUN_DIR/run.log"
printf 'NPU_PROFILE_RUN_DIR=%s\n' "$RUN_DIR"
