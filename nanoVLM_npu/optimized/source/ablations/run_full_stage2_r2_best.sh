#!/usr/bin/env bash
set -eo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ROOT=/models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning
SNAPSHOT="$ROOT/analysis/round2/c1custom_c2c_source_snapshot"
RESULTS="$ROOT/benchmark_results_npu"
RUN_ID="${SCO_JOB_ID:-${JOB_ID:-manual}}_$(date +%Y%m%d_%H%M%S)_stage2_r2"
RUN_DIR="$RESULTS/$RUN_ID"
WORK=/tmp/nanovlm_stage2_r2_full_${SCO_JOB_ID:-$$}
mkdir -p "$RUN_DIR" "$WORK" "$ROOT/.cache/huggingface"
cp "$SNAPSHOT/train.py" "$WORK/train.py"
cp -a "$SNAPSHOT/data" "$SNAPSHOT/models" "$WORK/"
ln -s "$ROOT/offline_assets" "$WORK/offline_assets"
export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1
export NANOVLM_BENCHMARK_CASE=eager
cd "$WORK"
pip install --no-index --no-deps \
  "$ROOT/offline_assets/wheels/torchvision-0.22.1"*.whl \
  "$ROOT/offline_assets/wheels/einops-"*.whl \
  "$ROOT/offline_assets/wheels/datasets-"*.whl 2>&1 | tail -5 || true
start_s=$(date +%s)
set +e
torchrun --standalone --nproc_per_node=16 train.py \
  --no-compile --no_log_wandb --no_lmms_eval --no_eval --eval_at_end \
  --benchmark_warmup_steps 10 --max_training_steps 210 --stats_log_interval 50 \
  --gradient_accumulation_steps 4 --batch_size 2 \
  --train_dataset_files "$ROOT/offline_assets/datasets/FineVision_concat_shuffled_2/full/data-*.parquet" \
  --vit_model_type "$ROOT/offline_assets/models/siglip2-base-patch16-512" \
  --lm_model_type "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  --lm_tokenizer "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  2>&1 | tee "$RUN_DIR/eager.log"
rc=${PIPESTATUS[0]}
set -e
end_s=$(date +%s)
printf 'BENCH_PROCESS case=eager elapsed_s=%s exit_code=%s\n' "$((end_s-start_s))" "$rc" | tee -a "$RUN_DIR/eager.log"
ln -sfn "$RUN_DIR" "$RESULTS/latest"
printf 'BENCHMARK_RUN_DIR=%s\n' "$RUN_DIR"
exit "$rc"
