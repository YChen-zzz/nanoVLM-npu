#!/usr/bin/env bash
set -eo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ROOT=/models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning
export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1
export NANOVLM_BENCHMARK_CASE=probe_stage2_exit_cleanup
WORK=/tmp/nanovlm_stage2_exit_${SCO_JOB_ID:-$$}
mkdir -p "$WORK"
cp "$ROOT/train.py" "$WORK/train.py"
cp -a "$ROOT/data" "$ROOT/models" "$WORK/"
ln -s "$ROOT/offline_assets" "$WORK/offline_assets"
cd "$WORK"
pip install --no-index --no-deps \
  "$ROOT/offline_assets/wheels/torchvision-0.22.1"*.whl \
  "$ROOT/offline_assets/wheels/einops-"*.whl \
  "$ROOT/offline_assets/wheels/datasets-"*.whl 2>&1 | tail -5 || true
torchrun --standalone --nproc_per_node=2 train.py \
  --no-compile --no_log_wandb --no_lmms_eval --no_eval --eval_at_end \
  --val_size 128 --benchmark_warmup_steps 0 --max_training_steps 2 --stats_log_interval 1 \
  --gradient_accumulation_steps 4 --batch_size 2 \
  --train_dataset_files "$ROOT/offline_assets/datasets/FineVision_concat_shuffled_2/full/data-*.parquet" \
  --vit_model_type "$ROOT/offline_assets/models/siglip2-base-patch16-512" \
  --lm_model_type "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  --lm_tokenizer "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  2>&1 | tee "$ROOT/logs/probe_stage2_exit_cleanup.log"
