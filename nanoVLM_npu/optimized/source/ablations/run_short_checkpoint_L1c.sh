#!/usr/bin/env bash
set -eo pipefail
# SN-LM-Checkpoint L1c: checkpoint one of every eight LM blocks.
# Full benchmark values are 210 total steps,
# 10 warmup steps, batch_size=2, grad_accum=4. This short run keeps batch/data/
# seed/dynamics and measures 25 post-warmup optimizer steps (~60-120 seconds).
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ROOT=/models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning
export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NANOVLM_BENCHMARK_CASE=sn_lm_checkpoint_L1c
WORK=/tmp/nanovlm_checkpoint_L1c_${SCO_JOB_ID:-$$}
mkdir -p "$WORK"
cp "$ROOT/train.py" "$WORK/train.py"
cp -a "$ROOT/data" "$ROOT/models" "$WORK/"
cp "$ROOT/ablations/sn_lm_checkpoint_L1c/language_model.py" "$WORK/models/language_model.py"
ln -s "$ROOT/offline_assets" "$WORK/offline_assets"
cd "$WORK"
mkdir -p logs
pip install --no-index --no-deps \
  "$ROOT/offline_assets/wheels/torchvision-0.22.1"*.whl \
  "$ROOT/offline_assets/wheels/einops-"*.whl \
  "$ROOT/offline_assets/wheels/datasets-"*.whl 2>&1 | tail -5 || true
torchrun --standalone --nproc_per_node=16 train.py \
  --no-compile --no_log_wandb --no_lmms_eval --no_eval \
  --benchmark_warmup_steps 10 --max_training_steps 35 \
  --stats_log_interval 50 --gradient_accumulation_steps 4 --batch_size 2 \
  --train_dataset_files "$ROOT/offline_assets/datasets/FineVision_concat_shuffled_2/full/data-*.parquet" \
  --vit_model_type "$ROOT/offline_assets/models/siglip2-base-patch16-512" \
  --lm_model_type "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  --lm_tokenizer "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  2>&1 | tee "$ROOT/logs/sn_lm_checkpoint_L1c.log"
