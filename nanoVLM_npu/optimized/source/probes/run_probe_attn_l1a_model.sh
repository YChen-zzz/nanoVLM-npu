#!/usr/bin/env bash
set -eo pipefail
# Stage-1 diagnostic only: execute the real model/data/checkpoint path with the
# L1a attention implementation on two NPUs. This is not a performance ablation.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_LAUNCH_BLOCKING=1
ROOT=/models/share/chenyupeng/chenyupeng/Agent_training_optimization/diff_models/VLM/nanoVLM_skill-tuning
WORK=/tmp/nanovlm_attn_l1a_probe_${SCO_JOB_ID:-$$}
mkdir -p "$WORK"
cp "$ROOT/train.py" "$WORK/train.py"
cp -a "$ROOT/data" "$ROOT/models" "$WORK/"
cp "$ROOT/ablations/sn_lm_attention_L1a/language_model.py" "$WORK/models/language_model.py"
ln -s "$ROOT/offline_assets" "$WORK/offline_assets"
export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
pip install --no-index --no-deps \
  "$ROOT/offline_assets/wheels/torchvision-0.22.1"*.whl \
  "$ROOT/offline_assets/wheels/einops-"*.whl \
  "$ROOT/offline_assets/wheels/datasets-"*.whl 2>&1 | tail -5 || true
cd "$WORK"
torchrun --standalone --nproc_per_node=2 train.py \
  --no-compile --no_log_wandb --no_lmms_eval --no_eval \
  --benchmark_warmup_steps 0 --max_training_steps 1 \
  --stats_log_interval 1 --gradient_accumulation_steps 1 --batch_size 2 \
  --train_dataset_files "$ROOT/offline_assets/datasets/FineVision_concat_shuffled_2/full/data-*.parquet" \
  --vit_model_type "$ROOT/offline_assets/models/siglip2-base-patch16-512" \
  --lm_model_type "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  --lm_tokenizer "$ROOT/offline_assets/models/SmolLM2-360M-Instruct" \
  2>&1 | tee "$ROOT/logs/probe_attn_l1a_model.log"
