#!/usr/bin/env bash
cd "$(dirname "$0")/source"
export NANOVLM_ASSETS="${NANOVLM_ASSETS:-../../offline_assets}"
export HF_HOME=../archive/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 NANOVLM_BENCHMARK_CASE=eager
torchrun --standalone --nproc_per_node=16 train.py \
  --no-compile --no_log_wandb --no_lmms_eval --no_eval --eval_at_end \
  --benchmark_warmup_steps 10 --max_training_steps 210 --stats_log_interval 50 \
  --gradient_accumulation_steps 4 --batch_size 2 \
  --train_dataset_files "$NANOVLM_ASSETS/datasets/FineVision_concat_shuffled_2/full/data-*.parquet" \
  --vit_model_type "$NANOVLM_ASSETS/models/siglip2-base-patch16-512" \
  --lm_model_type "$NANOVLM_ASSETS/models/SmolLM2-360M-Instruct" \
  --lm_tokenizer "$NANOVLM_ASSETS/models/SmolLM2-360M-Instruct"
