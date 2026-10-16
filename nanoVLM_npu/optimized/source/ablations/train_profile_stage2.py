import os
import sys
if os.environ.get("NANOVLM_GPU_TEACHER_DIR") and os.environ.get("LOCAL_RANK", "0") != "0":
    # Collect compile IR only once. All ranks still execute the compiled model.
    os.environ.pop("TORCH_COMPILE_DEBUG", None)
import re
import json
import math
import time
import torch

# --- NPU compatibility ---
_USE_NPU = False
try:
    import torch_npu
    if torch.npu.is_available():
        _USE_NPU = True
except ImportError:
    pass

def _accelerator_is_available():
    return torch.npu.is_available() if _USE_NPU else torch.cuda.is_available()

def _accelerator_device_count():
    return torch.npu.device_count() if _USE_NPU else torch.cuda.device_count()

def _accelerator_set_device(dev):
    if _USE_NPU:
        torch.npu.set_device(dev)
    else:
        torch.cuda.set_device(dev)

def _accelerator_synchronize():
    if _USE_NPU:
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()

def _accelerator_current_device():
    return torch.npu.current_device() if _USE_NPU else torch.cuda.current_device()

def _accelerator_device_type():
    return "npu" if _USE_NPU else "cuda"
# --- end NPU compatibility ---

import numpy
import random
import argparse
import contextlib
import subprocess
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names, load_from_disk

torch.manual_seed(0)
if _accelerator_is_available():
    if _USE_NPU:
        torch.npu.manual_seed_all(0)
    else:
        torch.cuda.manual_seed_all(0)

PG_CPU = None

from data.datasets import VQADataset
from data.collators import VQACollator
from data.data_utils import synchronized_dataloader_step
from data.advanced_datasets import ConstantLengthDataset
from data.processors import get_image_processor, get_tokenizer

import models.config as config
from models.vision_language_model import VisionLanguageModel

#Otherwise, the tokenizer will throw a warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if not _USE_NPU:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

# Fix for "Decompressed data too large" error with certain PNGs
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def init_dist():
    backend = 'hccl' if _USE_NPU else 'nccl'
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    _accelerator_set_device(local_rank)
    # torch.cuda.manual_seed(0)           # seed *this* GPU only

def destroy_dist():
    global PG_CPU
    debug_teardown = os.environ.get("NANOVLM_DEBUG_TEARDOWN") == "1"
    rank = dist.get_rank()
    def mark(stage):
        if debug_teardown:
            print(f"TEARDOWN rank={rank} stage={stage}", flush=True)
    mark("start")
    # Drain device work and make every rank enter teardown together. This is
    # required after the long validation path; otherwise one rank can destroy
    # communicators while peers still own asynchronous NPU work.
    if _USE_NPU:
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()
    mark("device_sync_done")
    dist.barrier()
    mark("default_barrier_done")
    # Explicitly tear down the auxiliary Gloo group on every rank before the
    # default HCCL group. Leaving it to interpreter shutdown can race with
    # HCCL teardown after long validation runs and abort a worker on exit.
    if PG_CPU is not None:
        dist.barrier(group=PG_CPU)
        mark("gloo_barrier_done")
        dist.destroy_process_group(PG_CPU)
        PG_CPU = None
        mark("gloo_destroy_done")
    dist.destroy_process_group()
    mark("default_destroy_done")

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(obj):
    """
    Gather *any* picklable object from every rank without allocating
    temporary CUDA buffers.  Returns a list [rank0_obj, rank1_obj, …].

    Falls back to a single-rank list when torch.distributed is not initialised.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]

    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj, group=PG_CPU)  # CPU path
    return result

def dist_mean_scalar(x) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)

    t = torch.tensor(x, device=_accelerator_current_device(), dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)           # in‑place, returns None
    t /= dist.get_world_size()
    return t.item()

def wrap_model(model):
    local_rank = int(os.environ["LOCAL_RANK"])
    return DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

def get_run_name(train_cfg, vlm_cfg):
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr_vision_{train_cfg.lr_vision_backbone}-language_{train_cfg.lr_language_backbone}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"

def get_dataloaders(train_cfg, vlm_cfg):
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    dataset_names_to_load = train_cfg.train_dataset_name
    if "shards" in train_cfg.train_dataset_name:
        print("Loading shards")
        total_shards = 56
        dataset_names_to_load = [train_cfg.train_dataset_path + f"/shard_{i}" for i in range(total_shards)]

    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    # Load and combine all training datasets
    combined_train_data = []

    for dataset_name in dataset_names_to_load:
        print(f"Loading dataset: {dataset_name}")
        if "shard_" in dataset_name:
            try:
                train_ds = load_from_disk(dataset_name)
                combined_train_data.append(train_ds)
                continue
            except Exception as e:
                print(f"Warning: Failed to load dataset shard '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
                continue
        last_error = None
        for attempt in range(1, 7):
            try:
                local_data_files = getattr(train_cfg, "train_dataset_files", None)
                if local_data_files:
                    train_ds = load_dataset("parquet", data_files={"train": local_data_files}, streaming=train_cfg.stream_dataset)['train']
                else:
                    train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name, streaming=train_cfg.stream_dataset, on_bad_files='warn')['train']
                if train_cfg.stream_dataset:
                    next(iter(train_ds)) # Check if the dataset is loaded correctly
                else:
                    train_ds[0] # Check if the dataset is loaded correctly
                combined_train_data.append(train_ds)
                break
            except Exception as e:
                last_error = e
                print(f"Rank {get_rank()} dataset load attempt {attempt}/6 failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(5 * attempt)
        else:
            print(f"Warning: Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {last_error}", flush=True)
            continue

    if not combined_train_data:
        raise ValueError("No valid datasets were loaded. Please check your dataset path and configurations.")

    train_ds = concatenate_datasets(combined_train_data)

    if not train_cfg.stream_dataset:
        train_ds = train_ds.shuffle(seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets


    if is_dist():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    # train_ds = train_ds.shuffle(buffer_size=10000, seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets

    val_size = int(train_cfg.val_size/get_world_size())
    print(f"Val size per GPU: {val_size}")

    if train_cfg.stream_dataset:
        val_ds = train_ds.take(val_size)
        train_ds = train_ds.skip(val_size)
    else:
        val_ds = train_ds.select(range(val_size))
        train_ds = train_ds.select(range(val_size, len(train_ds)))

    train_dataset = VQADataset(
        train_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )
    val_dataset = VQADataset(
        val_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )

    packing_num_sequences = 8
    train_dataset = ConstantLengthDataset(train_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=packing_num_sequences, queue_size=8,
                                        max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack)

    val_dataset = ConstantLengthDataset(val_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=packing_num_sequences, queue_size=8,
                                        max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=vqa_collator,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Warmup dataloaders to kickstart worker processes
    print("Warming up dataloaders...")
    iter_train_loader = iter(train_loader)
    iter_val_loader = iter(val_loader)
    next(iter_train_loader)
    next(iter_val_loader)
    print("Warmup complete.")

    return train_loader, val_loader, iter_train_loader, iter_val_loader

# Cosine learning rate schedule with warmup (from Karpathy)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def train(train_cfg, vlm_cfg):
    train_loader, val_loader, iter_train_loader, iter_val_loader = get_dataloaders(train_cfg, vlm_cfg)

    if is_dist():
        print("Rank", get_rank(), "Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
        if is_master():
            print("All workers have gotten dataloaders.")

    run_name = get_run_name(train_cfg, vlm_cfg)
    if train_cfg.log_wandb and is_master():
        import wandb
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM",
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg)
            },
            name=run_name,
        )
        # Define a custom x-axis for lmms-eval metrics
        lmms_eval_step = "<lmms-eval-step>"
        run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step)

    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    if is_master():
        print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Training summary{' (global)' if is_dist() else ''}: {-1*get_world_size()} samples, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Training summary per GPU: batch size {train_loader.batch_size}")
        print(f"Validation summary{' (global)' if is_dist() else ''}: {-1*get_world_size()} samples, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Validation summary per GPU: batch size {val_loader.batch_size}")

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = []
    if train_cfg.lr_mp > 0:
        param_groups.append({'params': list(model.MP.parameters()), 'lr': train_cfg.lr_mp})
    else:
        for p in list(model.MP.parameters()):
            p.requires_grad = False
    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({'params': list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_vision_backbone})
    else:
        for p in list(model.vision_encoder.parameters()):
            p.requires_grad = False
    if train_cfg.lr_language_backbone > 0:
        param_groups.append({'params': list(model.decoder.parameters()), 'lr': train_cfg.lr_language_backbone})
    else:
        for p in list(model.decoder.parameters()):
            p.requires_grad = False

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group['params']]

    device = (
        torch.device("npu") if _USE_NPU
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()

    print(f"Using device: {device}")
    model.to(device)

    if _USE_NPU:
        # SN-LM-Checkpoint L1a: test whether the reduced-memory attention
        # winner makes full activation retention feasible at batch_size=2.
        use_grad_ckpt = False
        model.decoder.gradient_checkpointing = use_grad_ckpt
        if is_master():
            print(f"NPU mode: gradient_checkpointing={use_grad_ckpt}, using npu_fusion_attention")

    if train_cfg.compile:
        model = torch.compile(model)
    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    epoch_times = []
    best_val_loss = float('inf')
    best_model_path = None
    logged_eval_steps = set()
    global_step = 0
    epoch = 0

    # Training stats accumulators
    accumulated_stats = {
        'tokens_per_second': [],
        'data_load_time': [],
        'fw_bw_time': [],
        'post_process_time': [],
        'images_per_sample': [],
    }
    benchmark_case = os.environ.get("NANOVLM_BENCHMARK_CASE")
    benchmark_warmup_steps = getattr(train_cfg, "benchmark_warmup_steps", 0)
    benchmark_warmup_start = None
    benchmark_measure_start = None
    benchmark_measured_loss_sum = 0.0
    benchmark_measured_loss_count = 0
    if benchmark_case and device.type in ("cuda", "npu"):
        _accelerator_synchronize()
        benchmark_warmup_start = time.perf_counter()
        if benchmark_warmup_steps == 0:
            benchmark_measure_start = benchmark_warmup_start

    gpu_teacher_dir = os.environ.get("NANOVLM_GPU_TEACHER_DIR")
    gpu_teacher_profiler = None
    if gpu_teacher_dir and is_master():
        trace_dir = os.path.join(gpu_teacher_dir, "rank0", "traces")
        summary_dir = os.path.join(gpu_teacher_dir, "rank0", "summaries")
        cuda_summary_path = os.path.join(summary_dir, "kernel_summary_cuda.txt")
        cpu_summary_path = os.path.join(summary_dir, "kernel_summary_cpu.txt")
        os.makedirs(trace_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)

        def save_gpu_teacher_trace(prof):
            prof.export_chrome_trace(os.path.join(trace_dir, f"trace_step_{prof.step_num}.json"))
            with open(cuda_summary_path, "w", encoding="utf-8") as f:
                f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
                f.write("\n")
            with open(cpu_summary_path, "w", encoding="utf-8") as f:
                f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
                f.write("\n")

        gpu_teacher_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=benchmark_warmup_steps * train_cfg.gradient_accumulation_steps,
                warmup=2,
                active=16,
                repeat=1,
            ),
            on_trace_ready=save_gpu_teacher_trace,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        gpu_teacher_profiler.start()

    npu_profile_dir = os.environ.get("NANOVLM_NPU_PROFILE_DIR")
    npu_profile_level = os.environ.get("NANOVLM_NPU_PROFILE_LEVEL", "").upper()
    npu_profiler = None
    if npu_profile_dir and device.type == "npu":
        import torch_npu

        rank_dir = os.path.join(npu_profile_dir, f"rank_{get_rank()}")
        os.makedirs(rank_dir, exist_ok=True)
        profile_kwargs = {
            "activities": [torch_npu.profiler.ProfilerActivity.NPU],
            "schedule": torch_npu.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1,
                skip_first=benchmark_warmup_steps * train_cfg.gradient_accumulation_steps,
            ),
            "on_trace_ready": torch_npu.profiler.tensorboard_trace_handler(rank_dir),
        }
        if npu_profile_level == "L1":
            profile_kwargs.update({
                "activities": [
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                "with_stack": True,
                "record_shapes": True,
                "profile_memory": True,
                "experimental_config": torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                ),
            })
        npu_profiler = torch_npu.profiler.profile(**profile_kwargs)
        npu_profiler.start()
        if is_master():
            print(
                f"NPU_PROFILE_START level={npu_profile_level or 'L0'} "
                f"dir={npu_profile_dir}",
                flush=True,
            )

    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        defer_loss_readback = bool(benchmark_case and not train_cfg.log_wandb)
        deferred_batch_losses = []
        data_load_start = time.time()

        print("Starting training loop")
        for i, batch in enumerate(synchronized_dataloader_step(iter_train_loader, is_dist(), PG_CPU)):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            detailed_telemetry = not benchmark_case or train_cfg.log_wandb
            if benchmark_case and detailed_telemetry and device.type in ("cuda", "npu"):
                _accelerator_synchronize()
            batch_start_time = time.perf_counter()
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask_cpu = batch["attention_mask"]
            attention_mask = attention_mask_cpu.to(device)
            data_load_time = time.time() - data_load_start

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            fw_bw_start = time.perf_counter()
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type in ['cuda', 'cpu', 'npu'] else torch.float16
            )
            with autocast_context:
                with context:
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()

            if benchmark_case and detailed_telemetry and device.type in ("cuda", "npu"):
                _accelerator_synchronize()
            fw_bw_time = time.perf_counter() - fw_bw_start
            post_process_start = time.perf_counter()
            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)

                param_group_idx = 0
                if train_cfg.lr_mp > 0:
                    adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_mp
                    param_group_idx += 1

                if train_cfg.lr_vision_backbone > 0:
                    adj_lr_vision_backbone = get_lr(global_step, train_cfg.lr_vision_backbone, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_vision_backbone
                    param_group_idx += 1

                if train_cfg.lr_language_backbone > 0:
                    adj_lr_language_backbone = get_lr(global_step, train_cfg.lr_language_backbone, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_language_backbone

                optimizer.step()
                optimizer.zero_grad()

            if defer_loss_readback:
                deferred_batch_losses.append(
                    (loss.detach(), global_step >= benchmark_warmup_steps)
                )
                batch_loss = None
            else:
                batch_loss = loss.item()
                if train_cfg.gradient_accumulation_steps > 1:
                    batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
                total_train_loss += batch_loss
                if benchmark_case and global_step >= benchmark_warmup_steps:
                    benchmark_measured_loss_sum += batch_loss
                    benchmark_measured_loss_count += 1

            # Token count is metadata already available on CPU. Avoid a device
            # reduction followed by D2H scalar extraction in every microbatch.
            if benchmark_case and not train_cfg.log_wandb:
                num_tokens = torch.sum(attention_mask_cpu).item()
            else:
                num_tokens = torch.sum(attention_mask).item()
            total_tokens_processed += num_tokens
            if benchmark_case and detailed_telemetry and device.type in ("cuda", "npu"):
                _accelerator_synchronize()
            post_process_time = time.perf_counter() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.perf_counter()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = get_world_size() * num_tokens / batch_duration  # Multiply by world size to get global tokens/s

            # Accumulate training stats
            accumulated_stats['tokens_per_second'].append(tokens_per_second)
            accumulated_stats['data_load_time'].append(data_load_time)
            accumulated_stats['fw_bw_time'].append(fw_bw_time)
            accumulated_stats['post_process_time'].append(post_process_time)
            accumulated_stats['images_per_sample'].extend(images_per_sample)

            if benchmark_case and detailed_telemetry and is_master():
                print(
                    "BENCH_STEP " + json.dumps({
                        "case": benchmark_case,
                        "phase": "measure" if global_step >= benchmark_warmup_steps else "warmup",
                        "optimizer_step_before": global_step,
                        "micro_step": i,
                        "is_update_step": is_update_step,
                        "batch_duration_s": batch_duration,
                        "fw_bw_time_s": fw_bw_time,
                        "post_process_time_s": post_process_time,
                        "data_load_time_s": data_load_time,
                        "tokens": num_tokens,
                        "tokens_per_second_global": tokens_per_second,
                        "loss": batch_loss,
                    }, sort_keys=True),
                    flush=True,
                )
            if gpu_teacher_profiler is not None:
                gpu_teacher_profiler.step()
            if npu_profiler is not None:
                npu_profiler.step()

            if train_cfg.eval_in_epochs and global_step % train_cfg.eval_interval == 0 and is_update_step:
                print("Starting evaluation")
                model.eval()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "npu":
                    torch.npu.empty_cache()
                with torch.no_grad():
                    total_val_loss = 0
                    val_batches = 0
                    for batch in synchronized_dataloader_step(iter_val_loader, is_dist(), PG_CPU):
                        if val_batches > 64:
                            print(f"Evaluated {val_batches} batches")
                            break
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with autocast_context:
                            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

                        total_val_loss += loss.item()
                        val_batches += 1

                    iter_val_loader = iter(val_loader)
                    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
                    avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss

                    checkpoint_path_step = ""
                    if is_master():
                        # Save a checkpoint for this evaluation step
                        checkpoint_path_step = os.path.join(vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}")
                        save_model = model.module if is_dist() else model # unwrap the model for saving if DDP
                        save_model.save_pretrained(save_directory=checkpoint_path_step)

                        if train_cfg.use_lmms_eval and global_step % (train_cfg.eval_interval*2) == 0:
                            # Submit evaluation job
                            cmd = f"sbatch eval.slurm {checkpoint_path_step} {global_step} {run_name} {train_cfg.lmms_eval_limit} {train_cfg.lmms_eval_tasks} {train_cfg.lmms_eval_batch_size}"
                            print(f"Submitting evaluation job: {cmd}")
                            subprocess.run(cmd, shell=True)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            best_model_path = checkpoint_path_step

                    if is_master():
                        print(f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")
                        if train_cfg.log_wandb:
                            run.log({"val_loss": avg_val_loss}, step=global_step)

                model.train()

            # These statistics are consumed only by W&B. Avoid nine Python
            # object collectives at each logging interval when logging is off.
            if (train_cfg.log_wandb
                and global_step % train_cfg.stats_log_interval == 0
                and len(accumulated_stats['tokens_per_second']) > 0
                and is_update_step):
                # ALL RANKS: Perform collective operations for training stats
                stats = {}
                for key in ['tokens_per_second', 'data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]  # Flatten list of lists
                        stats[f'avg_{key}'] = mean(all_values_flat)
                    else:
                        stats[f'avg_{key}'] = mean(accumulated_stats[key])

                for key in ['data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]
                        stats[f'max_{key}'] = max(all_values_flat)
                    else:
                        stats[f'max_{key}'] = max(accumulated_stats[key])

                if is_dist():
                    all_images_values = dist_gather(accumulated_stats['images_per_sample'])
                    all_images_flat = [item for sublist in all_images_values for item in sublist]
                    stats['min_images_per_sample'] = min(all_images_flat)
                else:
                    stats['min_images_per_sample'] = min(accumulated_stats['images_per_sample'])

                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    run.log({
                        **{f"training_stats/{key}": value for key, value in stats.items()},
                    }, step=global_step)

                    # Check for and log new lmms-eval results
                    eval_results_dir = os.path.join('eval_results', run_name)
                    if os.path.exists(eval_results_dir):
                        logged_results_count = 0
                        for result_file in os.listdir(eval_results_dir):
                            # Match only files like "step_1234.json" (no extra text)
                            match = re.fullmatch(r"step_(\d+)\.json", result_file)
                            if not match:
                                continue  # skip if the filename has extra text like taskname

                            try:
                                step = int(match.group(1))
                                if step not in logged_eval_steps:
                                    with open(os.path.join(eval_results_dir, result_file), 'r') as f:
                                        eval_data = json.load(f)

                                    lmms_results = eval_data.get('results', {})
                                    if lmms_results:
                                        metrics = {f"lmms_eval/{key}": value for key, value in lmms_results.items()}
                                        metrics[lmms_eval_step] = eval_data['global_step']
                                        if logged_results_count > 0:
                                            print(f"Logging more than one lmms-eval result for step {global_step}, try to avoid this.")
                                        run.log(metrics, step=global_step + logged_results_count)
                                        logged_results_count += 1
                                        print(f"Logged lmms-eval results from step {eval_data['global_step']}")

                                    logged_eval_steps.add(step)
                            except (ValueError, KeyError, json.JSONDecodeError) as e:
                                print(f"Warning: Could not process eval result file {result_file}. Error: {e}")
                                continue

                # ALL RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # Loss reduction is required only by W&B; it does not participate
            # in optimization and otherwise adds a scalar HCCL synchronization.
            if is_update_step and train_cfg.log_wandb:
                # ALL RANKS: gather loss from all ranks if DDP
                if is_dist():
                    batch_loss_gathered = dist_mean_scalar(batch_loss)
                else:
                    batch_loss_gathered = batch_loss
                if is_master():
                    run.log({
                        "batch_loss": batch_loss_gathered,
                        **({"grad_norm": grad_norm} if train_cfg.max_grad_norm is not None else {})
                    }, step=global_step)

            if is_update_step:
                global_step += 1
                if benchmark_case and global_step == benchmark_warmup_steps:
                    if device.type in ("cuda", "npu"):
                        _accelerator_synchronize()
                    benchmark_measure_start = time.perf_counter()
                    if is_master():
                        print(
                            f"BENCH_WARMUP case={benchmark_case} steps={benchmark_warmup_steps} "
                            f"elapsed_s={benchmark_measure_start - benchmark_warmup_start:.6f}",
                            flush=True,
                        )
                if global_step >= train_cfg.max_training_steps:
                    break
            data_load_start = time.time()

        iter_train_loader = iter(train_loader)
        if defer_loss_readback:
            # Preserve the exact scalar reporting order while replacing one
            # D2H synchronization per microbatch with one batched readback.
            deferred_loss_values = torch.stack(
                [entry[0] for entry in deferred_batch_losses]
            ).cpu().tolist()
            if train_cfg.gradient_accumulation_steps > 1:
                deferred_loss_values = [
                    value * train_cfg.gradient_accumulation_steps
                    for value in deferred_loss_values
                ]
            total_train_loss = sum(deferred_loss_values)
            measured_loss_values = [
                value for value, entry in zip(deferred_loss_values, deferred_batch_losses)
                if entry[1]
            ]
            benchmark_measured_loss_sum += sum(measured_loss_values)
            benchmark_measured_loss_count += len(measured_loss_values)
        avg_train_loss = total_train_loss / i
        # gather average batch loss from all ranks if DDP
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                run.log({"epoch_loss": avg_train_loss,
                         "epoch_duration": epoch_duration,
                         "epoch_tokens_per_second": epoch_tokens_per_second})

            print(f"Epoch: {epoch}, Step: {global_step}/{train_cfg.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

    if gpu_teacher_profiler is not None:
        gpu_teacher_profiler.stop()
        print(f"GPU_TEACHER_PROFILE_COMPLETE dir={gpu_teacher_dir}", flush=True)
    if npu_profiler is not None:
        npu_profiler.stop()
        if is_master():
            print(
                f"NPU_PROFILE_COMPLETE level={npu_profile_level or 'L0'} "
                f"dir={npu_profile_dir}",
                flush=True,
            )

    if benchmark_case and benchmark_measure_start is not None:
        if device.type in ("cuda", "npu"):
            _accelerator_synchronize()
        benchmark_measure_elapsed = time.perf_counter() - benchmark_measure_start
        local_measured_loss = benchmark_measured_loss_sum / benchmark_measured_loss_count
        measured_train_loss = mean(dist_gather(local_measured_loss)) if is_dist() else local_measured_loss
        if is_master():
            print(
                f"BENCH_MEASURE case={benchmark_case} "
                f"steps={train_cfg.max_training_steps - benchmark_warmup_steps} "
                f"microbatches={benchmark_measured_loss_count} "
                f"elapsed_s={benchmark_measure_elapsed:.6f} "
                f"train_loss={measured_train_loss:.8f}",
                flush=True,
            )

    if getattr(train_cfg, "eval_at_end", False):
        del train_loader, iter_train_loader, optimizer, all_params
        import gc; gc.collect()
        if _USE_NPU:
            torch.npu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        model.eval()
        eval_model = model.module if is_dist() else model
        if hasattr(eval_model, "_orig_mod"):
            eval_model = eval_model._orig_mod
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in synchronized_dataloader_step(iter(val_loader), is_dist(), PG_CPU):
                if val_batches >= 64:
                    break
                images = batch["images"]
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if device.type in ['cuda', 'cpu', 'npu'] else torch.float16,
                ):
                    _, val_loss = eval_model(input_ids, images, attention_mask=attention_mask, targets=labels)
                total_val_loss += val_loss.item()
                val_batches += 1
                del images, input_ids, labels, attention_mask, val_loss
        avg_val_loss = total_val_loss / val_batches
        avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
        if is_master():
            print(f"FINAL_VAL_LOSS value={avg_val_loss:.8f} batches_per_rank={val_batches}", flush=True)

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in the config!)
        if vlm_cfg.hf_repo_name is not None and best_model_path:
            print(f"Training complete. Pushing best model from {best_model_path} to Hugging Face Hub...")
            hf_model = VisionLanguageModel.from_pretrained(best_model_path)
            hf_model.push_to_hub(vlm_cfg.hf_repo_name)

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()

def main():
    global PG_CPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_vision_backbone', type=float, help='Learning rate for the vision backbone')
    parser.add_argument('--lr_language_backbone', type=float, help='Learning rate for the language backbone')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=None, help='Use torch.compile to optimize the model')
    parser.add_argument('--log_wandb', type=bool, help='Log to wandb')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
    parser.add_argument('--no_log_wandb', action='store_true', help='Do not log to wandb')
    parser.add_argument('--train_dataset_path', type=str, help='Train dataset path')
    parser.add_argument('--train_dataset_files', type=str, help='Local parquet file or glob used instead of a Hub dataset')
    parser.add_argument('--relevance_min_rating', type=int, help='Minimum relevance rating of images per sample')
    parser.add_argument('--image_correspondence_min_rating', type=int, help='Minimum image correspondence rating of images per sample')
    parser.add_argument('--visual_dependency_min_rating', type=int, help='Minimum visual dependency rating of images per sample')
    parser.add_argument('--formatting_min_rating', type=int, help='Minimum formatting rating of images per sample')
    parser.add_argument('--max_training_steps', type=int, help='Override the number of optimizer steps')
    parser.add_argument('--stats_log_interval', type=int, help='Override the training statistics interval')
    parser.add_argument('--no_eval', action='store_true', help='Disable validation/checkpointing during training')
    parser.add_argument('--no_lmms_eval', action='store_true', help='Disable lmms-eval job submission')
    parser.add_argument('--val_size', type=int, help='Override validation set size')
    parser.add_argument('--vit_model_type', type=str, help='Hub id or local directory for the vision backbone')
    parser.add_argument('--lm_model_type', type=str, help='Hub id or local directory for the language backbone')
    parser.add_argument('--lm_tokenizer', type=str, help='Hub id or local directory for the tokenizer')
    parser.add_argument('--eval_at_end', action='store_true', help='Run loss-only validation after the final training step')
    parser.add_argument('--benchmark_warmup_steps', type=int, default=0, help='Optimizer steps excluded from measured benchmark statistics')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Override gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, help='Override per-device batch size')

    args = parser.parse_args()

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_vision_backbone is not None:
        train_cfg.lr_vision_backbone = args.lr_vision_backbone
    if args.lr_language_backbone is not None:
        train_cfg.lr_language_backbone = args.lr_language_backbone
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.no_log_wandb is True:
        train_cfg.log_wandb = False
    if args.train_dataset_path is not None:
        train_cfg.train_dataset_path = args.train_dataset_path
    if args.train_dataset_files is not None:
        train_cfg.train_dataset_files = args.train_dataset_files
    if args.relevance_min_rating is not None:
        train_cfg.relevance_min_rating = args.relevance_min_rating
    if args.image_correspondence_min_rating is not None:
        train_cfg.image_correspondence_min_rating = args.image_correspondence_min_rating
    if args.visual_dependency_min_rating is not None:
        train_cfg.visual_dependency_min_rating = args.visual_dependency_min_rating
    if args.formatting_min_rating is not None:
        train_cfg.formatting_min_rating = args.formatting_min_rating
    if args.max_training_steps is not None:
        train_cfg.max_training_steps = args.max_training_steps
    if args.stats_log_interval is not None:
        train_cfg.stats_log_interval = args.stats_log_interval
    if args.no_eval:
        train_cfg.eval_in_epochs = False
    if args.no_lmms_eval:
        train_cfg.use_lmms_eval = False
    if args.val_size is not None:
        train_cfg.val_size = args.val_size
    if args.vit_model_type is not None:
        vlm_cfg.vit_model_type = args.vit_model_type
    if args.lm_model_type is not None:
        vlm_cfg.lm_model_type = args.lm_model_type
    if args.lm_tokenizer is not None:
        vlm_cfg.lm_tokenizer = args.lm_tokenizer
    if args.eval_at_end:
        train_cfg.eval_at_end = True
    train_cfg.benchmark_warmup_steps = args.benchmark_warmup_steps
    if args.gradient_accumulation_steps is not None:
        train_cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        PG_CPU = dist.new_group(backend="gloo")   # host‑RAM, zero GPU allocations

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg)

    if is_dist():
        destroy_dist()
        # See train.py: all distributed work is complete; avoid the known
        # torch_npu static-destruction abort after long multi-rank runs.
        if _USE_NPU:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

if __name__ == "__main__":
    main()
