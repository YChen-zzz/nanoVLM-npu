import torch
import torch.distributed as dist
from collections.abc import Iterator

def _get_current_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.npu.current_device()
    except ImportError:
        pass
    return torch.cuda.current_device()

def _is_batch_valid(batch):
    """
    Check if a batch is valid for training/evaluation.
    A valid batch must have input_ids and at least one image.
    """
    if not batch:
        return False
    # The collator can return a batch with empty lists
    if len(batch['input_ids']) == 0:
        return False
    
    if len(batch['images']) == 0:
        return False
    
    # `images` is a list of lists of tensors. Check that at least one image is not None.
    if len([img for sublist in batch['images'] for img in sublist]) == 0:
        # During training, not having images creates gradients computed without all model parameters.
        # This creates deadlocks in DDP.
        return False

    return True


def synchronized_dataloader_step(train_loader, is_dist, process_group=None):
    """
    Create a synchronized iterator that handles uneven data distribution in DDP.
    All ranks will stop when the first rank runs out of data.
    This happens because when packing a presharded dataset, a rank might have less groups than the others.
    It also handles cases where a collator returns an empty/invalid batch on some ranks,
    by ensuring all ranks skip the invalid batch and attempt to fetch a new one.
    """
    if not is_dist:
        # For single GPU, we don't need synchronization, just filter invalid batches.
        for batch in train_loader:
            if _is_batch_valid(batch):
                yield batch
        return
    
    # For DDP, we need synchronization.
    if isinstance(train_loader, Iterator):
        train_iter = train_loader
    else:
        train_iter = iter(train_loader)
    
    while True:
        is_valid = False
        try:
            while not is_valid:
                batch = next(train_iter)
                is_valid = _is_batch_valid(batch)
            has_data_value = 1
        except StopIteration:
            batch = None
            has_data_value = 0

        # Keep this control-plane collective off the NPU stream when a CPU
        # process group is available. The MIN result and stop semantics are
        # unchanged, while the following .item() no longer drains NPU work.
        sync_device = "cpu" if process_group is not None else _get_current_device()
        has_data = torch.tensor(has_data_value, device=sync_device)
        
        # We synchronize across all ranks. If any rank is out of data, all ranks stop.
        dist.all_reduce(has_data, op=dist.ReduceOp.MIN, group=process_group)
        
        if has_data.item() == 0:
            # At least one rank is out of data. All ranks should stop.
            break
        yield batch
    return None
