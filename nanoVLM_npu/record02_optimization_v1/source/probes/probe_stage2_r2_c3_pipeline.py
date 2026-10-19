import os

import torch
import torch.distributed as dist

from data.data_utils import synchronized_dataloader_step


def valid(value):
    return {"input_ids": [value], "images": [[torch.tensor(value)]]}


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    if rank == 0:
        source = iter([{}, valid(10), valid(11), valid(12)])
        expected = [10, 11]
    else:
        source = iter([valid(20), {}, valid(21)])
        expected = [20, 21]

    actual = [
        batch["input_ids"][0]
        for batch in synchronized_dataloader_step(
            source, True, dist.group.WORLD
        )
    ]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, actual)
    passed = actual == expected and gathered == [[10, 11], [20, 21]]
    print(
        f"PIPELINED_VALIDITY_EQUIV rank={rank} pass={passed} "
        f"actual={actual} gathered={gathered}",
        flush=True,
    )
    if not passed:
        raise AssertionError("pipelined validity iterator changed sequence semantics")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
