import os
from types import SimpleNamespace

import torch
import torch_npu

from models.language_model import RotaryEmbedding


def main() -> None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    cfg = SimpleNamespace(
        lm_hidden_dim=960,
        lm_n_heads=15,
        lm_re_base=10_000.0,
        lm_max_position_embeddings=4096,
        lm_attn_scaling=1.0,
    )
    rope = RotaryEmbedding(cfg).to(device)
    cases = (
        (0, 4096),
        (113, 1024),
        (4090, 32),
    )
    for start_pos, length in cases:
        position_ids = torch.arange(
            start_pos, start_pos + length, device=device
        ).unsqueeze(0).expand(2, -1)
        fallback_cos, fallback_sin = rope(position_ids)
        host_cos, host_sin = rope(
            position_ids, max_seq=start_pos + length
        )
        torch.npu.synchronize()
        cos_diff = (fallback_cos - host_cos).abs().max().item()
        sin_diff = (fallback_sin - host_sin).abs().max().item()
        passed = cos_diff == 0.0 and sin_diff == 0.0
        print(
            "ROPE_HOST_BOUND_EQUIV "
            f"start_pos={start_pos} length={length} pass={passed} "
            f"cos_max_diff={cos_diff:.9g} sin_max_diff={sin_diff:.9g}",
            flush=True,
        )
        if not passed:
            raise AssertionError("host-known RoPE bound is not bit-exact")
    print("ROPE_HOST_BOUND_PROBE_PASS", flush=True)


if __name__ == "__main__":
    main()
