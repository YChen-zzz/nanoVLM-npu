import importlib.util
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu


ROOT = os.environ["NANOVLM_ROOT"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def run_case(seq_len, base_mod, opt_mod, device):
    cfg = SimpleNamespace(
        lm_hidden_dim=960,
        lm_inter_dim=2560,
        lm_rms_eps=1e-5,
        lm_re_base=10000.0,
        lm_max_position_embeddings=8192,
        lm_attn_scaling=1.0,
        lm_n_heads=15,
        lm_n_kv_heads=5,
        lm_dropout=0.0,
        lm_n_blocks=2,
        lm_vocab_size=128,
        lm_use_tokens=False,
        lm_tie_weights=False,
    )
    torch.manual_seed(20260831 + seq_len)
    torch.npu.manual_seed_all(20260831 + seq_len)
    base = base_mod.LanguageModel(cfg).to(device).train()
    opt = opt_mod.LanguageModel(cfg).to(device).train()
    opt.load_state_dict(base.state_dict(), strict=True)

    x_seed = torch.randn(2, seq_len, cfg.lm_hidden_dim, device=device)
    mask = torch.ones(2, seq_len, dtype=torch.long, device=device)
    mask[0, : max(1, seq_len // 8)] = 0
    mask[1, : max(1, 7 * seq_len // 8)] = 0

    x0 = x_seed.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        y0, _ = base(x0, attention_mask=mask)
        loss0 = y0.float().square().mean()
    loss0.backward()
    torch.npu.synchronize()
    out0 = y0.detach()
    grad0 = x0.grad.detach()
    param_grads0 = {
        name: p.grad.detach().clone()
        for name, p in base.named_parameters()
        if p.grad is not None
    }

    x1 = x_seed.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        y1, _ = opt(x1, attention_mask=mask)
        loss1 = y1.float().square().mean()
    loss1.backward()
    torch.npu.synchronize()

    output_max = (out0 - y1.detach()).abs().max().item()
    output_cos = cosine(out0, y1.detach())
    input_grad_max = (grad0 - x1.grad.detach()).abs().max().item()
    input_grad_cos = cosine(grad0, x1.grad.detach())
    param_grad_max = 0.0
    param_grad_cos = 1.0
    for name, p in opt.named_parameters():
        if p.grad is None:
            continue
        ref = param_grads0[name]
        param_grad_max = max(param_grad_max, (ref - p.grad).abs().max().item())
        param_grad_cos = min(param_grad_cos, cosine(ref, p.grad))

    with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
        repeat1, _ = opt(x_seed, attention_mask=mask)
        repeat2, _ = opt(x_seed, attention_mask=mask)
    torch.npu.synchronize()
    deterministic = torch.equal(repeat1, repeat2)

    print(
        "MASK_EQUIV "
        f"seq={seq_len} loss_abs={abs(loss0.item() - loss1.item()):.9g} "
        f"output_max={output_max:.9g} output_cos={output_cos:.9g} "
        f"input_grad_max={input_grad_max:.9g} input_grad_cos={input_grad_cos:.9g} "
        f"param_grad_max={param_grad_max:.9g} param_grad_cos={param_grad_cos:.9g} "
        f"deterministic={deterministic}",
        flush=True,
    )
    assert abs(loss0.item() - loss1.item()) <= 1e-7
    assert output_max <= 1e-6 and output_cos >= 0.999999
    # Cosine is ill-conditioned when an individual gradient tensor is almost
    # zero. In that regime the absolute-error gate is the meaningful test.
    assert input_grad_max <= 1e-6 and (input_grad_max <= 1e-10 or input_grad_cos >= 0.999999)
    assert param_grad_max <= 1e-6 and (param_grad_max <= 1e-10 or param_grad_cos >= 0.999999)
    assert deterministic


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    torch.npu.set_compile_mode(jit_compile=False)
    base_mod = load_module("stage2_mask_base", os.path.join(ROOT, "models/language_model.py"))
    opt_mod = load_module(
        "stage2_mask_opt",
        os.path.join(ROOT, "ablations/stage2_mask_shared/language_model.py"),
    )
    for seq_len in (1024, 4096):
        run_case(seq_len, base_mod, opt_mod, torch.device("npu", local_rank))
    dist.barrier()
    if local_rank == 0:
        print("MASK_EQUIV_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
