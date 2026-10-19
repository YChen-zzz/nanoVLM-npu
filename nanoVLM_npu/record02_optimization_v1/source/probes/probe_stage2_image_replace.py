import os
import statistics
import time

import torch
import torch_npu


def baseline(input_ids, token_embd, image_embd, image_token_id):
    out = token_embd.clone()
    mask = input_ids == image_token_id
    out[mask] = image_embd.reshape(-1, image_embd.size(-1)).to(out.dtype)
    return out


def masked_scatter(input_ids, token_embd, image_embd, image_token_id):
    mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(token_embd)
    source = image_embd.reshape(-1).to(token_embd.dtype)
    return torch.masked_scatter(token_embd, mask, source)


def dense_gather_where(input_ids, token_embd, image_embd, image_token_id):
    mask = input_ids == image_token_id
    flat_mask = mask.reshape(-1)
    slots = torch.cumsum(flat_mask, dim=0, dtype=torch.int32).sub_(1).clamp_min_(0)
    source = image_embd.reshape(-1, image_embd.size(-1)).to(token_embd.dtype)
    replacement = source.index_select(0, slots).view_as(token_embd)
    return torch.where(mask.unsqueeze(-1), replacement, token_embd)


def make_case(device, dtype=torch.bfloat16):
    batch, seq, dim = 2, 4096, 960
    image_token_id = 49151
    ids = torch.arange(batch * seq, device=device, dtype=torch.int64).reshape(batch, seq)
    ids = ids.remainder_(32000)
    # Three images, deliberately split and unevenly distributed across samples.
    ids[0, 137:265] = image_token_id
    ids[1, 17:81] = image_token_id
    token = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    image = torch.randn(3, 64, dim, device=device, dtype=dtype)
    weight = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    return ids, token, image, weight, image_token_id


def check_equivalence(name, fn, case):
    ids, token0, image0, weight, image_token_id = case
    token_a = token0.detach().clone().requires_grad_(True)
    image_a = image0.detach().clone().requires_grad_(True)
    token_b = token0.detach().clone().requires_grad_(True)
    image_b = image0.detach().clone().requires_grad_(True)
    ref = baseline(ids, token_a, image_a, image_token_id)
    out = fn(ids, token_b, image_b, image_token_id)
    ref.backward(weight)
    out.backward(weight)
    torch.npu.synchronize()
    diffs = {
        'output': (ref - out).abs().max().item(),
        'token_grad': (token_a.grad - token_b.grad).abs().max().item(),
        'image_grad': (image_a.grad - image_b.grad).abs().max().item(),
    }
    passed = all(value == 0.0 for value in diffs.values())
    print(f"IMAGE_REPLACE_EQUIV name={name} pass={passed} " + ' '.join(f'{k}={v:.9g}' for k, v in diffs.items()), flush=True)
    if not passed:
        raise AssertionError(f'{name} is not bit-exact: {diffs}')


def benchmark(name, fn, case, warmup=8, repeats=25):
    ids, token0, image0, weight, image_token_id = case
    samples = []
    for i in range(warmup + repeats):
        token = token0.detach().clone().requires_grad_(True)
        image = image0.detach().clone().requires_grad_(True)
        torch.npu.synchronize()
        begin = time.perf_counter()
        out = fn(ids, token, image, image_token_id)
        out.backward(weight)
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - begin) * 1000
        if i >= warmup:
            samples.append(elapsed_ms)
    print(
        f"IMAGE_REPLACE_BENCH name={name} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} max_ms={max(samples):.6f} repeats={repeats}",
        flush=True,
    )


def main():
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.npu.set_device(rank)
    device = torch.device('npu', rank)
    torch.manual_seed(20260831)
    torch.npu.manual_seed_all(20260831)
    case = make_case(device)
    candidates = [('masked_scatter', masked_scatter), ('dense_gather_where', dense_gather_where)]
    for name, fn in candidates:
        check_equivalence(name, fn, case)
    benchmark('baseline_index_put', baseline, case)
    for name, fn in candidates:
        benchmark(name, fn, case)
    print('IMAGE_REPLACE_PROBE_PASS', flush=True)


if __name__ == '__main__':
    main()
