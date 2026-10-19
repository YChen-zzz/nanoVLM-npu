import json
import sys


def size_mb(info):
    bw = info.get("Communication Bandwidth Info", {})
    return max((v.get("Transit Size(MB)", 0.0) for v in bw.values()), default=0.0)


data = json.load(open(sys.argv[1], encoding="utf-8"))
for step, groups in data.items():
    ops = groups.get("collective", {})
    print(f"[{step}] ops={len(ops)}")
    for name, info in ops.items():
        timing = info.get("Communication Time Info", {})
        print(
            f"{name.split('@')[0]:42s} "
            f"size={size_mb(info):9.6f}MB "
            f"elapsed={timing.get('Elapse Time(ms)', 0):9.3f}ms "
            f"wait={timing.get('Wait Time(ms)', 0):9.3f}ms"
        )
