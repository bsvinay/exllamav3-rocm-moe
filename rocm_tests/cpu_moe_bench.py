# CPU expert GEMM throughput at Qwen3.8-Flash-Next shapes (hid 2560, inter 640, 512 experts, top-10)
import os, sys, time, torch
from exllamav3.ext import exllamav3_ext as ext
K = int(os.environ.get("KB", 3)); NL = int(os.environ.get("NL", 8)); swz = int(os.environ.get("SWZ", 1))
hid, inter, E, topk = 2560, 640, 512, 10
g = torch.Generator().manual_seed(1)
def tr(k, n):
    t = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype = torch.int16, generator = g)
    if swz and K != 8:
        tk, tn, ps = t.shape
        t = t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).contiguous().view(tk, tn, ps)
    return t
def sv(n): return ((torch.randint(0, 2, (n,), generator = g).float() * 2 - 1) * 0.015).half()
layers = []
for l in range(NL):
    gt, gs, gv, ut, us, uv, dt, ds, dv = ([] for _ in range(9))
    for e in range(E):
        gt.append(tr(hid, inter)); gs.append(sv(hid)); gv.append(sv(inter))
        ut.append(tr(hid, inter)); us.append(sv(hid)); uv.append(sv(inter))
        dt.append(tr(inter, hid)); ds.append(sv(inter)); dv.append(sv(hid))
    layers.append(ext.exl3_moe_cpu_make_layer(gt, gs, gv, ut, us, uv, dt, ds, dv, [], [], [], 0, 0.0, swz))
    del gt, ut, dt
expert_bytes = 3 * hid * inter * K / 8
print(f"K={K} layers={NL} ({NL*E*expert_bytes/1e9:.1f} GB) swz={swz}", flush = True)
for th in [int(x) for x in os.environ.get("TH", "8,12,16,24,32").split(",")]:
    for T in (1, 2, 4, 8):
        x = torch.randn(T, hid).half()
        out = torch.zeros(T, hid, dtype = torch.float32)
        w = torch.full((T, topk), 0.1).half()
        ts = []; uniq = 0
        for it in range(6 * NL):
            l = layers[it % NL]
            sel = torch.stack([torch.randperm(E, generator = g)[:topk] for _ in range(T)]).int()
            uniq += len(torch.unique(sel))
            t0 = time.perf_counter(); ext.exl3_moe_cpu_forward(l, x, sel, w, out, th); ts.append(time.perf_counter() - t0)
        ts = ts[NL:]; m = sorted(ts)[len(ts) // 2]
        ue = uniq / (6 * NL)
        print(f"th {th:2d} T {T}: {m*1e6:7.1f} us/layer  uniq experts {ue:5.1f}  {ue*expert_bytes/m/1e9:6.1f} GB/s", flush = True)
