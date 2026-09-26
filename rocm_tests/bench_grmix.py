# gr_mix (GatedResidual decode mix) timing at Qwen3.8-Flash-Next shapes, 48 weight sets rotated (cold L2/MALL)
import torch, sys
from exllamav3.ext import exllamav3_ext as ext
H, D, LR = 4, 2560, 320
NS = 48
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1
torch.manual_seed(0)
sets = []
for i in range(NS):
    post = i % 2 == 0
    M = LR + (H if post else 0)
    fn = (torch.randn(M, H * D, device = "cuda") * 0.01).half()
    upt = (torch.randn(H, D // 4, LR, 4, device = "cuda") * 0.05).half()
    w = (torch.rand(H * D, device = "cuda") + 0.5).half()
    sets.append((fn, upt, w, post, M))
st = torch.randn(R, H, D, device = "cuda")
mixed = torch.empty(R, D, device = "cuda", dtype = torch.half)
postb = torch.empty(R, H, device = "cuda")
dots = torch.empty(R, LR + H + 1, H, device = "cuda")
def run(i):
    fn, upt, w, post, M = sets[i % NS]
    ext.gr_mix(st, fn, upt, w, 1e-6, dots[:, :M + 1], postb if post else None, mixed)
for i in range(96): run(i)
torch.cuda.synchronize()
e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
n = 960
e0.record()
for i in range(n): run(i)
e1.record(); torch.cuda.synchronize()
us = e0.elapsed_time(e1) * 1000 / n
mb = (sets[0][0].numel() + sets[0][1].numel()) * 2 / 1e6
print(f"R={R}: gr_mix {us:.1f} us/call, {mb:.1f} MB -> {mb * 1e-3 / (us * 1e-6):.0f} GB/s")
# reference check vs torch
fn, upt, w, post, M = sets[0]
ext.gr_mix(st, fn, upt, w, 1e-6, dots[:, :M + 1], postb, mixed)
x = st
normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-6) * w.float().view(H, D)
wf = fn.float() / w.float()   # fn folds w in: recover down/inject
flat = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-6)).flatten(-2)
dm = flat @ fn.float().t()
t = torch.nn.functional.silu(dm[:, :LR] / H)
up = upt.permute(0, 1, 3, 2).reshape(H * D, LR).float()
g = torch.sigmoid(t @ up.t())
ref = (g.view(R, H, D) * normed).mean(1)
print("rel err", ((mixed.float() - ref).norm() / ref.norm()).item())
