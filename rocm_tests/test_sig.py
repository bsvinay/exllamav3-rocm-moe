import torch
from exllamav3.ext import exllamav3_ext as ext
torch.manual_seed(0)
for bsz in (1, 4, 40):
    dim = 2560
    x = torch.randn(bsz, dim, device = "cuda")          # shared expert output (fp32)
    y = torch.randn(bsz, dim, device = "cuda").half()   # mlp input
    w = (torch.randn(dim, 1, device = "cuda") * 0.02).half()
    z0 = torch.randn(bsz, dim, device = "cuda")
    ref = z0 + x * torch.sigmoid(y.float() @ w.float())
    z = z0.clone(); ext.add_sigmoid_gate_proj(x, y, z, w)
    print(bsz, "proj rel err", ((z - ref).norm() / ref.norm()).item())
    g = (y.float() @ w.float()).half()
    z = z0.clone(); ext.add_sigmoid_gate(x, g, z)
    print(bsz, "gate rel err", ((z - ref).norm() / ref.norm()).item())
