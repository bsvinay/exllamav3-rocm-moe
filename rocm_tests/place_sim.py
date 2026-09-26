# Offline placement simulation on route_trace.pt: CPU share of decode picks for S GPU slots per layer
import os, sys, json, torch
tr = torch.load(os.path.expanduser("~/route_trace.pt"))
stats = json.load(open(sys.argv[1]))
S = int(sys.argv[2]) if len(sys.argv) > 2 else 166
E = 512
def share(sel_by_layer, gpu_by_layer):
    miss = tot = 0
    for k, sel in sel_by_layer.items():
        g = torch.zeros(E, dtype = torch.bool); g[gpu_by_layer[k]] = True
        miss += int((~g[sel.view(-1)]).sum()); tot += sel.numel()
    return miss / tot
static = {k: torch.topk(torch.tensor(v, dtype = torch.float), S).indices for k, v in stats.items()}
for name, layers in tr.items():
    n = next(iter(layers.values())).shape[0]; h = n // 2
    first = {k: v[:h] for k, v in layers.items()}; second = {k: v[h:] for k, v in layers.items()}
    res = {"static": share(second, static)}
    for mix in (0.25, 0.5, 1.0, 4.0):
        # blend: normalized static prior + mix * normalized first-half counts
        pl = {}
        for k in layers:
            st = torch.tensor(stats[k], dtype = torch.float); st /= st.sum()
            c = torch.bincount(first[k].view(-1), minlength = E).float(); c /= c.sum()
            pl[k] = torch.topk(st + mix * c, S).indices
        res[f"adapt{mix}"] = share(second, pl)
    orc = {k: torch.topk(torch.bincount(v.view(-1), minlength = E).float(), S).indices for k, v in second.items()}
    res["oracle(2nd half)"] = share(second, orc)
    print(f"{name} ({n} tok): " + ", ".join(f"{a} {b:.3f}" for a, b in res.items()))
# cross-prompt: adapt on code_a, evaluate code_b / chat
if "code_a" in tr:
    for tgt in [k for k in tr if k != "code_a"]:
        for mix in (0.5, 1.0):
            pl = {}
            for k in tr["code_a"]:
                st = torch.tensor(stats[k], dtype = torch.float); st /= st.sum()
                c = torch.bincount(tr["code_a"][k].view(-1), minlength = E).float(); c /= c.sum()
                pl[k] = torch.topk(st + mix * c, S).indices
            print(f"adapt on code_a (mix {mix}) -> {tgt}: {share(tr[tgt], pl):.3f} (static {share(tr[tgt], static):.3f})")
