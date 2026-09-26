# Decode-vs-prefill consistency: logits of token-by-token cached decode must match the no-cache
# prefill logits at the same positions. With --modules, report the per-module output error of the
# first decode step (first module that diverges = the broken decode kernel).
import os, sys, math, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 0)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--prefix", type = int, default = 200)
    ap.add_argument("--steps", type = int, default = 32)
    ap.add_argument("--chunk", type = int, default = 1, help = "tokens per decode forward")
    ap.add_argument("--modules", action = "store_true")
    ap.add_argument("--sub", type = int, default = -1, help = "hook the internals of model.modules[i]")
    ap.add_argument("--leakcheck", action = "store_true")
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    if args.mcs: config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    text = open(os.path.expanduser("~/wikitext2_test.txt")).read()[5000:40000]
    ids = tok.encode(text)[:, :args.prefix + args.steps]
    N = ids.shape[1]

    rec = {}
    tag = [None]
    if args.modules:
        for i, m in enumerate(model.modules):
            def mk(f, i, m):
                def w(x, params, *a, **kw):
                    y = f(x, params, *a, **kw)
                    if tag[0] is not None and isinstance(y, torch.Tensor):
                        rec[(tag[0], i)] = (type(m).__name__, getattr(m, "key", ""), y.detach().float().cpu())
                    return y
                return w
            m.forward = mk(m.forward, i, m)

    sub_rec = {}
    if args.sub >= 0:
        blk = model.modules[args.sub]
        hooks = []
        for an in ("attn_hc", "attn", "mlp_hc", "mlp"):
            o = getattr(blk, an, None)
            if o is None: continue
            for mn in ("mix", "apply_", "forward"):
                if hasattr(o, mn) and (mn != "forward" or an in ("attn", "mlp")):
                    hooks.append((f"{an}.{mn}", o, mn))
        mm = blk.mlp
        if os.environ.get("DD_MOE"):
            hooks += [("mlp.routing_fn", mm, "routing_fn"), ("mlp.cpu_split_submit", mm, "cpu_split_submit"),
                      ("mlp.cpu_split_combine", mm, "cpu_split_combine")]
            if mm.shared_experts is not None:
                hooks.append(("mlp.shared.forward", mm.shared_experts, "forward"))
        for name, o, mn in hooks:
            def mk(f, name):
                def w(*a, **kw):
                    if tag[0] is not None and name == "mlp.cpu_split_combine":
                        sub_rec.setdefault((tag[0], name + ".in_gpu", 0), a[0].detach().float().cpu().clone())
                    y = f(*a, **kw)
                    if tag[0] is not None and name == "mlp.cpu_split_combine" and isinstance(a[1], torch.Tensor):
                        sub_rec.setdefault((tag[0], name + ".in_cpu", 0), a[1].detach().float().cpu().clone())
                    if tag[0] is not None:
                        ys = y if isinstance(y, tuple) else (y,)
                        for k, t in enumerate(ys):
                            if isinstance(t, torch.Tensor):
                                sub_rec.setdefault((tag[0], name, k), t.detach().float().cpu().clone())
                    return y
                return w
            setattr(o, mn, mk(getattr(o, mn), name))

    if args.leakcheck:
        g = torch.Generator().manual_seed(0)
        rid = torch.randint(1000, 100000, (1, 512), generator = g)
        lg = model.forward(rid, {"attn_mode": "flash_attn_nc"}).float().cpu()[0]
        lp = torch.log_softmax(lg[:-1], -1).gather(1, rid[0, 1:].unsqueeze(1))
        print(f"random-token ppl (should be huge): {math.exp(-lp.mean().item()):.1f}")

    tag[0] = "ref"
    ref = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float().cpu()[0]
    tag[0] = None

    P = args.prefix
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
    model.prefill(input_ids = ids[:, :P], params = params)
    rs = params.get("recurrent_states")
    outs = []
    pos = P
    first = True
    while pos < N:
        c = min(args.chunk, N - pos)
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 4096), "recurrent_states": rs}
        tag[0] = "dec" if first else None
        lg = model.forward(input_ids = ids[:, pos:pos + c], params = p).float().cpu()[0]
        tag[0] = None
        first = False
        outs.append(lg)
        pos += c
    dec = torch.cat(outs, 0)
    torch.cuda.synchronize()
    for m in model.modules:
        h = getattr(getattr(m, "mlp", None), "cpu_host", None)
        if h is not None:
            print(f"cpu_host: abort {int(h.v_abort[0])} jobs head {int(h.v_jobs_head[0])} tail {int(h.v_jobs_tail[0])} seq {h.seq}")
            break
    r = ref[P - 1:N - 1] if False else ref[P:N]
    # decode forward of token t yields logits for position t (predicting t+1); ref row t likewise
    agree = (dec.argmax(-1) == r.argmax(-1)).float().mean().item()
    lpr = torch.log_softmax(r, -1); lpd = torch.log_softmax(dec, -1)
    kl = (lpr.exp() * (lpr - lpd)).sum(-1)
    tgt = ids[0, P + 1:N]
    nll_r = -lpr[:-1].gather(1, tgt.unsqueeze(1)).mean().item()
    nll_d = -lpd[:-1].gather(1, tgt.unsqueeze(1)).mean().item()
    print(f"chunk {args.chunk}: top1 agree {agree:.3f}, KL mean {kl.mean():.4f} max {kl.max():.4f}, "
          f"ppl ref {math.exp(nll_r):.3f} dec {math.exp(nll_d):.3f}")
    for (tg, name, k), a in sorted(sub_rec.items()):
        if tg != "ref" or ("dec", name, k) not in sub_rec: continue
        b = sub_rec[("dec", name, k)]
        a2 = _pick(a, N, P); b2 = _pick(b, 1, 0)
        if name == "mlp.routing_fn":
            print(f"  routing[{k}] ref {a2.tolist()} dec {b2.tolist()}"); continue
        if a2.numel() != b2.numel():
            print(f"  {name}[{k}] shapes ref {tuple(a.shape)} dec {tuple(b.shape)}"); continue
        print(f"  {name}[{k}] ref {tuple(a.shape)} rel err {((a2 - b2).norm() / (a2.norm() + 1e-9)).item():.4f}")
    k_out, k_in = "mlp.cpu_split_combine", "mlp.cpu_split_combine.in_gpu"
    if all((t, n, 0) in sub_rec for t in ("ref", "dec") for n in (k_out, k_in)):
        cr = _pick(sub_rec[("ref", k_out, 0)], N, P) - _pick(sub_rec[("ref", k_in, 0)], N, P)
        cd = _pick(sub_rec[("dec", k_out, 0)], 1, 0) - _pick(sub_rec[("dec", k_in, 0)], 1, 0)
        print(f"  derived CPU partial: rel err {((cr - cd).norm() / (cr.norm() + 1e-9)).item():.4f} (ref norm {cr.norm():.3f}, dec norm {cd.norm():.3f})")
    if args.modules:
        for i in range(len(model.modules)):
            if ("ref", i) not in rec or ("dec", i) not in rec: continue
            name, key, a = rec[("ref", i)]
            _, _, b = rec[("dec", i)]
            a = a.reshape(a.shape[0] * a.shape[1], -1)[P:P + args.chunk].flatten() if a.dim() >= 2 else a
            b = b.flatten()
            if a.numel() != b.numel():
                print(f"{i:3d} {name:22s} {key:50s} shape mismatch {a.numel()} vs {b.numel()}"); continue
            err = ((a - b).norm() / (a.norm() + 1e-9)).item()
            print(f"{i:3d} {name:22s} {key:50s} rel err {err:.4f}")

def _pick(t, N, P):
    while t.dim() > 1 and t.shape[0] == 1: t = t[0]
    if t.shape[0] == N: t = t[P]
    return t.flatten()

if __name__ == "__main__":
    main()
