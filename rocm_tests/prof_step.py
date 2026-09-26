# Per-module GPU-stream time of cached single-token decode steps (CUDA events around every hooked call;
# host-blocking waits inside a call show up as that call's time)
import os, sys, time, argparse, collections, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 0)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--steps", type = int, default = 24)
    ap.add_argument("--nohooks", action = "store_true")
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    if args.mcs: config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    ids = tok.encode(open(os.path.expanduser("~/wikitext2_test.txt")).read()[5000:60000])[:, :200 + args.steps]

    ev = []
    on = [False]
    def wrap(obj, mn, name):
        f = getattr(obj, mn)
        def w(*a, **kw):
            if not on[0]: return f(*a, **kw)
            e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
            t0 = time.perf_counter(); e0.record()
            y = f(*a, **kw)
            e1.record(); ev.append((name, e0, e1, time.perf_counter() - t0))
            return y
        setattr(obj, mn, w)
    if not args.nohooks:
        for m in model.modules:
            cls = type(m).__name__
            if cls == "TransformerBlock":
                kind = type(m.attn).__name__
                for an, mn in (("attn_hc", "mix"), ("attn_hc", "apply_"), ("mlp_hc", "mix"), ("mlp_hc", "apply_")):
                    wrap(getattr(m, an), mn, f"{an}.{mn}")
                wrap(m.attn, "forward", f"attn[{kind}]")
                mm = m.mlp
                wrap(mm, "routing_fn", "mlp.routing")
                wrap(mm, "cpu_split_submit", "mlp.cpu_submit")
                wrap(mm, "_rdna3_moe_forward", "mlp.gpu_experts")
                wrap(mm, "cpu_split_combine", "mlp.cpu_collect")
                if mm.shared_experts is not None:
                    wrap(mm.shared_experts, "forward", "mlp.shared")
                wrap(mm, "forward", "mlp(total)")
            else:
                wrap(m, "forward", cls)

    P = 200
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
    model.prefill(input_ids = ids[:, :P], params = params)
    rs = params.get("recurrent_states")
    torch.cuda.synchronize()
    times = []; enq = []
    for s in range(args.steps):
        pos = P + s
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 4096), "recurrent_states": rs}
        on[0] = s >= 4
        t0 = time.perf_counter()
        model.forward(input_ids = ids[:, pos:pos + 1], params = p)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        if s >= 4: times.append(time.perf_counter() - t0); enq.append(t1 - t0)
    n = len(times)
    print(f"step: median {sorted(times)[n // 2] * 1000:.2f} ms ({1 / sorted(times)[n // 2]:.1f} tok/s), "
          f"host enqueue median {sorted(enq)[n // 2] * 1000:.2f} ms")
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for name, e0, e1, host in ev:
        a = agg[name]; a[0] += e0.elapsed_time(e1); a[1] += host * 1000; a[2] += 1
    for name, (g, h, c) in sorted(agg.items(), key = lambda kv: -kv[1][0]):
        print(f"  {name:22s} calls/step {c / n:5.1f}  gpu {g / n:7.3f} ms/step  host {h / n:7.3f} ms/step")

if __name__ == "__main__":
    main()
