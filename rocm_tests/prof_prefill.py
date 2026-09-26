# Per-module time of one cached prefill chunk (CUDA events + host wall per hooked call)
import os, sys, time, argparse, collections, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 346)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--chunk", type = int, default = 2048)
    ap.add_argument("--past", type = int, default = 8192)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 32768, max_batch_size = 1, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    model.load(progressbar = False, max_chunk_size = args.chunk)
    tok = Tokenizer.from_config(config)
    ids = tok.encode(open(os.path.expanduser("~/wikitext2_test.txt")).read())[:, :args.past + 2 * args.chunk]

    ev = []; on = [False]
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
    for m in model.modules:
        cls = type(m).__name__
        if cls == "TransformerBlock":
            kind = type(m.attn).__name__
            wrap(m.attn_hc, "mix", "hc.mix"); wrap(m.mlp_hc, "mix", "hc.mix")
            wrap(m.attn_hc, "apply_", "hc.apply"); wrap(m.mlp_hc, "apply_", "hc.apply")
            wrap(m.attn, "forward", f"attn[{kind}]")
            mm = m.mlp
            wrap(mm, "routing_fn", "mlp.routing")
            wrap(mm, "cpu_split_submit", "mlp.cpu_submit(prefill)")
            wrap(mm, "cpu_split_combine", "mlp.cpu_combine")
            if mm.shared_experts is not None: wrap(mm.shared_experts, "forward", "mlp.shared")
            wrap(mm, "forward", "mlp(total)")
        else:
            wrap(m, "forward", cls)

    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 32768)}
    pos = 0
    rs = None
    for c in range(0, args.past, args.chunk):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 32768)}
        if rs is not None: p["recurrent_states"] = rs
        model.forward(input_ids = ids[:, pos:pos + args.chunk], params = p, last_tokens_only = 1) \
            if "last_tokens_only" in model.forward.__code__.co_varnames else model.forward(input_ids = ids[:, pos:pos + args.chunk], params = p)
        rs = p.get("recurrent_states"); pos += args.chunk
    torch.cuda.synchronize()
    on[0] = True
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 32768), "recurrent_states": rs}
    t0 = time.perf_counter()
    model.forward(input_ids = ids[:, pos:pos + args.chunk], params = p)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    on[0] = False
    print(f"chunk {args.chunk} at past {pos}: {t * 1000:.0f} ms -> {args.chunk / t:.0f} tok/s")
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for name, e0, e1, host in ev:
        a = agg[name]; a[0] += e0.elapsed_time(e1); a[1] += host * 1000; a[2] += 1
    for name, (g, h, c) in sorted(agg.items(), key = lambda kv: -kv[1][0]):
        print(f"  {name:26s} calls {c:4d}  gpu {g:8.1f} ms  host {h:8.1f} ms")

if __name__ == "__main__":
    main()
