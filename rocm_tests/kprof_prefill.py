# Kernel-level profile of one prefill chunk (torch.profiler): top GPU kernels by total time, with call counts
import os, sys, time, argparse, collections, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 380)
    ap.add_argument("--chunk", type = int, default = 8192)
    ap.add_argument("--past", type = int, default = 8192)
    ap.add_argument("--top", type = int, default = 30)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 32768, max_batch_size = 1, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    model.load(progressbar = False, max_chunk_size = args.chunk)
    tok = Tokenizer.from_config(config)
    ids = tok.encode(open(os.path.expanduser("~/wikitext2_test.txt")).read())[:, :args.past + 2 * args.chunk]
    pos = 0; rs = None
    while pos < args.past:
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 32768)}
        if rs is not None: p["recurrent_states"] = rs
        model.forward(input_ids = ids[:, pos:pos + args.chunk], params = p)
        rs = p.get("recurrent_states"); pos += args.chunk
    torch.cuda.synchronize()
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 32768), "recurrent_states": rs}
    from torch.profiler import profile, ProfilerActivity
    t0 = time.perf_counter()
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes = True) as prof:
        model.forward(input_ids = ids[:, pos:pos + args.chunk], params = p)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    agg = collections.defaultdict(lambda: [0.0, 0])
    total = 0.0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            d = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            agg[e.name[:90]][0] += d; agg[e.name[:90]][1] += 1; total += d
    print(f"chunk {args.chunk}: wall {wall * 1000:.0f} ms (profiled), GPU kernel time total {total / 1000:.0f} ms")
    for name, (t, c) in sorted(agg.items(), key = lambda kv: -kv[1][0])[:args.top]:
        print(f"  {t / 1000:8.1f} ms  {100 * t / total:5.1f}%  x{c:6d}  {name}")
    print("aten ops by device time (with input shapes):")
    rows = []
    for e in prof.key_averages(group_by_input_shape = True):
        d = getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0)
        if e.key.startswith("aten::") and d > 20000:
            rows.append((d, e.count, e.key, str(e.input_shapes)[:110]))
    for d, c, k, sh in sorted(rows, reverse = True)[:25]:
        print(f"  {d / 1000:8.1f} ms  x{c:6d}  {k:28s} {sh}")

if __name__ == "__main__":
    main()
