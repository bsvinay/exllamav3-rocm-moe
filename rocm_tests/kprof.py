# Kernel-level breakdown of cached decode steps (torch.profiler; long kernels may be under-reported on ROCm)
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 346)
    ap.add_argument("--steps", type = int, default = 10)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    time.sleep(90)
    ids = tok.encode(open(os.path.expanduser("~/wikitext2_test.txt")).read()[5000:60000])[:, :300]
    P = 200
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
    model.prefill(input_ids = ids[:, :P], params = params)
    rs = params.get("recurrent_states")
    def step(pos):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 4096), "recurrent_states": rs}
        model.forward(input_ids = ids[:, pos:pos + 1], params = p)
    for i in range(5): step(P + i)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(args.steps): step(P + 5 + i)
        torch.cuda.synchronize()
    evs = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA or getattr(e, "self_device_time_total", 0) > 0]
    rows = []
    for e in prof.key_averages():
        dt = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if dt > 0: rows.append((dt, e.count, e.key))
    rows.sort(reverse = True)
    tot = sum(r[0] for r in rows)
    ncnt = sum(r[1] for r in rows)
    print(f"device time {tot / args.steps / 1000:.2f} ms/step, {ncnt / args.steps:.0f} kernels/step")
    for dt, c, k in rows[:40]:
        print(f"  {dt / args.steps / 1000:7.3f} ms/step  {c / args.steps:6.1f}/step  {k[:110]}")

if __name__ == "__main__":
    main()
