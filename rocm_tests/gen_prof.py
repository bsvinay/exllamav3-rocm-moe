# Where a generator decode step goes: model.forward (host wall, synchronized) vs everything else
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 0)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--temp", type = float, default = 0.6)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    if args.mcs: config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 32768, max_batch_size = 1)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok)
    fw = [0.0, 0]
    orig = model.forward
    def f(*a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig(*a, **kw)
        torch.cuda.synchronize(); fw[0] += time.perf_counter() - t0; fw[1] += 1
        return r
    model.forward = f
    prompt = "<|im_start|>user\nWrite a Python quicksort with tests.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    import cProfile, pstats
    pr = cProfile.Profile()
    for rep in range(2):
        if rep == 1: pr.enable()
        fw[0] = 0.0; fw[1] = 0
        ids = tok.encode(prompt, encode_special_tokens = True)
        job = Job(input_ids = ids, max_new_tokens = 200, sampler = ComboSampler(temperature = args.temp, top_p = 0.95, top_k = 20))
        gen.enqueue(job)
        n = 0; t0 = None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("text") and t0 is None: t0 = time.perf_counter(); fw[0] = 0.0; fw[1] = 0
                if r.get("text"): n += 1
        t = time.perf_counter() - t0
        if rep == 1: pr.disable()
        print(f"rep {rep}: {n} results in {t * 1000:.0f} ms -> {t / max(n, 1) * 1000:.2f} ms/tok; forward calls {fw[1]} "
              f"{fw[0] / max(fw[1], 1) * 1000:.2f} ms each; non-forward {(t - fw[0]) / max(n, 1) * 1000:.2f} ms/tok")

    st = pstats.Stats(pr); st.sort_stats("tottime").print_stats(8); st.print_callers("stloader_read|posix.read|_cuda_synchronize")

if __name__ == "__main__":
    main()
