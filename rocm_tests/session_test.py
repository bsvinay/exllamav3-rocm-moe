# A session of sequential requests: per request, the CPU share of decode expert picks and decode speed
# (expert placement sweeps run between requests when dynamic placement is on)
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler

PROMPTS = [
    ("code", "Write a Python asyncio web crawler with a rate limiter, robots.txt handling, retries with backoff and a SQLite result store."),
    ("code", "Implement a thread-safe LRU cache in Go with generics, TTL expiry and benchmarks."),
    ("code", "Write a TypeScript React hook for paginated infinite scrolling with request cancellation and caching, plus tests."),
    ("code", "Implement Dijkstra and A* in C++ on a grid with obstacles, with a small CLI and unit tests."),
    ("chat", "What are good strategies to improve sleep quality for someone who works night shifts?"),
    ("code", "Write a Python CLI that diffs two JSON files structurally and prints a colored tree of changes, with tests."),
]

def main():
    M = sys.argv[1]; mcs = int(sys.argv[2]) if len(sys.argv) > 2 else 346
    config = Config.from_directory(M)
    config.infer_params.moe_cpu_split = mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 32768, max_batch_size = 1)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    hits = [0, 0]
    for m in model.modules:
        mm = getattr(m, "mlp", None)
        if mm is None or getattr(mm, "cpu_split_submit", None) is None or mm.cpu_split_first is None: continue
        g = mm.cpu_split_submit
        def w(y, bsz, sel, rw, g = g, mm = mm):
            r = g(y, bsz, sel, rw)
            if bsz == 1:
                s = sel.view(-1)
                hits[0] += int((s >= mm.cpu_split_first).sum()); hits[1] += s.numel()
            return r
        mm.cpu_split_submit = w
    gen = Generator(model = model, cache = cache, tokenizer = tok)
    for kind, p in PROMPTS:
        hits[0] = hits[1] = 0
        text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids = tok.encode(text, encode_special_tokens = True)
        gen.enqueue(Job(input_ids = ids, max_new_tokens = 500, sampler = ComboSampler(temperature = 0.6, top_p = 0.95, top_k = 20)))
        t0 = time.perf_counter(); first = None; last = None; n = 0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("text"):
                    if first is None: first = time.perf_counter()
                    last = time.perf_counter()
                    n += 1
        t1 = last
        print(f"[{kind}] CPU share {hits[0] / max(hits[1], 1):.3f}, decode {(n - 1) / (t1 - first):.1f} tok/s, {n} tok", flush = True)

if __name__ == "__main__":
    main()
