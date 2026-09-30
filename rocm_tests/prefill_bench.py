# Prefill throughput through the Generator (the served path: chunking, recurrent checkpoints, MTP layer prefill):
# a fresh wikitext prompt of --len tokens per run (different offsets, so no prefix-cache reuse), one generated
# token; reports prompt tokens/s per run and the n-gram prefetch hit/miss counts
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 360)
    ap.add_argument("--cache", type = int, default = 65536)
    ap.add_argument("--chunk", type = int, default = 2048)
    ap.add_argument("--len", type = int, default = 16384)
    ap.add_argument("--runs", type = int, default = 3)
    ap.add_argument("--mtp", action = "store_true")
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp") if args.mtp else None
    q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    cache = Cache(model, max_num_tokens = args.cache, max_history = 4 if draft else 0, max_batch_size = 1, **q8)
    model.load(progressbar = False, max_chunk_size = args.chunk)
    dcache = None
    if draft is not None:
        dcache = Cache(draft, max_num_tokens = args.cache, **q8)
        draft.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft, draft_cache = dcache,
                    num_draft_tokens = 2 if draft else None, max_chunk_size = args.chunk)
    text = tok.encode(open(os.path.expanduser("~/wikitext2_test.txt")).read())
    ple = [m for m in model.modules if type(m).__name__ == "PLELayer"]
    for r in range(args.runs):
        off = 1000 + r * (args.len + 512)
        ids = text[:, off:off + args.len]
        job = Job(input_ids = ids, max_new_tokens = 1, sampler = ComboSampler(temperature = 0.0, top_k = 1), stop_conditions = [])
        gen.enqueue(job)
        t0 = time.perf_counter(); first = None
        while gen.num_remaining_jobs():
            for res in gen.iterate():
                if first is None and res.get("token_ids") is not None and res["token_ids"].numel():
                    first = time.perf_counter()
        dt = (first or time.perf_counter()) - t0
        stats = ple[0].ple_embedding.prefetch_stats if ple else {}
        print(f"run {r}: {args.len} tokens in {dt:.2f} s -> {args.len / dt:.0f} tok/s  ngram prefetch {dict(stats)}", flush = True)

if __name__ == "__main__":
    main()
