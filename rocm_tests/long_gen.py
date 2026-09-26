# Long-context prefill and decode speed: wikitext prompt of N tokens + a question, 128 new tokens
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 346)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--cache", type = int, default = 131072)
    ap.add_argument("--lens", default = "8000,32000,100000")
    ap.add_argument("--chunk", type = int, default = 2048)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache, max_batch_size = 1, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    model.load(progressbar = False, max_chunk_size = args.chunk)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_chunk_size = args.chunk)
    text = open(os.path.expanduser("~/wikitext2_test.txt")).read()
    all_ids = tok.encode(text)
    q = tok.encode("<|im_end|>\n<|im_start|>user\nSummarize the article above in three sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", encode_special_tokens = True)
    head = tok.encode("<|im_start|>user\nRead this:\n", encode_special_tokens = True)
    for L in [int(x) for x in args.lens.split(",")]:
        body = all_ids[:, :L]
        while body.shape[1] < L: body = torch.cat([body, all_ids[:, :L - body.shape[1]]], 1)
        ids = torch.cat([head, body, q], 1)
        job = Job(input_ids = ids, max_new_tokens = 128, sampler = ComboSampler(temperature = 0.6, top_p = 0.95, top_k = 20))
        gen.enqueue(job)
        t0 = time.perf_counter(); first = None; n = 0; out = []
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("text"):
                    if first is None: first = time.perf_counter()
                    n += 1; out.append(r["text"])
        t1 = time.perf_counter()
        print(f"ctx {ids.shape[1]}: prefill {ids.shape[1] / (first - t0):.0f} tok/s (TTFT {first - t0:.1f} s), "
              f"decode {(n - 1) / (t1 - first):.1f} tok/s", flush = True)
        print("   ", "".join(out)[:160].replace("\n", " "), flush = True)
        cache.reset() if hasattr(cache, "reset") else None

if __name__ == "__main__":
    main()
