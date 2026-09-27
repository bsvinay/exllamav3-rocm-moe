# Long decode across a context boundary: wikitext prompt of L tokens, then the model is asked to
# repeat the article from the start; snippets are printed every --every tokens with the absolute
# context position, so output that turns to garbage past some length is easy to spot
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 396)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--cache", type = int, default = 196608)
    ap.add_argument("--len", type = int, default = 31000)
    ap.add_argument("--tokens", type = int, default = 3000)
    ap.add_argument("--every", type = int, default = 256)
    ap.add_argument("--mtp", action = "store_true")
    ap.add_argument("--temp", type = float, default = 0.0)
    ap.add_argument("--pre", type = int, default = 0, help = "first run a job with this many prompt tokens (leaves longer state behind)")
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    draft_model = draft_cache = None
    if args.mtp:
        draft_model = Model.from_config(config, component = "mtp")
    max_history = draft_model.caps.get("default_draft_size", 4) if draft_model else 0
    q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    cache = Cache(model, max_num_tokens = args.cache, max_history = max_history, max_batch_size = 1, **q8)
    model.load(progressbar = False)
    if draft_model is not None:
        draft_cache = Cache(draft_model, max_num_tokens = args.cache, **q8)
        draft_model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft_model,
                    draft_cache = draft_cache, num_draft_tokens = 2 if args.mtp else None)
    text = open(os.path.expanduser("~/wikitext2_test.txt")).read()
    body = tok.encode(text)[:, :args.len]
    head = tok.encode("<|im_start|>user\nRead this:\n", encode_special_tokens = True)
    q = tok.encode("<|im_end|>\n<|im_start|>user\nRepeat the text above word for word, from the very "
                   "beginning, as far as you can.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                   encode_special_tokens = True)
    ids = torch.cat([head, body, q], 1)
    if args.pre:
        pre_ids = torch.cat([head, tok.encode(text[len(text) // 2:])[:, :args.pre], q], 1)
        gen.enqueue(Job(input_ids = pre_ids, max_new_tokens = 32, sampler = ComboSampler(temperature = 0.0, top_k = 1)))
        pre_out = []
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("text"): pre_out.append(r["text"])
        print(f"pre job {pre_ids.shape[1]} tok:", "".join(pre_out)[:120].replace("\n", " "), flush = True)
    L0 = ids.shape[1]
    sampler = ComboSampler(temperature = args.temp, top_k = 1) if args.temp == 0 else \
        ComboSampler(temperature = args.temp, top_p = 0.95, top_k = 20)
    job = Job(input_ids = ids, max_new_tokens = args.tokens, sampler = sampler, stop_conditions = [])
    gen.enqueue(job)
    t0 = time.perf_counter(); first = None; n = 0; buf = []; mark = args.every
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            t = r.get("token_ids")
            if t is not None and t.numel():
                if first is None:
                    first = time.perf_counter()
                    print(f"prompt {L0} tok, TTFT {first - t0:.1f} s", flush = True)
                n += t.numel()
            if r.get("text"): buf.append(r["text"])
            if n >= mark or r.get("eos"):
                s = "".join(buf)[-160:].replace("\n", " ")
                dt = time.perf_counter() - first
                print(f"[ctx {L0 + n:6d} gen {n:5d} {n / max(dt, 1e-3):5.1f} T/s] ...{s}", flush = True)
                mark += args.every
            if r.get("eos"):
                print("eos:", r.get("eos_reason"), "accept", r.get("accepted_draft_tokens"), "/", r.get("rejected_draft_tokens"), flush = True)

if __name__ == "__main__":
    main()
