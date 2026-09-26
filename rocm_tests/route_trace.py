# Record per-token router selections (raw checkpoint expert ids) during long generations, for offline
# placement simulations. Saves {layer: LongTensor (tokens, topk)} per prompt.
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator.sampler import ComboSampler

def main():
    M = sys.argv[1]
    os.environ["EXL3_MOE_CPU_SWAP"] = "1"   # checkpoint order: traces index raw expert ids
    config = Config.from_directory(M)
    config.infer_params.moe_cpu_split = 346
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 32768, max_batch_size = 1)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    trace = {}
    cur = {}
    for m in model.modules:
        mm = getattr(m, "mlp", None)
        if mm is None or getattr(mm, "routing_fn", None) is None: continue
        f = mm.routing_fn
        def w(bsz, cfg, y, params, f = f, key = mm.key):
            sel, wt = f(bsz, cfg, y, params)
            if bsz == 1: cur.setdefault(key, []).append(sel.view(-1).cpu().clone())
            return sel, wt
        mm.routing_fn = w
    gen = Generator(model = model, cache = cache, tokenizer = tok)
    prompts = {
        "code_a": "Implement a complete Python HTTP/1.1 server from scratch using only the socket module: request parsing, keep-alive, chunked transfer encoding, static file serving with MIME types, and a small routing decorator. Include tests.",
        "code_b": "Write a Rust command-line tool that indexes a directory of Markdown files into an inverted index, supports boolean queries with AND/OR/NOT, and prints ranked results with BM25 scores. Include the full Cargo project.",
        "chat": "I'm planning a two-week trip to Japan in autumn with a moderate budget. Suggest an itinerary with cities, transport between them, where to stay, and what to eat, with rough costs.",
    }
    for name, p in prompts.items():
        cur.clear()
        text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        gen.generate(prompt = text, max_new_tokens = 2000, sampler = ComboSampler(temperature = 0.6, top_p = 0.95, top_k = 20),
                     encode_special_tokens = True, completion_only = True)
        trace[name] = {k: torch.stack(v) for k, v in cur.items()}
        print(name, len(next(iter(cur.values()))), "tokens", flush = True)
    torch.save(trace, os.path.expanduser("~/route_trace.pt"))

if __name__ == "__main__":
    main()
