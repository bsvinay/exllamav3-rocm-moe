# Router selection histogram during generation (code + prose), then the CPU share of expert activations
# as a function of GPU-resident slots per layer with ideal (frequency-ranked) placement
import os, sys, torch, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler

def main():
    M = sys.argv[1]
    config = Config.from_directory(M)
    config.infer_params.moe_cpu_split = int(sys.argv[2]) if len(sys.argv) > 2 else 352
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 16384, max_batch_size = 1)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    hist = collections.defaultdict(lambda: torch.zeros(512, dtype = torch.long))
    cpu_hits = [0, 0]
    for m in model.modules:
        if type(m).__name__ != "TransformerBlock": continue
        mm = m.mlp
        f = mm.routing_fn
        def w(bsz, cfg, y, params, f = f, key = mm.key):
            sel, wt = f(bsz, cfg, y, params)
            if bsz == 1: hist[key] += torch.bincount(sel.view(-1).cpu(), minlength = 512)
            return sel, wt
        mm.routing_fn = w
        g = mm.cpu_split_submit
        def w2(y, bsz, sel, rw, g = g, mm = mm):
            r = g(y, bsz, sel, rw)
            if bsz == 1:
                s = sel.view(-1).cpu()
                cpu_hits[0] += int((s >= mm.cpu_split_first).sum()); cpu_hits[1] += s.numel()
            return r
        mm.cpu_split_submit = w2
    gen = Generator(model = model, cache = cache, tokenizer = tok)
    prompts = ["Write a complete Python module implementing an LRU cache with TTL expiry, thread safety and pytest tests.",
               "Write a vivid 600-word short story about a lighthouse keeper.",
               "Explain step by step how TCP congestion control works, with examples.",
               "Implement a red-black tree in C++ with insert, delete and an iterator."]
    for p in prompts:
        text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        gen.generate(prompt = text, max_new_tokens = 400, sampler = ComboSampler(temperature = 0.6, top_p = 0.95, top_k = 20),
                     encode_special_tokens = True, completion_only = True)
    print(f"live CPU share of decode activations: {cpu_hits[0] / max(cpu_hits[1], 1):.3f} ({cpu_hits[1]} picks)")
    tot = sum(h.sum().item() for h in hist.values())
    for slots in (128, 160, 192, 224, 256, 320):
        on_gpu = sum(torch.sort(h, descending = True).values[:slots].sum().item() for h in hist.values())
        print(f"ideal placement, {slots} GPU slots/layer: CPU share {1 - on_gpu / tot:.3f}")
    torch.save({k: v for k, v in hist.items()}, os.path.expanduser("~/expert_hist.pt"))

if __name__ == "__main__":
    main()
