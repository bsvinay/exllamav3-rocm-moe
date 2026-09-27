# Paired decode-speed A/B for the MTP drafting options: for each seed, sample the same prompt with every
# combination of Gumbel coupling (EXL3_MTP_COUPLED) and prompt lookup (EXL3_MTP_LOOKUP) in one process, so the
# configurations share load, placement and prompt cache; reports tokens/s and draft acceptance per run and averaged
import os, sys, time, argparse, torch
os.environ.setdefault("EXL3_MTP_LOOKUP", "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import generator as genmod
from exllamav3.generator.sampler import ComboSampler
from gen_check import PROMPT, EDIT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 396)
    ap.add_argument("--len", type = int, default = 16000)
    ap.add_argument("--tokens", type = int, default = 1500)
    ap.add_argument("--seeds", default = "1,2,3")
    ap.add_argument("--temp", type = float, default = 1.0)
    ap.add_argument("--task", choices = ["aurora", "edit"], default = "aurora")
    ap.add_argument("--configs", default = "00,10,01,11", help = "coupled,lookup bits per config")
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    draft_model = Model.from_config(config, component = "mtp")
    q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    cache = Cache(model, max_num_tokens = (args.len + args.tokens + 1024 + 255) // 256 * 256, max_history = int(os.environ.get("EXL3_MTP_LOOKUP_MAX", 5)),
                  max_batch_size = 1, **q8)
    model.load(progressbar = False)
    draft_cache = Cache(draft_model, max_num_tokens = cache.max_num_tokens, **q8)
    draft_model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft_model,
                    draft_cache = draft_cache, num_draft_tokens = 2)
    lookup_min = gen.mtp_lookup_min
    parts = [tok.encode("<|im_start|>user\nRead this:\n", encode_special_tokens = True)]
    if args.task == "edit":
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3", "generator", "job.py")
        parts.append(tok.encode("```python\n" + open(src).read() + "\n```")[:, :args.len])
        parts.append(tok.encode("\n\n" + EDIT + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", encode_special_tokens = True))
    else:
        parts.append(tok.encode(open(os.path.expanduser("~/wikitext2_test.txt")).read())[:, :args.len])
        parts.append(tok.encode("\n\n" + PROMPT + "<|im_end|>\n<|im_start|>assistant\n<think>\n", encode_special_tokens = True))
    ids = torch.cat(parts, 1)

    def run(seed, coupled, lookup):
        genmod._mtp_coupled = coupled
        gen.mtp_lookup_min = lookup_min if lookup else 0
        gen.lookup_stats = [0, 0, 0]
        gen.lookup_thresh = lookup_min
        sampler = ComboSampler(temperature = args.temp, top_k = 20, top_p = 0.95) if args.temp else \
            ComboSampler(temperature = 0.0, top_k = 1)
        job = Job(input_ids = ids, max_new_tokens = args.tokens, stop_conditions = [], seed = seed, sampler = sampler)
        gen.enqueue(job)
        first = None; n = 0; acc = rej = 0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                t = r.get("token_ids")
                if t is not None and t.numel():
                    if first is None: first = time.perf_counter()
                    n += t.numel()
                if r.get("eos"):
                    acc, rej = r.get("accepted_draft_tokens", 0), r.get("rejected_draft_tokens", 0)
        dt = time.perf_counter() - first
        return n / dt, acc / max(acc + rej, 1), (acc + rej) and n / max(n - acc, 1), list(gen.lookup_stats)

    run(0, False, False)   # warm: prompt cache, placement
    configs = [(c[0] == "1", c[1] == "1") for c in args.configs.split(",")]
    res = {c: [] for c in configs}
    for seed in [int(s) for s in args.seeds.split(",")]:
        for c in configs:
            tps, acc, tpr, ls = run(seed, *c)
            res[c].append((tps, acc, tpr))
            print(f"seed {seed} coupled {int(c[0])} lookup {int(c[1])}: {tps:6.1f} T/s  accept {acc:.3f}  "
                  f"tokens/round {tpr:.2f}  lookup {ls}", flush = True)
    print("mean:")
    for c, v in res.items():
        print(f"  coupled {int(c[0])} lookup {int(c[1])}: {sum(x[0] for x in v) / len(v):6.1f} T/s  "
              f"accept {sum(x[1] for x in v) / len(v):.3f}  tokens/round {sum(x[2] for x in v) / len(v):.2f}")

if __name__ == "__main__":
    main()
