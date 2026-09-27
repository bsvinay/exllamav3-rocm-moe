# Distribution check for speculative sampling: sample many short continuations of one high-entropy prompt with MTP
# drafting, Gumbel-coupled (EXL3_MTP_COUPLED) and uncoupled, and compare the per-position token distributions by total
# variation distance. An exact scheme is as close to the uncoupled reference as two independent uncoupled runs are
import os, sys, argparse, collections, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import generator as genmod
from exllamav3.generator.sampler import ComboSampler

PROMPT = ("<|im_start|>user\nName one animal, then one color, then one city. Answer with three words only, "
          "no punctuation.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

def tv(a, b, n):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / n - b[k] / n) for k in keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 396)
    ap.add_argument("--n", type = int, default = 300)
    ap.add_argument("--tokens", type = int, default = 6)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    draft_model = Model.from_config(config, component = "mtp")
    q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    cache = Cache(model, max_num_tokens = 4096, max_history = 4, max_batch_size = 1, **q8)
    model.load(progressbar = False)
    draft_cache = Cache(draft_model, max_num_tokens = 4096, **q8)
    draft_model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft_model,
                    draft_cache = draft_cache, num_draft_tokens = 2)
    ids = tok.encode(PROMPT, encode_special_tokens = True)

    def run(coupled, seed0):
        genmod._mtp_coupled = coupled
        pos = [collections.Counter() for _ in range(args.tokens)]
        acc = tot = 0
        for s in range(args.n):
            job = Job(input_ids = ids, max_new_tokens = args.tokens, stop_conditions = [], seed = seed0 + s,
                      sampler = ComboSampler(temperature = 1.0, top_k = 20, top_p = 0.95))
            gen.enqueue(job)
            out = []
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    t = r.get("token_ids")
                    if t is not None and t.numel(): out += t.view(-1).tolist()
                    if r.get("eos"):
                        acc += r.get("accepted_draft_tokens", 0); tot += r.get("accepted_draft_tokens", 0) + r.get("rejected_draft_tokens", 0)
            for i, t in enumerate(out[:args.tokens]):
                # position i conditioned on nothing: the joint prefix up to i, so later positions compare the paths
                pos[i][tuple(out[:i + 1])] += 1
        return pos, acc / max(tot, 1)

    ref_a, acc_a = run(False, 1000)
    ref_b, acc_b = run(False, 50000)
    cpl, acc_c = run(True, 90000)
    print(f"draft acceptance: uncoupled {acc_a:.3f} / {acc_b:.3f}, coupled {acc_c:.3f}")
    print("pos  TV(unc,unc')  TV(unc,coupled)  TV(unc',coupled)")
    for i in range(args.tokens):
        print(f"{i:3d}  {tv(ref_a[i], ref_b[i], args.n):12.3f}  {tv(ref_a[i], cpl[i], args.n):15.3f}  {tv(ref_b[i], cpl[i], args.n):16.3f}")
    top = ref_a[0].most_common(5)
    print("top first tokens:", [(tok.decode(torch.tensor([k[-1]]).view(1, 1))[0], c, ref_b[0][k], cpl[0][k]) for k, c in top])

if __name__ == "__main__":
    main()
