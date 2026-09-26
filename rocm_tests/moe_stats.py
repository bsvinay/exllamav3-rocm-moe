"""
Per-layer expert selection counts for frequency-guided CPU expert placement.

Teacher-forces token sequences through the model (prefill, no cache) and counts every router
selection per MoE layer, then writes {layer key: [count per expert]} as JSON. With the file in the
model directory as expert_stats.json (or pointed to by EXL3_MOE_CPU_SPLIT_STATS), a --moe_cpu_split
load keeps each layer's most-selected experts on the GPU and the tail on the CPU worker.

Sources: the model's own qbench_prompts.json rows (prompt + sampled response, when present) and
any extra UTF-8 text files given with --text (chunked).

usage: moe_stats.py -m <model_dir> [--mcs 352] [--text file ...] [--out <model_dir>/expert_stats.json]
"""
import os, sys, json, argparse, collections, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 352)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--text", nargs = "*", default = [])
    ap.add_argument("--chunk", type = int, default = 2048)
    ap.add_argument("--max_text_tokens", type = int, default = 32768)
    ap.add_argument("--out", default = None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.model, "expert_stats.json")

    os.environ["EXL3_MOE_CPU_SWAP"] = "1"   # checkpoint expert order (no stats permutation)
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)

    counts = {}
    for m in model.modules:
        mm = getattr(m, "mlp", None)
        if mm is None or not hasattr(mm, "routing_fn") or mm.routing_fn is None:
            continue
        counts[mm.key] = torch.zeros(mm.num_experts, dtype = torch.long)
        f = mm.routing_fn
        def w(bsz, cfg, y, params, f = f, key = mm.key):
            sel, wt = f(bsz, cfg, y, params)
            counts[key] += torch.bincount(sel.reshape(-1).cpu(), minlength = counts[key].numel())
            return sel, wt
        mm.routing_fn = w

    seqs = []
    qb = os.path.join(args.model, "qbench_prompts.json")
    if os.path.exists(qb):
        for r in json.load(open(qb)).get("rows", []):
            seqs.append(torch.tensor([r["input_ids"] + r.get("response_ids", [])], dtype = torch.long))
    for path in args.text:
        ids = tok.encode(open(path, encoding = "utf-8").read())[:, :args.max_text_tokens]
        seqs += [ids[:, i:i + args.chunk] for i in range(0, ids.shape[1], args.chunk)]

    ntok = 0
    with torch.inference_mode():
        for s in seqs:
            for i in range(0, s.shape[1], args.chunk):
                x = s[:, i:i + args.chunk]
                if x.shape[1] < 2: continue
                model.forward(x, {"attn_mode": "flash_attn_nc"})
                ntok += x.shape[1]
    torch.cuda.synchronize()
    json.dump({k: v.tolist() for k, v in counts.items()}, open(out, "w"))
    tot = sum(v.sum().item() for v in counts.values())
    for slots in (128, 160, 192, 256):
        on = sum(torch.sort(v, descending = True).values[:slots].sum().item() for v in counts.values())
        print(f"{slots} GPU slots/layer: CPU share {1 - on / tot:.3f} (on the calibration tokens)")
    print(f"{ntok} tokens, {len(counts)} layers -> {out}")

if __name__ == "__main__":
    main()
