# Quick perplexity on wikitext-2 (prefill, no cache) for a model with CPU expert offload
import os, sys, math, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 0)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--rows", type = int, default = 4)
    ap.add_argument("--len", type = int, default = 1024)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    if args.mcs: config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    text = open(os.path.expanduser("~/wikitext2_test.txt")).read()
    ids = tok.encode(text[:400000])
    nll = cnt = 0.0
    for r in range(args.rows):
        x = ids[:, r * args.len:(r + 1) * args.len]
        logits = model.forward(x, {"attn_mode": "flash_attn_nc"}).float().cpu()
        lp = torch.log_softmax(logits[0, :-1], dim = -1)
        tgt = x[0, 1:]
        n = -lp.gather(1, tgt.unsqueeze(1)).sum().item()
        nll += n; cnt += tgt.numel()
        print(f"row {r}: ppl {math.exp(n / tgt.numel()):.3f}", flush = True)
    print(f"PPL {math.exp(nll / cnt):.4f}")

if __name__ == "__main__":
    main()
