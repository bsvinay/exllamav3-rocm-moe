# Generation fidelity at long context: greedy-generate N tokens through the Generator (optionally with
# MTP drafting), then re-run prompt + generated tokens through a chunked prefill-path reference and
# report, per window of generated tokens, how often the reference argmax equals the generated token
# and the reference log-prob of the generated tokens. A correct decode path agrees ~all the time;
# state corruption shows up as agreement decaying with position
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

PROMPT = ("Now, unrelated to the text above. I want a single-file web page that renders a realistic, "
          "animated 3D aurora borealis over a snowy landscape with WebGL2 raymarching: layered curtains "
          "with folds, altitude-dependent color (green low, red/purple high), stars, and a performance "
          "budget of 60 fps at 720p. Think carefully about the rendering approach, the noise functions, "
          "the density model and the performance trade-offs, then write the complete code.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 396)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--cache", type = int, default = 40960)
    ap.add_argument("--len", type = int, default = 24000, help = "wikitext tokens before the question")
    ap.add_argument("--tokens", type = int, default = 4096)
    ap.add_argument("--window", type = int, default = 256)
    ap.add_argument("--mtp", action = "store_true")
    ap.add_argument("--ndt", type = int, default = 2)
    ap.add_argument("--rc", type = int, default = 0, help = "recurrent cache MB (Tabby sysmem_recurrent_cache)")
    ap.add_argument("--turns", type = int, default = 1)
    ap.add_argument("--keep", type = int, default = 1000, help = "turn-1 generated tokens carried into turn 2")
    ap.add_argument("--temp", type = float, default = 0.0)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    draft_model = draft_cache = None
    if args.mtp:
        draft_model = Model.from_config(config, component = "mtp")
    max_history = max(draft_model.caps.get("default_draft_size", 4), args.ndt) if draft_model else 0
    q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    cache = Cache(model, max_num_tokens = args.cache, max_history = max_history, max_batch_size = 1, **q8)
    model.load(progressbar = False)
    if draft_model is not None:
        draft_cache = Cache(draft_model, max_num_tokens = args.cache, **q8)
        draft_model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft_model,
                    draft_cache = draft_cache, num_draft_tokens = args.ndt if args.mtp else None,
                    recurrent_cache_size = args.rc * 1024**2)
    text = open(os.path.expanduser("~/wikitext2_test.txt")).read()
    parts = [tok.encode("<|im_start|>user\nRead this:\n", encode_special_tokens = True)]
    if args.len: parts.append(tok.encode(text)[:, :args.len])
    parts.append(tok.encode("\n\n" + PROMPT + "<|im_end|>\n<|im_start|>assistant\n<think>\n", encode_special_tokens = True))
    ids = torch.cat(parts, 1)
    P = ids.shape[1]

    def run(ids):
        return _run(gen, tok, ids, args)
    g = run(ids)
    for turn in range(1, args.turns):
        follow = tok.encode("<|im_end|>\n<|im_start|>user\nGood. Now continue: refine the density model and "
                            "write the final complete code.<|im_end|>\n<|im_start|>assistant\n<think>\n",
                            encode_special_tokens = True)
        ids = torch.cat([ids, g[:args.keep].view(1, -1), follow], 1)
        P = ids.shape[1]
        print(f"--- turn {turn + 1}", flush = True)
        g = run(ids)
    check(model, cache, ids, g, args)

def _run(gen, tok, ids, args):
    P = ids.shape[1]
    samp = ComboSampler(temperature = 0.0, top_k = 1) if args.temp == 0 else \
        ComboSampler(temperature = args.temp, top_p = 0.95, top_k = 20)
    job = Job(input_ids = ids, max_new_tokens = args.tokens, sampler = samp, stop_conditions = [])
    gen.enqueue(job)
    t0 = time.perf_counter(); first = None; toks = []; acc = rej = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            t = r.get("token_ids")
            if t is not None and t.numel():
                if first is None: first = time.perf_counter()
                toks.append(t.view(-1).cpu())
            if r.get("eos"):
                acc, rej = r.get("accepted_draft_tokens", 0), r.get("rejected_draft_tokens", 0)
    g = torch.cat(toks)
    dt = time.perf_counter() - first
    print(f"prompt {P}, generated {g.numel()} in {dt:.0f} s ({g.numel() / dt:.1f} T/s), draft accept {acc}/{acc + rej}", flush = True)
    text_out = tok.decode(g.view(1, -1))[0] if hasattr(tok, "decode") else ""
    for i in range(0, g.numel(), 1024):
        print(f"  @{i}:", tok.decode(g[i:i + 60].view(1, -1))[0].replace("\n", " ")[:150], flush = True)
    rcache = gen.recurrent_cache
    if rcache is not None:
        print("  recurrent cache:", len(rcache), "entries,", dict(getattr(rcache, "metrics", {})), flush = True)
    return g

def check(model, cache, ids, g, args):
    P = ids.shape[1]

    # Reference: chunked prefill of prompt[:-1], then forward over [prompt[-1], generated[:-1]]
    full = torch.cat([ids[0], g]).view(1, -1)
    N = full.shape[1]
    rs = None
    pos = 0
    B = (1, args.cache)
    while pos < P - 1:
        c = min(2048, P - 1 - pos)
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": B, "recurrent_states": rs}
        model.prefill(input_ids = full[:, pos:pos + c], params = p)
        rs = p.get("recurrent_states"); pos += c
    agree = []; lp = []
    while pos < N - 1:
        c = min(512, N - 1 - pos)
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": B, "recurrent_states": rs}
        lg = model.forward(input_ids = full[:, pos:pos + c], params = p).float()[0]
        rs = p.get("recurrent_states")
        tgt = full[0, pos + 1:pos + 1 + c].to(lg.device)
        agree.append((lg.argmax(-1) == tgt).cpu())
        lp.append(torch.log_softmax(lg, -1).gather(1, tgt.unsqueeze(1)).squeeze(1).cpu())
        pos += c
    agree = torch.cat(agree).float(); lp = torch.cat(lp)
    W = args.window
    print("window  agree  mean-logp  min-logp")
    for i in range(0, agree.numel(), W):
        a = agree[i:i + W]; l = lp[i:i + W]
        print(f"{i:6d}  {a.mean():.3f}  {l.mean():8.3f}  {l.min():8.2f}", flush = True)
    print(f"ALL agree {agree.mean():.4f} mean-logp {lp.mean():.4f}")

if __name__ == "__main__":
    main()
