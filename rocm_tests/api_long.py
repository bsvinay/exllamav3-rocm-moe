# Long code-context request through the OpenAI endpoint: TTFT and decode chunk rate
import json, sys, time, glob, os, urllib.request
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1/chat/completions"
target_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 110000
root = os.path.expanduser("~/exl3-moe/exllamav3/modules")
text = ""
for f in sorted(glob.glob(root + "/*.py")):
    if len(text) > target_chars: break
    text += f"\n\n# ===== {os.path.basename(f)} =====\n" + open(f).read()
text = text[:target_chars]
q = "Above is part of a codebase. Explain how the block-sparse MLP chooses between its forward paths, then write a short unit test for one of them."
body = {"model": "x", "messages": [{"role": "user", "content": text + "\n\n" + q}], "stream": True, "max_tokens": 600,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
for rep in range(2):
    req = urllib.request.Request(url, data = json.dumps(body).encode(), headers = {"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; out = []
    with urllib.request.urlopen(req, timeout = 1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if first is None: first = time.time()
                n += 1; out.append(ch[0]["delta"]["content"])
    print(f"rep {rep}: {len(text)} chars, TTFT {first - t0:.1f} s, {n} chunks in {time.time() - first:.1f} s", flush = True)
    print("   ", "".join(out)[:160].replace("\n", " "), flush = True)
