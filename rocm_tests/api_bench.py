# Streaming chat-completions benchmark against an OpenAI-compatible server (decode tok/s per prompt)
import json, sys, time, urllib.request
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1/chat/completions"
prompts = {
    "code": "Write a Python function that parses an ISO-8601 duration string (like P1DT2H30M) into seconds, with tests.",
    "prose": "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
}
for name, p in prompts.items():
    body = {"model": "x", "messages": [{"role": "user", "content": p}], "stream": True, "max_tokens": 400,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data = json.dumps(body).encode(), headers = {"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = []
    with urllib.request.urlopen(req, timeout = 600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if first is None: first = time.time()
                n += 1; text.append(ch[0]["delta"]["content"])
    t1 = time.time()
    print(f"[{name}] TTFT {(first - t0) * 1000:.0f} ms, {n} chunks, {(n - 1) / (t1 - first):.1f} chunks/s")
    print("   ", "".join(text)[:200].replace("\n", " "))
