# One image request through the OpenAI endpoint (base64 data URL)
import base64, json, sys, time, urllib.request
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098/v1/chat/completions"
img = sys.argv[2] if len(sys.argv) > 2 else "doc/cat.png"
b64 = base64.b64encode(open(img, "rb").read()).decode()
body = {"model": "x", "max_tokens": 200, "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": "Describe this image in two sentences."}]}]}
t0 = time.time()
req = urllib.request.Request(url, data = json.dumps(body).encode(), headers = {"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout = 600).read())
print(f"vision request {time.time() - t0:.1f} s:", r["choices"][0]["message"]["content"][:300].replace("\n", " "))
