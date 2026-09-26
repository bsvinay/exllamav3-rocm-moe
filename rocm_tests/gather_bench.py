# Raw n-gram row gather speed from the safetensors file (random rows, like one prefill chunk)
import os, sys, time, json, struct, torch
from exllamav3.ext import exllamav3_ext as ext
M = sys.argv[1]
f = os.path.join(M, "ngram_embedding.safetensors")
with open(f, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
keys = [k for k in h if k != "__metadata__"]
k = [x for x in keys if "shard_7" in x][0]; meta = h[k]
rows, cols = meta["shape"][0], meta["shape"][1]
row_bytes = cols * 2
base = 8 + n + meta["data_offsets"][0]
fd = os.open(f, os.O_RDONLY)
if len(sys.argv) > 2 and sys.argv[2] == "random": os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
print(f"{len(keys)} shards, shard {k}: {rows} rows x {row_bytes} B")
for U in (16, 2048, 32768):
    uids = torch.sort(torch.randint(0, rows, (U,))).values.unique()
    out = torch.empty((uids.numel(), cols), dtype = torch.int16)
    t0 = time.perf_counter()
    ext.ngram_gather_cpu(fd, base, row_bytes, uids, 0, out)
    t = time.perf_counter() - t0
    print(f"U={uids.numel():6d}: {t * 1000:8.1f} ms  ({uids.numel() / t:9.0f} rows/s)")
