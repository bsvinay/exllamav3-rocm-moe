# Fetch the question sets for iqbench.py into ~/iqbench (or DIR): 150 random MMLU-Pro test questions, 40 random
# MATH-500 problems of level 4-5, and all of HumanEval. Needs the `datasets` package.
import json, os, random, sys
from datasets import load_dataset

out = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/iqbench")
os.makedirs(out, exist_ok = True)
random.seed(0)
mp = list(load_dataset("TIGER-Lab/MMLU-Pro", split = "test"))
random.shuffle(mp)
json.dump([{k: r[k] for k in ("question", "options", "answer", "category")} for r in mp[:150]],
          open(os.path.join(out, "mmlu_pro.json"), "w"))
hard = [r for r in load_dataset("HuggingFaceH4/MATH-500", split = "test") if r["level"] >= 4]
random.shuffle(hard)
json.dump([{k: r[k] for k in ("problem", "answer", "level", "subject")} for r in hard[:40]],
          open(os.path.join(out, "math500.json"), "w"))
he = list(load_dataset("openai/openai_humaneval", split = "test"))
json.dump([{k: r[k] for k in ("task_id", "prompt", "test", "entry_point")} for r in he],
          open(os.path.join(out, "humaneval.json"), "w"))
print(f"wrote {out}: MMLU-Pro 150, MATH-500 L4-5 40, HumanEval {len(he)}")
