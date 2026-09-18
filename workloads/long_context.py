"""Needle-in-a-haystack at growing context sizes. Adjust sizes to your --max-model-len."""
import argparse, random
from qwen_workloads import QwenClient, Workload
c = QwenClient(); w = Workload("long_context")
FILLER = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
          "Sphinx of black quartz, judge my vow. How vexingly quick daft zebras jump. ")

def make(n_tokens):
    def fn():
        rnd = random.Random(n_tokens); words = FILLER.split()
        n_words = int(n_tokens * 0.75)
        body = " ".join(rnd.choice(words) for _ in range(n_words))
        code = f"{rnd.randint(100000, 999999)}"
        pos = rnd.randint(int(n_words * 0.2), int(n_words * 0.8))
        toks = body.split(); toks.insert(pos, f" [SECRET CODE: {code}] "); body = " ".join(toks)
        r = c.ask(f"{body}\n\nWhat is the SECRET CODE in the text above? Digits only.", max_tokens=20)
        return {"ok": code in r.content, "output": r.content, "prompt_tokens": r.prompt_tokens, "latency_s": r.latency_s,
                "note": f"prompt_tokens={r.prompt_tokens} depth={pos/n_words:.0%}"}
    return fn

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="8000,32000,100000"); a = ap.parse_args()
    for s in [int(x) for x in a.sizes.split(",")]: w.case(f"niah_{s}", make(s))
    w.run()
else:
    for s in [8000, 32000]: w.case(f"niah_{s}", make(s))
