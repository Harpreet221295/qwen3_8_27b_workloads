"""Async load test: throughput and latency percentiles."""
import argparse, asyncio, statistics, time
from qwen_workloads import QwenClient, save_results
from rich.console import Console
console = Console()

PROMPTS = ["Explain what a KV cache is in 3 sentences.", "Write a haiku about Toronto winters.",
           "List 5 uses of Python.", "What is tensor parallelism? Two sentences.", "Give 3 tips for interviews."]

async def one(c, i, sem, max_tokens):
    async with sem:
        t = time.perf_counter()
        try:
            r = await c.achat([{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], max_tokens=max_tokens)
            return {"i": i, "ok": True, "latency_s": r.latency_s, "completion_tokens": r.completion_tokens, "prompt_tokens": r.prompt_tokens}
        except Exception as e:
            return {"i": i, "ok": False, "latency_s": time.perf_counter() - t, "error": str(e)[:200]}

async def main(a):
    c = QwenClient(); sem = asyncio.Semaphore(a.concurrency)
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[one(c, i, sem, a.max_tokens) for i in range(a.requests)])
    wall = time.perf_counter() - t0
    oks = [r for r in rows if r["ok"]]; lats = sorted(r["latency_s"] for r in oks)
    toks = sum(r["completion_tokens"] for r in oks)
    pct = lambda p: lats[min(len(lats) - 1, int(p * len(lats)))] if lats else 0
    summary = {"requests": a.requests, "concurrency": a.concurrency, "ok": len(oks), "wall_s": round(wall, 1),
               "req_per_s": round(len(oks) / wall, 2), "gen_tok_per_s": round(toks / wall, 1),
               "p50_s": round(pct(0.5), 2), "p95_s": round(pct(0.95), 2), "mean_s": round(statistics.mean(lats), 2) if lats else 0}
    console.print(summary); save_results("concurrency_bench", rows + [{"summary": summary}])
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=200)
    asyncio.run(main(ap.parse_args()))
