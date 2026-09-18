"""Run every workload and write results/summary.md."""
import argparse, importlib, sys
from pathlib import Path
from qwen_workloads.runner import print_summary, RESULTS
from qwen_workloads import QwenClient

ALL = ["chat_basic", "thinking_modes", "multi_turn", "vision_describe", "vision_ocr", "vision_chart",
       "vision_multi_image", "tool_calling", "structured_output", "coding_snippets", "long_context", "streaming"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", help="comma list"); ap.add_argument("--skip", default="")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else [n for n in ALL if n not in a.skip.split(",")]
    c = QwenClient()
    try: print("models:", [m.id for m in c.sync.models.list().data])
    except Exception as e: sys.exit(f"endpoint not reachable at {c.base_url}: {e}")
    summaries = []
    for n in names:
        mod = importlib.import_module(f"workloads.{n}")
        summaries.append(mod.w.run())
    print_summary(summaries)
    md = ["# Workload summary", "", "| workload | cases | passed/scored | avg latency (s) |", "|---|---|---|---|"]
    md += [f"| {s['workload']} | {s['cases']} | {s['passed']}/{s['scored']} | {s['avg_latency_s']:.1f} |" for s in summaries]
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n"); print("wrote", RESULTS / "summary.md")
