"""Tiny workload framework: collect cases, run, save JSONL, print a table."""
from __future__ import annotations
import json, time, traceback
from pathlib import Path
from typing import Callable
from rich.console import Console
from rich.table import Table

RESULTS = Path(__file__).resolve().parent.parent / "results"
console = Console()


def save_results(name: str, rows: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{name}.jsonl"
    with p.open("w") as f:
        for r in rows: f.write(json.dumps(r, default=str) + "\n")
    return p


class Workload:
    """Subclass or instantiate; add cases with .case(name, fn). fn returns dict with at least
    'ok' (bool|None) and optionally 'latency_s', 'tokens', 'note', 'output'."""
    def __init__(self, name: str):
        self.name = name; self.cases: list[tuple[str, Callable[[], dict]]] = []

    def case(self, name: str, fn: Callable[[], dict]):
        self.cases.append((name, fn)); return self

    def run(self, quiet: bool = False) -> dict:
        rows = []
        for cname, fn in self.cases:
            t = time.perf_counter()
            try:
                out = fn() or {}
            except Exception as e:
                out = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
            out.setdefault("latency_s", time.perf_counter() - t)
            out.update({"workload": self.name, "case": cname, "ts": time.time()})
            rows.append(out)
            if not quiet:
                mark = "✅" if out.get("ok") else ("➖" if out.get("ok") is None else "❌")
                console.print(f"{mark} [bold]{self.name}/{cname}[/] {out['latency_s']:.1f}s  {str(out.get('note',''))[:100]}")
        path = save_results(self.name, rows)
        scored = [r for r in rows if r.get("ok") is not None]
        summary = {"workload": self.name, "cases": len(rows), "passed": sum(1 for r in scored if r["ok"]),
                   "scored": len(scored), "avg_latency_s": sum(r["latency_s"] for r in rows) / max(1, len(rows)),
                   "path": str(path)}
        if not quiet: console.print(f"→ {summary['passed']}/{summary['scored']} passed, saved {path}\n")
        return summary


def print_summary(summaries: list[dict]):
    t = Table(title="Qwen3.8-27B workload summary")
    for c in ["workload", "cases", "passed/scored", "avg latency (s)"]: t.add_column(c)
    for s in summaries:
        t.add_row(s["workload"], str(s["cases"]), f"{s['passed']}/{s['scored']}", f"{s['avg_latency_s']:.1f}")
    console.print(t)
