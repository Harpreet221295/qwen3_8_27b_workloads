"""Single-shot coding: generated Python is executed locally against hidden tests."""
import re, subprocess, sys, tempfile, textwrap
from qwen_workloads import QwenClient, Workload
c = QwenClient(); w = Workload("coding_snippets")

def extract_code(s: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", s, re.S)
    return m.group(1) if m else s

def run_python(code: str, test: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + textwrap.dedent(test)); path = f.name
    p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
    return p.returncode == 0, (p.stdout + p.stderr)[-500:]

TASKS = [
    ("two_sum", "Write a Python function two_sum(nums, target) returning indices of two numbers adding to target. Only code in one ```python block.",
     "assert sorted(two_sum([2,7,11,15], 9)) == [0,1]\nassert sorted(two_sum([3,2,4], 6)) == [1,2]\nprint('ok')", False),
    ("lru_cache", "Implement class LRUCache(capacity) with get(key)->int(-1 if missing) and put(key,value), O(1) each. Only code in one ```python block.",
     "c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1; assert c.get(3)==3; assert c.get(4)==4; print('ok')", False),
    ("fix_bug", "This function should return the median of a list but is buggy. Fix it. Only code in one ```python block.\n\n```python\ndef median(xs):\n    xs.sort()\n    n = len(xs)\n    return xs[n//2]\n```",
     "assert median([3,1,2])==2\nassert median([4,1,3,2])==2.5\nprint('ok')", False),
    ("parse_log", "Write parse_log(text)->dict counting occurrences of each log level (INFO/WARN/ERROR) in lines formatted like '2026-01-01 12:00:00 [LEVEL] message'. Only code in one ```python block.",
     "t='2026-01-01 12:00:00 [INFO] a\\n2026-01-01 12:00:01 [ERROR] b\\n2026-01-01 12:00:02 [INFO] c\\n'\nr=parse_log(t); assert {k:v for k,v in r.items() if v}=={'INFO':2,'ERROR':1}, r\nprint('ok')", False),
    ("interval_merge_thinking", "Write merge_intervals(intervals) merging overlapping [start,end] pairs, returning sorted merged list. Only code in one ```python block.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nprint('ok')", "low"),
]

def make(name, prompt, test, think):
    def fn():
        r = c.ask(prompt, thinking=think, max_tokens=4096 if think else 800)
        ok, out = run_python(extract_code(r.content), test)
        return {"ok": ok, "output": r.content[-400:], "exec": out, "latency_s": r.latency_s, "note": out.strip()[-80:]}
    return fn

for name, p, t, th in TASKS: w.case(name, make(name, p, t, th))
if __name__ == "__main__": w.run()
