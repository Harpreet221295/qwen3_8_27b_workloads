"""Compare thinking off vs low / medium / xhigh on small math + logic problems."""
import re
from qwen_workloads import QwenClient, Workload
c = QwenClient(); w = Workload("thinking_modes")

PROBLEMS = [
    ("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How many cents does the ball cost? Reply with the number only.", "5"),
    ("If 3 machines make 3 widgets in 3 minutes, how many minutes do 100 machines take to make 100 widgets? Number only.", "3"),
    ("What is 123 * 47? Number only.", "5781"),
    ("How many times does the letter 'r' appear in 'strawberry raspberry'? Number only.", "6"),
]

def make(mode, q, ans):
    def fn():
        r = c.ask(q, thinking=mode, max_tokens=8192 if mode else 64)
        nums = re.findall(r"-?\d+", r.content)
        return {"ok": bool(nums) and nums[-1] == ans, "mode": str(mode), "output": r.content[-80:],
                "reasoning_chars": len(r.reasoning), "tokens": r.completion_tokens, "latency_s": r.latency_s,
                "note": f"mode={mode} tokens={r.completion_tokens}"}
    return fn

for mode in [False, "low", "medium", "xhigh"]:
    for i, (q, a) in enumerate(PROBLEMS):
        w.case(f"{mode}_p{i}", make(mode, q, a))

if __name__ == "__main__": w.run()
