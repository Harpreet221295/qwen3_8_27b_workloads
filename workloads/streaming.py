"""SSE streaming and time-to-first-token, with and without thinking."""
from qwen_workloads import QwenClient, Workload
c = QwenClient(); w = Workload("streaming")

def make(think):
    def fn():
        content, reasoning, stats = [], [], {}
        for kind, x in c.stream([{"role": "user", "content": "Write a 4-line poem about GPUs."}], thinking=think, max_tokens=1024):
            if kind == "content": content.append(x)
            elif kind == "reasoning": reasoning.append(x)
            else: stats = x
        text = "".join(content)
        return {"ok": bool(text) and stats["chunks"] > 3, "output": text, "ttft_s": stats.get("ttft_s"), "latency_s": stats["total_s"],
                "reasoning_chars": len("".join(reasoning)), "note": f"ttft={stats.get('ttft_s'):.2f}s chunks={stats['chunks']}"}
    return fn

w.case("stream_nothink", make(False)).case("stream_think_low", make("low"))
if __name__ == "__main__": w.run()
