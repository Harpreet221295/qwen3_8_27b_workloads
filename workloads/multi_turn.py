"""State carried across a 6-turn conversation."""
from qwen_workloads import QwenClient, Workload
c = QwenClient(); w = Workload("multi_turn")

def conversation():
    msgs = [{"role": "system", "content": "You are a concise assistant."}]
    script = [
        ("My name is Harpreet and my favorite number is 42. Just say OK.", None),
        ("I live in Toronto. Just say OK.", None),
        ("What is my favorite number plus 8? Number only.", "50"),
        ("Which city do I live in? One word.", "toronto"),
        ("Now forget the number and remember 7 instead. Just say OK.", None),
        ("What's my number times 3? Number only.", "21"),
    ]
    log, ok = [], True
    for user, expect in script:
        msgs.append({"role": "user", "content": user})
        r = c.chat(msgs, max_tokens=40); msgs.append({"role": "assistant", "content": r.content})
        log.append((user, r.content))
        if expect and expect not in r.content.lower(): ok = False
    return {"ok": ok, "output": log}

w.case("six_turns", conversation)
if __name__ == "__main__": w.run()
