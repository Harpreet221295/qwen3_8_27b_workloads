"""Chart reading with known values."""
import re
from qwen_workloads import QwenClient, Workload, to_data_url, make_bar_chart
c = QwenClient(); w = Workload("vision_chart")
VALUES = {"Jan": 120, "Feb": 95, "Mar": 150, "Apr": 80, "May": 170, "Jun": 130}
URL = to_data_url(make_bar_chart(VALUES))

def highest():
    r = c.ask_image("Which month has the highest value? One word.", [URL], max_tokens=10)
    return {"ok": "may" in r.content.lower(), "output": r.content, "latency_s": r.latency_s}

def read_value():
    r = c.ask_image("What is the value for March? Number only.", [URL], max_tokens=10)
    return {"ok": "150" in r.content, "output": r.content, "latency_s": r.latency_s}

def total():
    r = c.ask_image("Sum all six bar values. Show the number only.", [URL], thinking="low", max_tokens=2048)
    n = re.findall(r"\d+", r.content)
    return {"ok": bool(n) and int(n[-1]) == sum(VALUES.values()), "output": r.content[-60:], "latency_s": r.latency_s}

def to_table():
    r = c.ask_image("Return the chart data as JSON object month->value.", [URL], max_tokens=150,
                    response_format={"type": "json_object"})
    import json
    try: d = json.loads(r.content); hits = sum(1 for k, v in VALUES.items() if int(d.get(k, -1)) == v)
    except Exception: hits = 0
    return {"ok": hits >= 5, "output": r.content, "latency_s": r.latency_s, "note": f"{hits}/6 values exact"}

for n, f in [("highest", highest), ("read_value", read_value), ("total", total), ("to_table", to_table)]: w.case(n, f)
if __name__ == "__main__": w.run()
