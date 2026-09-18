"""JSON mode, JSON schema (guided decoding), regex/choice constraints."""
import json
from qwen_workloads import QwenClient, Workload
c = QwenClient(); w = Workload("structured_output")

def json_object():
    r = c.ask("Give me a JSON object with keys name, age, city for a fictional person.", max_tokens=100, response_format={"type": "json_object"})
    try: d = json.loads(r.content); ok = {"name", "age", "city"} <= set(d)
    except Exception: ok = False
    return {"ok": ok, "output": r.content, "latency_s": r.latency_s}

def json_schema():
    schema = {"type": "object", "properties": {"title": {"type": "string"}, "year": {"type": "integer"},
              "genres": {"type": "array", "items": {"type": "string"}, "minItems": 2}}, "required": ["title", "year", "genres"], "additionalProperties": False}
    r = c.ask("Describe the movie Inception.", max_tokens=150,
              response_format={"type": "json_schema", "json_schema": {"name": "movie", "schema": schema}})
    try: d = json.loads(r.content); ok = d["year"] == 2010 and len(d["genres"]) >= 2
    except Exception: ok = False
    return {"ok": ok, "output": r.content, "latency_s": r.latency_s}

def guided_choice():
    r = c.ask("Is the sentiment of 'I absolutely loved this phone' positive or negative?", max_tokens=5,
              extra_body={"guided_choice": ["positive", "negative"]})
    return {"ok": r.content.strip() == "positive", "output": r.content, "latency_s": r.latency_s}

def guided_regex():
    r = c.ask("Give a fake US phone number.", max_tokens=20, extra_body={"guided_regex": r"\(\d{3}\) \d{3}-\d{4}"})
    import re
    return {"ok": bool(re.fullmatch(r"\(\d{3}\) \d{3}-\d{4}", r.content.strip())), "output": r.content, "latency_s": r.latency_s}

def json_in_thinking_mode():
    r = c.ask("Return JSON {\"answer\": <int>} for 12*12.", thinking="low", max_tokens=1024, response_format={"type": "json_object"})
    try: ok = json.loads(r.content)["answer"] == 144
    except Exception: ok = False
    return {"ok": ok, "output": r.content, "latency_s": r.latency_s}

for n, f in [("json_object", json_object), ("json_schema", json_schema), ("guided_choice", guided_choice),
             ("guided_regex", guided_regex), ("json_in_thinking_mode", json_in_thinking_mode)]: w.case(n, f)
if __name__ == "__main__": w.run()
