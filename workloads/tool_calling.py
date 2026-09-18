"""Tool calling through vLLM's --tool-call-parser."""
import json
from qwen_workloads import QwenClient, Workload
c = QwenClient(); w = Workload("tool_calling")

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies",
        "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_ccy": {"type": "string"}, "to_ccy": {"type": "string"}}, "required": ["amount", "from_ccy", "to_ccy"]}}},
]
FAKE = {"get_weather": lambda city, unit="c": {"city": city, "temp": 21, "unit": unit, "sky": "cloudy"},
        "convert_currency": lambda amount, from_ccy, to_ccy: {"result": round(amount * 1.36, 2), "to": to_ccy}}

def single():
    r = c.chat([{"role": "user", "content": "What's the weather in Toronto?"}], tools=TOOLS, max_tokens=200)
    ok = len(r.tool_calls) == 1 and r.tool_calls[0]["name"] == "get_weather" and "toronto" in r.tool_calls[0]["arguments"].lower()
    return {"ok": ok, "output": r.tool_calls or r.content, "latency_s": r.latency_s}

def parallel():
    r = c.chat([{"role": "user", "content": "Get the weather in Paris and in Tokyo."}], tools=TOOLS, max_tokens=300)
    names = [t["name"] for t in r.tool_calls]
    return {"ok": names.count("get_weather") == 2, "output": r.tool_calls or r.content, "latency_s": r.latency_s}

def no_tool_needed():
    r = c.chat([{"role": "user", "content": "What is 2+2? Number only."}], tools=TOOLS, max_tokens=50)
    return {"ok": not r.tool_calls and "4" in r.content, "output": r.tool_calls or r.content, "latency_s": r.latency_s}

def multi_step_loop():
    msgs = [{"role": "user", "content": "Convert 100 USD to CAD, then tell me the weather in the capital of Canada. Finish with a one-line summary."}]
    steps, seen = 0, []
    while steps < 5:
        r = c.chat(msgs, tools=TOOLS, max_tokens=400); steps += 1
        if not r.tool_calls:
            msgs.append({"role": "assistant", "content": r.content}); break
        msgs.append({"role": "assistant", "content": r.content or None,
                     "tool_calls": [{"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}} for t in r.tool_calls]})
        for t in r.tool_calls:
            seen.append(t["name"])
            try: out = FAKE[t["name"]](**json.loads(t["arguments"]))
            except Exception as e: out = {"error": str(e)}
            msgs.append({"role": "tool", "tool_call_id": t["id"], "content": json.dumps(out)})
    final = msgs[-1].get("content") or ""
    ok = "convert_currency" in seen and "get_weather" in seen and "136" in final and bool(final)
    return {"ok": ok, "output": final, "note": f"tools used: {seen}, steps={steps}"}

def thinking_with_tools():
    r = c.chat([{"role": "user", "content": "I have 250 EUR. How much is that in INR? Use a tool."}], tools=TOOLS, thinking="low", max_tokens=2048)
    return {"ok": any(t["name"] == "convert_currency" for t in r.tool_calls), "output": r.tool_calls or r.content,
            "reasoning_chars": len(r.reasoning), "latency_s": r.latency_s}

for n, f in [("single", single), ("parallel", parallel), ("no_tool_needed", no_tool_needed),
             ("multi_step_loop", multi_step_loop), ("thinking_with_tools", thinking_with_tools)]: w.case(n, f)
if __name__ == "__main__": w.run()
