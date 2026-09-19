"""Chat playground for the deployed Qwen3.8-27B.
Run:  .venv/bin/python -m uvicorn ui.app:app --port 7860 --reload   (then open http://localhost:7860)
Streams the answer and the model's thinking as separate event streams. The API key never leaves this process.
"""
from __future__ import annotations
import base64, json, os, time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("QWEN_API_KEY", "none")
MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b")
RESULTS = Path(__file__).resolve().parent.parent / "results"
STATIC = Path(__file__).resolve().parent / "static"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600)
app = FastAPI(title="Qwen3.8-27B playground")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

DEMO_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies",
        "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_ccy": {"type": "string"}, "to_ccy": {"type": "string"}}, "required": ["amount", "from_ccy", "to_ccy"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
RATES = {("USD", "CAD"): 1.36, ("CAD", "USD"): 0.74, ("EUR", "INR"): 97.2, ("USD", "INR"): 88.5, ("EUR", "USD"): 1.09}


def run_demo_tool(name: str, args: dict) -> dict:
    if name == "get_weather": return {"city": args.get("city"), "temp_c": 21, "sky": "partly cloudy", "note": "fake demo data"}
    if name == "convert_currency":
        r = RATES.get((args.get("from_ccy", "").upper(), args.get("to_ccy", "").upper()), 1.0)
        return {"result": round(float(args.get("amount", 0)) * r, 2), "rate": r, "note": "fake demo rate"}
    if name == "calculator":
        try: return {"result": eval(args.get("expression", "0"), {"__builtins__": {}}, {})}
        except Exception as e: return {"error": str(e)}
    return {"error": f"unknown tool {name}"}


def sampling(mode: str, temperature: float | None):
    if mode == "off":
        return dict(temperature=0.7 if temperature is None else temperature, top_p=0.8,
                    extra_body={"top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}})
    return dict(temperature=1.0 if temperature is None else temperature, top_p=0.95,
                extra_body={"top_k": 20, "chat_template_kwargs": {"reasoning_effort": mode}})


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
async def index(): return FileResponse(STATIC / "index.html")


@app.get("/api/info")
async def info():
    try:
        models = await client.models.list(); mm = models.data[0]
        return {"base_url": BASE_URL, "model": MODEL, "served": mm.id, "max_model_len": getattr(mm, "max_model_len", None), "ok": True}
    except Exception as e:
        return {"base_url": BASE_URL, "model": MODEL, "ok": False, "error": str(e)[:200]}


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    messages = body["messages"]                      # [{role, content}] ; content may be a list with image parts
    mode = body.get("thinking", "off")               # off | low | medium | xhigh
    max_tokens = int(body.get("max_tokens", 4096))
    use_tools = bool(body.get("tools", False))
    kw = sampling(mode, body.get("temperature"))
    if body.get("system"): messages = [{"role": "system", "content": body["system"]}] + messages

    async def gen():
        msgs = list(messages); t0 = time.perf_counter(); first = None
        for hop in range(6):  # tool loop
            reasoning, content, calls = [], [], {}
            usage = None
            try:
                stream = await client.chat.completions.create(
                    model=MODEL, messages=msgs, stream=True, max_tokens=max_tokens,
                    stream_options={"include_usage": True}, **({"tools": DEMO_TOOLS, "tool_choice": "auto"} if use_tools else {}), **kw)
                async for chunk in stream:
                    if chunk.usage: usage = chunk.usage
                    if not chunk.choices: continue
                    d = chunk.choices[0].delta
                    rc = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                    if rc:
                        if first is None: first = time.perf_counter() - t0; yield sse("ttft", {"s": first})
                        reasoning.append(rc); yield sse("reasoning", {"t": rc})
                    if d.content:
                        if first is None: first = time.perf_counter() - t0; yield sse("ttft", {"s": first})
                        content.append(d.content); yield sse("content", {"t": d.content})
                    for tc in (d.tool_calls or []):
                        c = calls.setdefault(tc.index, {"id": tc.id or f"call_{tc.index}", "name": "", "arguments": ""})
                        if tc.id: c["id"] = tc.id
                        if tc.function and tc.function.name: c["name"] += tc.function.name
                        if tc.function and tc.function.arguments: c["arguments"] += tc.function.arguments
            except Exception as e:
                yield sse("error", {"message": str(e)[:500]}); return
            if not calls:
                break
            # execute demo tools, append, loop
            msgs.append({"role": "assistant", "content": "".join(content) or None,
                         "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls.values()]})
            for c in calls.values():
                try: args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError: args = {"_raw": c["arguments"]}
                out = run_demo_tool(c["name"], args)
                yield sse("tool_call", {"name": c["name"], "args": args, "result": out})
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
        total = time.perf_counter() - t0
        yield sse("done", {"total_s": total, "ttft_s": first, "reasoning_chars": sum(map(len, reasoning)),
                           "prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None),
                           "tok_per_s": (usage.completion_tokens / max(1e-6, total - (first or 0))) if usage else None})
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/results")
async def results_list():
    out = []
    for p in sorted(RESULTS.glob("*.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if "case" in r]
        scored = [r for r in rows if r.get("ok") is not None]
        out.append({"workload": p.stem, "cases": len(rows), "passed": sum(1 for r in scored if r["ok"]), "scored": len(scored),
                    "avg_latency_s": round(sum(r.get("latency_s", 0) for r in rows) / max(1, len(rows)), 2)})
    return out


@app.get("/api/results/{name}")
async def results_detail(name: str):
    p = RESULTS / f"{name}.jsonl"
    if not p.exists(): return JSONResponse({"error": "not found"}, status_code=404)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in rows: r.pop("raw", None)
    return rows
