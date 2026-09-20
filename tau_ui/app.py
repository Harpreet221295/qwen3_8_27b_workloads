"""τ²-bench console for the deployed Qwen3.8-27B.
Run:  .venv/bin/python -m uvicorn tau_ui.app:app --port 7862   → http://localhost:7862
Uses the real tau2-bench domains, tasks, user simulator, orchestrator and evaluator (third_party/tau2-bench).
Qwen plays the agent; the customer is either tau2's LLM user simulator (also Qwen) or you, typing."""
from __future__ import annotations
import asyncio, json, os, queue, threading, time, traceback, uuid
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("QWEN_API_KEY", "none")
MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b")
LLM = f"openai/{MODEL}"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "tau"
STATIC = Path(__file__).resolve().parent / "static"

import litellm
litellm.suppress_debug_info = True
try: litellm.register_model({MODEL: {"input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"},
                             LLM: {"input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}})
except Exception: pass
from loguru import logger
logger.remove(); logger.add(lambda m: None, level="ERROR")  # keep tau2 quiet; we stream our own events

from tau2.run import get_tasks
from tau2.registry import registry
from tau2.runner import build_environment, build_agent, build_user
from tau2.orchestrator import orchestrator as om
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

# --- NL-assertion judge: tau2 defaults to gpt-4.1 (needs an OpenAI key). Route it to our endpoint unless TAU2_JUDGE_* say otherwise.
from tau2.evaluator import evaluator_nl_assertions as _nl
_nl.DEFAULT_LLM_NL_ASSERTIONS = os.environ.get("TAU2_JUDGE_MODEL", LLM)
_nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0, "api_base": os.environ.get("TAU2_JUDGE_API_BASE", BASE_URL),
                                      "api_key": os.environ.get("TAU2_JUDGE_API_KEY", API_KEY),
                                      "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
from tau2.user.user_simulator import UserSimulator, UserState
from tau2.data_model.message import UserMessage, AssistantMessage, ToolMessage

DOMAINS = ["airline", "retail", "telecom", "mock"]
app = FastAPI(title="tau2 console")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_ENV_CACHE: dict = {}
JOBS: dict[str, dict] = {}


def llm_args(thinking: str, temperature: float | None = None) -> dict:
    if thinking == "off":
        return {"api_base": BASE_URL, "api_key": API_KEY, "temperature": 0.7 if temperature is None else temperature, "top_p": 0.8,
                "extra_body": {"top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}}}
    return {"api_base": BASE_URL, "api_key": API_KEY, "temperature": 1.0 if temperature is None else temperature, "top_p": 0.95,
            "extra_body": {"top_k": 20, "chat_template_kwargs": {"reasoning_effort": thinking}}}


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def task_summary(t) -> dict:
    ui = t.user_scenario.instructions
    ins = ui.model_dump() if hasattr(ui, "model_dump") else {"task_instructions": str(ui)}
    ec = t.evaluation_criteria
    return {"id": t.id, "purpose": (t.description.purpose if t.description else None),
            "persona": t.user_scenario.persona, "reason_for_call": ins.get("reason_for_call"), "known_info": ins.get("known_info"),
            "unknown_info": ins.get("unknown_info"), "task_instructions": ins.get("task_instructions"),
            "reward_basis": [str(b.value if hasattr(b, "value") else b) for b in (ec.reward_basis or [])] if ec else [],
            "expected_actions": [{"name": a.name, "arguments": a.arguments, "requestor": str(a.requestor)} for a in (ec.actions or [])] if ec else [],
            "nl_assertions": (ec.nl_assertions or []) if ec else [], "communicate_info": (ec.communicate_info or []) if ec else [],
            "has_initial_state": t.initial_state is not None}


def msg_to_dict(m) -> dict:
    d = {"role": m.role, "content": getattr(m, "content", None), "turn": getattr(m, "turn_idx", None)}
    tcs = getattr(m, "tool_calls", None) or []
    d["tool_calls"] = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tcs]
    if isinstance(m, ToolMessage): d.update({"error": m.error, "requestor": m.requestor, "id": m.id})
    raw = getattr(m, "raw_data", None) or {}
    try:
        mm = raw.get("choices", [{}])[0].get("message", {})
        d["reasoning"] = mm.get("reasoning_content") or mm.get("reasoning") or (mm.get("provider_specific_fields") or {}).get("reasoning_content")
    except Exception: d["reasoning"] = None
    u = getattr(m, "usage", None)
    if u: d["usage"] = {k: u.get(k) for k in ("prompt_tokens", "completion_tokens")} if isinstance(u, dict) else None
    d["gen_s"] = getattr(m, "generation_time_seconds", None)
    return d


class HumanUser(UserSimulator):
    """A user whose messages come from the UI (blocking queue) instead of an LLM."""
    def __init__(self, inbox: queue.Queue):
        super().__init__(llm="dummy"); self.inbox = inbox
    def get_init_state(self, message_history=None): return UserState(messages=[], system_messages=[])
    def set_seed(self, seed): pass
    def generate_next_message(self, message, state):
        text = self.inbox.get()
        um = UserMessage(role="user", content=text); state.messages.append(um); return um, state


@app.get("/")
async def index(): return FileResponse(STATIC / "index.html")


@app.get("/api/info")
async def info():
    return {"model": MODEL, "base_url": BASE_URL, "domains": DOMAINS, "tau2_version": getattr(__import__("tau2"), "__version__", "src")}


@app.get("/api/domain/{domain}")
async def domain_info(domain: str):
    env = build_environment(domain)
    tools = [{"name": t.name, "description": (t.openai_schema.get("function", {}).get("description") or "")[:200]} for t in env.get_tools()]
    try: user_tools = [t.name for t in env.get_user_tools()]
    except Exception: user_tools = []           # airline/retail: the customer has no tools
    return {"domain": domain, "policy": env.get_policy(), "tools": tools, "user_tools": user_tools}


@app.get("/api/tasks/{domain}")
async def tasks(domain: str, q: str = "", limit: int = 200):
    ts = get_tasks(domain)
    out = []
    for t in ts:
        s = task_summary(t)
        blob = json.dumps(s).lower()
        if q and q.lower() not in blob: continue
        out.append(s)
        if len(out) >= limit: break
    return {"total": len(ts), "shown": len(out), "tasks": out}


@app.post("/api/run")
async def run(req: Request):
    body = await req.json()
    domain, task_id = body["domain"], body["task_id"]
    mode = body.get("mode", "sim")                      # sim | human
    a_think = body.get("agent_thinking", "off"); u_think = body.get("user_thinking", "off")
    max_steps = int(body.get("max_steps", 60)); seed = int(body.get("seed", 0)) or None
    task = get_tasks(domain, task_ids=[task_id])[0]
    job = uuid.uuid4().hex[:8]; q: queue.Queue = queue.Queue(); inbox: queue.Queue = queue.Queue()
    JOBS[job] = {"inbox": inbox, "stop": threading.Event()}
    emit = lambda k, d: q.put((k, d))

    def worker():
        t0 = time.time(); sim = None
        try:
            env = build_environment(domain)
            agent = build_agent("llm_agent", env, llm=LLM, llm_args=llm_args(a_think), task=task)
            user = HumanUser(inbox) if mode == "human" else build_user("user_simulator", env, task, llm=LLM, llm_args=llm_args(u_think))
            orch = Orchestrator(domain=domain, agent=agent, user=user, environment=env, task=task, max_steps=max_steps, max_errors=10, seed=seed,
                                simulation_id=f"ui-{job}")
            orch._run_start_time = om.get_now(); orch._run_start_perf = time.perf_counter()
            emit("status", {"t": "initializing environment"}); orch.initialize()
            seen = 0
            def flush():
                nonlocal seen
                for m in orch.trajectory[seen:]: emit("message", msg_to_dict(m))
                seen = len(orch.trajectory)
            flush()
            while not orch.done:
                if JOBS[job]["stop"].is_set():
                    orch.done = True; orch.termination_reason = om.TerminationReason.USER_STOP; break
                to = str(orch.to_role).split(".")[-1].lower()
                emit("status", {"t": {"agent": "agent thinking…", "user": ("waiting for you…" if mode == "human" else "customer (simulated) typing…"), "env": "executing tools…"}.get(to, to), "to": to})
                orch.step(); orch._check_termination(); flush()
            sim = orch._finalize()
            emit("status", {"t": "evaluating…"})
            ri = evaluate_simulation(sim, task, EvaluationType.ALL, solo_mode=False, domain=domain)
            sim.reward_info = ri
            emit("reward", {"reward": ri.reward, "termination": str(sim.termination_reason).split(".")[-1], "info": ri.model_dump(mode="json"),
                            "expected_actions": task_summary(task)["expected_actions"], "duration_s": round(time.time() - t0, 1),
                            "agent_actions": [{"name": tc.name, "arguments": tc.arguments} for m in sim.messages if isinstance(m, AssistantMessage) for tc in (m.tool_calls or [])]})
            out = RESULTS / domain; out.mkdir(parents=True, exist_ok=True)
            p = out / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_id.replace('/', '_')[:60]}_{mode}_{a_think}.json"
            p.write_text(json.dumps({"domain": domain, "task_id": task_id, "mode": mode, "agent_thinking": a_think, "user_thinking": u_think,
                                     "reward": ri.reward, "termination": str(sim.termination_reason), "duration_s": round(time.time() - t0, 1),
                                     "reward_info": ri.model_dump(mode="json"), "messages": [msg_to_dict(m) for m in sim.messages]}, indent=1, default=str))
            emit("saved", {"path": str(p.relative_to(RESULTS.parent.parent))})
        except Exception as e:
            emit("error", {"message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
        finally:
            q.put(None); JOBS.pop(job, None)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        yield sse("job", {"job": job, "mode": mode})
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None: yield sse("end", {}); break
            yield sse(*item)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/jobs/{job}/say")
async def say(job: str, req: Request):
    if job not in JOBS: return JSONResponse({"error": "no such job"}, status_code=404)
    body = await req.json(); JOBS[job]["inbox"].put((body.get("text") or "").strip() or "…"); return {"ok": True}


@app.post("/api/jobs/{job}/stop")
async def stop(job: str):
    if job not in JOBS: return {"ok": False}
    JOBS[job]["stop"].set(); JOBS[job]["inbox"].put("###STOP###"); return {"ok": True}


@app.get("/api/results")
async def results():
    rows = []
    for p in sorted(RESULTS.glob("*/*.json")):
        try: d = json.loads(p.read_text())
        except Exception: continue
        rows.append({"domain": d["domain"], "task_id": d["task_id"], "mode": d["mode"], "agent_thinking": d.get("agent_thinking"), "reward": d["reward"],
                     "termination": d["termination"].split(".")[-1], "duration_s": d.get("duration_s"), "file": p.name, "n_msgs": len(d.get("messages", []))})
    # per domain: pass@1 (mean reward over sim runs) and pass^k over tasks with k sim trials
    agg = {}
    for r in rows:
        if r["mode"] != "sim": continue
        a = agg.setdefault(r["domain"], {"trials": 0, "passed": 0, "by_task": {}})
        a["trials"] += 1; a["passed"] += int(r["reward"] >= 1); a["by_task"].setdefault(r["task_id"], []).append(r["reward"] >= 1)
    summary = []
    for dmn, a in agg.items():
        ks = [len(v) for v in a["by_task"].values()]; kmin = min(ks) if ks else 0
        passk = sum(1 for v in a["by_task"].values() if all(v[:kmin])) / max(1, len(a["by_task"])) if kmin else None
        summary.append({"domain": dmn, "tasks": len(a["by_task"]), "trials": a["trials"], "pass_at_1": round(a["passed"] / a["trials"], 3), "k": kmin, "pass_hat_k": None if passk is None else round(passk, 3)})
    return {"rows": rows[::-1], "summary": summary}


@app.get("/api/results/{domain}/{file}")
async def result_detail(domain: str, file: str):
    p = RESULTS / domain / file
    if not p.exists(): return JSONResponse({"error": "not found"}, status_code=404)
    return json.loads(p.read_text())
