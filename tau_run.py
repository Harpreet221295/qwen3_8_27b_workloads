"""Batch τ²-bench runs with Qwen as agent and user simulator, through tau2's own runner (checkpointing, retries, pass^k).
  .venv/bin/python tau_run.py --domain airline --num-tasks 10 --trials 3 --concurrency 3
  .venv/bin/python tau_run.py --domain retail --task-ids 0,1,2 --thinking low
Results go to third_party/tau2-bench/data/simulations/<name>.json ; browse with:  .venv/bin/tau2 view
"""
import argparse, os
from dotenv import load_dotenv
load_dotenv()
import litellm; litellm.suppress_debug_info = True
from tau2.data_model.simulation import TextRunConfig
from tau2.run import run_domain

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="airline", choices=["airline", "retail", "telecom", "mock"])
ap.add_argument("--num-tasks", type=int); ap.add_argument("--task-ids", help="comma list")
ap.add_argument("--trials", type=int, default=1); ap.add_argument("--concurrency", type=int, default=3)
ap.add_argument("--max-steps", type=int, default=100); ap.add_argument("--seed", type=int, default=300)
ap.add_argument("--thinking", default="off", choices=["off", "low", "medium", "xhigh"], help="agent thinking mode")
ap.add_argument("--user-thinking", default="off", choices=["off", "low", "medium"])
ap.add_argument("--user-model", help="LiteLLM model for the customer simulator, e.g. gpt-4.1-2025-04-14 (needs OPENAI_API_KEY); default: Qwen")
ap.add_argument("--name", help="save name (default auto)")
a = ap.parse_args()

BASE, KEY, MODEL = os.environ["QWEN_BASE_URL"], os.environ["QWEN_API_KEY"], os.environ.get("QWEN_MODEL", "qwen3.8-27b")

# --- NL-assertion judge. Official τ²-bench uses gpt-4.1. If OPENAI_API_KEY is set we do the same; otherwise our own endpoint
# judges (indicative only). Override with TAU2_JUDGE_MODEL / TAU2_JUDGE_API_BASE / TAU2_JUDGE_API_KEY.
from tau2.evaluator import evaluator_nl_assertions as _nl
_have_openai = bool(os.environ.get("OPENAI_API_KEY"))
JUDGE_MODEL = os.environ.get("TAU2_JUDGE_MODEL", "gpt-4.1-2025-04-14" if _have_openai else f"openai/{MODEL}")
_nl.DEFAULT_LLM_NL_ASSERTIONS = JUDGE_MODEL
if JUDGE_MODEL.startswith("openai/") or os.environ.get("TAU2_JUDGE_API_BASE"):
    _nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0, "api_base": os.environ.get("TAU2_JUDGE_API_BASE", BASE),
                                          "api_key": os.environ.get("TAU2_JUDGE_API_KEY", KEY),
                                          "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
else:
    _nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0}          # real OpenAI via OPENAI_API_KEY

def args(mode):
    if mode == "off": return {"api_base": BASE, "api_key": KEY, "temperature": 0.7, "top_p": 0.8, "extra_body": {"top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}}}
    return {"api_base": BASE, "api_key": KEY, "temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 20, "chat_template_kwargs": {"reasoning_effort": mode}}}

cfg = TextRunConfig(domain=a.domain, agent="llm_agent", user="user_simulator", llm_agent=f"openai/{MODEL}", llm_args_agent=args(a.thinking),
                    llm_user=(a.user_model or f"openai/{MODEL}"), llm_args_user=({"temperature": 0.0} if a.user_model else args(a.user_thinking)), num_trials=a.trials, max_steps=a.max_steps, max_errors=10,
                    max_concurrency=a.concurrency, seed=a.seed, log_level="WARNING",
                    task_ids=a.task_ids.split(",") if a.task_ids else None, num_tasks=a.num_tasks,
                    save_to=a.name or f"qwen38_{a.domain}_{a.thinking}_k{a.trials}" + ("_user-" + a.user_model.split("/")[-1] if a.user_model else ""))
print(f"agent={cfg.llm_agent} · user={cfg.llm_user} · judge={JUDGE_MODEL}")
run_domain(cfg)
