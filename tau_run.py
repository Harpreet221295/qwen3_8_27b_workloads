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
ap.add_argument("--name", help="save name (default auto)")
a = ap.parse_args()

BASE, KEY, MODEL = os.environ["QWEN_BASE_URL"], os.environ["QWEN_API_KEY"], os.environ.get("QWEN_MODEL", "qwen3.8-27b")
# LiteLLM fallback: on some retry paths it drops the api_key kwarg and then fails with "Missing credentials";
# exposing the same values as env vars makes those retries succeed.
os.environ.setdefault("OPENAI_API_KEY", KEY); os.environ.setdefault("OPENAI_API_BASE", BASE); os.environ.setdefault("OPENAI_BASE_URL", BASE)
def args(mode):
    if mode == "off": return {"api_base": BASE, "api_key": KEY, "temperature": 0.7, "top_p": 0.8, "extra_body": {"top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}}}
    return {"api_base": BASE, "api_key": KEY, "temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 20, "chat_template_kwargs": {"reasoning_effort": mode}}}

cfg = TextRunConfig(domain=a.domain, agent="llm_agent", user="user_simulator", llm_agent=f"openai/{MODEL}", llm_args_agent=args(a.thinking),
                    llm_user=f"openai/{MODEL}", llm_args_user=args(a.user_thinking), num_trials=a.trials, max_steps=a.max_steps, max_errors=10,
                    max_concurrency=a.concurrency, seed=a.seed, log_level="WARNING",
                    task_ids=a.task_ids.split(",") if a.task_ids else None, num_tasks=a.num_tasks,
                    save_to=a.name or f"qwen38_{a.domain}_{a.thinking}_k{a.trials}")
run_domain(cfg)
