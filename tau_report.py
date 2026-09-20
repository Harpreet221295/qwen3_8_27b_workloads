"""Combine τ²-bench runs into one report, optionally re-judging NL assertions with another model (no re-running conversations).
  .venv/bin/python tau_report.py --runs qwen38_retail_off_k1 qwen38_retail_off_k1_nltasks
  .venv/bin/python tau_report.py --runs qwen38_retail_off_k1_nltasks --rejudge          # judge = gpt-4.1 if OPENAI_API_KEY set
Simulations that hit an infrastructure error (no messages) are excluded and reported separately.
"""
import argparse, json, os, sys
from pathlib import Path
from dotenv import load_dotenv; load_dotenv(".env")
import litellm; litellm.suppress_debug_info = True
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator import evaluator_nl_assertions as _nl
from tau2.utils import DATA_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--rejudge", action="store_true")
ap.add_argument("--judge", default=os.environ.get("TAU2_JUDGE_MODEL", "gpt-4.1-2025-04-14" if os.environ.get("OPENAI_API_KEY") else f"openai/{os.environ.get('QWEN_MODEL','qwen3.8-27b')}"))
a = ap.parse_args()
if a.rejudge:
    _nl.DEFAULT_LLM_NL_ASSERTIONS = a.judge
    _nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0} if not a.judge.startswith("openai/") else \
        {"temperature": 0.0, "api_base": os.environ["QWEN_BASE_URL"], "api_key": os.environ["QWEN_API_KEY"], "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

rows, infra = [], 0
for name in a.runs:
    p = Path(DATA_DIR) / "simulations" / name / "results.json"
    d = json.loads(p.read_text()); tasks = {t["id"]: Task.model_validate(t) for t in d["tasks"]}; domain = d["info"]["environment_info"]["domain_name"] if "environment_info" in d.get("info", {}) else d["info"].get("domain", "?")
    changed = False
    for s in d["simulations"]:
        if not s.get("messages"): infra += 1; continue
        sim = SimulationRun.model_validate(s); task = tasks[s["task_id"]]
        if a.rejudge and (task.evaluation_criteria and task.evaluation_criteria.nl_assertions):
            ri = evaluate_simulation(sim, task, EvaluationType.ALL, solo_mode=False, domain=domain)
            s["reward_info"] = ri.model_dump(mode="json"); s["rejudged_with"] = a.judge; changed = True
        r = s["reward_info"]["reward"]
        rows.append({"run": name, "domain": domain, "task": s["task_id"], "reward": r, "db": (s["reward_info"].get("db_check") or {}).get("db_match"),
                     "nl": [(x.get("met")) for x in (s["reward_info"].get("nl_assertions") or [])]})
    if changed:
        out = p.with_name(f"results_rejudged_{a.judge.split('/')[-1]}.json"); out.write_text(json.dumps(d, indent=1)); print("wrote", out)

n = len(rows); passed = sum(1 for r in rows if r["reward"] >= 1)
print(f"\n{' + '.join(a.runs)}  ·  judge={a.judge if a.rejudge else 'as recorded'}")
print(f"valid simulations: {n}   infra errors excluded: {infra}")
print(f"pass^1 (valid): {passed}/{n} = {passed/max(1,n):.3f}   avg reward: {sum(r['reward'] for r in rows)/max(1,n):.3f}")
nl_rows = [r for r in rows if r["nl"]]
if nl_rows: print(f"NL-assertion tasks: {len(nl_rows)}, all assertions met in {sum(1 for r in nl_rows if all(r['nl']))}")
fails = [r for r in rows if r["reward"] < 1]
print("failed tasks:", ", ".join(f"{r['domain']}/{r['task']}" for r in fails)[:600])
