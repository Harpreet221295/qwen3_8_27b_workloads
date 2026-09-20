# τ²-bench on Qwen3.8-27B: setup and usage

## What τ²-bench is
Sierra's benchmark for tool-agent-user interaction: a customer-service **agent** (the model under test) with a policy document
and a tool API over a mock database, talking to a **simulated customer** (an LLM following a hidden script) about airline,
retail or telecom problems. A task passes when the final database state matches the expected one (plus optional action /
NL-assertion / communicate checks). Telecom adds "dual control": the customer has phone tools too. Metric: pass^k = fraction of
tasks solved in *all* k trials. Repo: https://github.com/sierra-research/tau2-bench (v1.0.1, Python ≥3.12).

We use the real thing, not a re-implementation.

## Install (already done in this repo)
```bash
git submodule update --init                       # third_party/tau2-bench
.venv/bin/pip install -e third_party/tau2-bench   # uses litellm under the hood
.venv/bin/pip install websockets audioop-lts      # missing on py3.13: websockets pulled by voice code, audioop removed in 3.13
.venv/bin/python -c "import tau2; from tau2.run import get_tasks; print(len(get_tasks('airline')))"   # 50
```
Domains and task counts: airline 50, retail 114, telecom 2285 (variants × personas), mock 10, banking_knowledge (needs extras).

## Wiring Qwen in
tau2 calls models through LiteLLM, so the vLLM endpoint is `openai/<served-name>` with `api_base`/`api_key` in `llm_args`:
```python
llm = "openai/qwen3.8-27b"
llm_args = {"api_base": QWEN_BASE_URL, "api_key": QWEN_API_KEY, "temperature": 0.7, "top_p": 0.8,
            "extra_body": {"top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}}}
TextRunConfig(domain="airline", agent="llm_agent", user="user_simulator",
              llm_agent=llm, llm_args_agent=llm_args, llm_user=llm, llm_args_user=llm_args, ...)
```
The same Qwen plays the user simulator. The official leaderboard uses a stronger simulator; our numbers are therefore
indicative, not head-to-head. `litellm.register_model` silences the "model isn't mapped" cost warning.

## Batch runs (the numbers)
```bash
.venv/bin/python tau_run.py --domain airline --trials 1 --concurrency 3
.venv/bin/python tau_run.py --domain retail  --trials 3 --concurrency 3 --thinking low
.venv/bin/python tau_run.py --domain telecom --num-tasks 50
```
- Goes through tau2's own `run_domain` (checkpointing, retries, metrics table at the end).
- Results: `third_party/tau2-bench/data/simulations/<name>/` (gitignored). Browse with `.venv/bin/tau2 view`.
- `--thinking off|low|medium|xhigh` for the agent, `--user-thinking` for the simulator. Seed default 300.
- Concurrency 3 is comfortable for one A100; each conversation is 10–40 model calls.

## Console (port 7862)
```bash
.venv/bin/python -m uvicorn tau_ui.app:app --port 7862     # or ../start_uis.sh
```
`tau_ui/app.py` builds tau2's environment, agent and user itself and drives `Orchestrator.initialize()/step()` in a thread,
streaming every trajectory message over SSE, then calls tau2's `evaluate_simulation`. Modes:
- **simulated**: tau2's `UserSimulator` (Qwen) plays the customer from the task script.
- **I play the customer**: a `HumanUser` subclass blocks on a queue; your typed text becomes the user message; "End & grade" sends `###STOP###`.
The left panel shows the task's `known_info` / `reason_for_call` / `task_instructions` (your character sheet) and, as a
spoiler, the expected actions. "Policy & tools" shows what the agent was told. Single runs are saved to `results/tau/<domain>/`.
Limitation: telecom customers use phone tools (toggle roaming, speed test…), which only the simulator can call, so play
telecom in simulated mode.

## Results so far (thinking off, 1 trial, seed 300)
| domain | pass^1 | notes |
|---|---|---|
| airline | 0.74 (37/50) | reads 92% correct, writes 67%, DB match 76%, all runs ended normally |
| retail | running | |

## Reading a failure
`tau2 view` → pick the run → the failed task shows the expected vs actual DB diff and the conversation. Typical airline
misses: wrong payment method on a change, a cancellation that policy forbids, or missing a required `send_certificate`.

## Caveats
- The user simulator matters: a weak simulator can hallucinate details and make tasks unsolvable or trivially easy.
- τ²-bench-Verified (amazon-agi) fixes some inconsistent tasks; worth switching to for publishable numbers.
- Costs: ~50 conversations ≈ 20 min of pod time.
