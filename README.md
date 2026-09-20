# qwen3_8_27b_workloads — capability tests for the deployed Qwen3.8-27B

Runs on your Mac against the RunPod endpoint. Every workload is a small script that sends
requests, saves raw results to `results/<workload>.jsonl`, and prints a summary.

## Setup
```bash
cd qwen3_8_27b_workloads
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill QWEN_BASE_URL + QWEN_API_KEY
```

## Playground UI
```bash
.venv/bin/python -m uvicorn ui.app:app --port 7860
```
Open http://localhost:7860. Streams the answer and the model's thinking live, thinking-mode selector
(off / low / medium / xhigh), image attach, optional demo tools (weather, currency, calculator), and a tab
that browses `results/*.jsonl`. The API key stays in the local server.

## τ²-bench console (tool-agent-user multi-turn eval)
Uses the real [tau2-bench](https://github.com/sierra-research/tau2-bench) (git submodule in `third_party/`): its airline / retail /
telecom / mock domains, tasks, policies, LLM user simulator, orchestrator and evaluator. Qwen plays the agent; the customer is
either tau2's simulator (also Qwen) or you.
```bash
git submodule update --init
.venv/bin/pip install -e third_party/tau2-bench websockets audioop-lts
.venv/bin/python -m uvicorn tau_ui.app:app --port 7862        # http://localhost:7862
.venv/bin/python tau_run.py --domain airline --num-tasks 10 --trials 3   # batch, pass^k, via tau2's runner
.venv/bin/tau2 view                                                         # tau2's own result browser
```
Console: domain tabs, task list with the customer's script and the expected actions (spoiler), "simulated" or
"I play the customer" mode, every agent/customer/tool message as it happens, then tau2's reward with DB / action /
NL-assertion / communicate checks. Single runs are saved under `results/tau/`.

## Run
```bash
python run_all.py                 # everything, summary table at the end
python -m workloads.chat_basic    # one workload
python -m workloads.vision_ocr --image path/to/screenshot.png
python -m workloads.concurrency_bench --concurrency 16 --requests 64
```

## Workloads

| Module | What it tests |
|---|---|
| `chat_basic` | plain Q&A, instruction following, refusal sanity, multilingual |
| `thinking_modes` | thinking off vs low / medium / xhigh: latency, tokens, correctness on math |
| `multi_turn` | 6-turn conversation with state carried across turns |
| `vision_describe` | synthetic + real images: description, counting, color, spatial relations |
| `vision_ocr` | text extraction from rendered documents/screenshots |
| `vision_chart` | reads a generated bar chart, answers numeric questions |
| `vision_multi_image` | compares 2–3 images in one request |
| `tool_calling` | single tool, parallel tools, multi-step tool loop, no-tool case |
| `structured_output` | JSON schema / `response_format` and guided decoding |
| `coding_snippets` | write / fix / explain code; checked by executing the Python locally |
| `long_context` | needle-in-a-haystack at 8K / 32K / 100K tokens |
| `streaming` | SSE streaming, time-to-first-token |
| `concurrency_bench` | async load: throughput, p50/p95 latency, tokens/s |

Results go to `results/` (gitignored). `run_all.py` writes `results/summary.md`.

## Notes on the model
- Thinking is on by default; most workloads turn it off for speed and turn it on where it matters.
- Non-thinking sampling: temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
- Thinking sampling: temp 1.0, top_p 0.95, top_k 20.
- Images: sent as base64 data URLs in the OpenAI `image_url` content part.
