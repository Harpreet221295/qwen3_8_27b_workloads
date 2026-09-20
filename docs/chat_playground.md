# Chat playground (port 7860)

A single-page chat UI over the pod. Start: `.venv/bin/python -m uvicorn ui.app:app --port 7860` (or `../start_uis.sh`).
Backend `ui/app.py` (FastAPI) proxies to vLLM so the API key never reaches the browser; page `ui/static/index.html`.

## What it does
- **Streams** the answer and, separately, the model's **thinking** (gold panel) as they are generated. Reasoning tokens
  arrive on `delta.reasoning`; content on `delta.content`.
- **Thinking selector**: off / low / medium / xhigh. Sampling switches automatically (off: 0.7/0.8/top_k 20/presence 1.5;
  on: 1.0/0.95/top_k 20). Override temperature if you like.
- **Images**: attach a PNG/JPG; sent as a base64 `image_url` part. One image per message (server default).
- **Demo tools** checkbox: `get_weather`, `convert_currency`, `calculator` with fake local implementations; the server runs a
  tool loop (up to 6 hops) and shows each call and result inline. Good for feeling how the model decides to call tools.
- **System prompt**, **max tokens**, **Stop** button (aborts the stream).
- **Stats** under each reply: total time, time to first token, output tokens, tok/s, input tokens.
- **Math**: LaTeX in `\[ \]`, `\( \)`, `$$`, `$` is rendered with KaTeX (pulled out before markdown so underscores survive).
- **Workload results** tab: browses `results/*.jsonl` from `run_all.py`.

## What it does not do (yet)
- No persistence: conversation lives in the tab; refresh = gone. (Chat threads saved to disk are a planned addition.)
- No multi-image per message until `--limit-mm-per-prompt` is set on the pod.
- Web search / prompt caching / anything server-side beyond vLLM.

## Observed behaviour
- With thinking on, trivial prompts ("hello") produce little or no reasoning: the effort is adaptive.
- Single-stream speed ≈ 30 tok/s on the A100; TTFT ≈ 0.2–0.5 s.
- xhigh can spend thousands of tokens thinking on an easy question; use low/medium for daily use.
