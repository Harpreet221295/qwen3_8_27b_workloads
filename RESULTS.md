# Results — 2026-09-19, first run

Endpoint: Qwen/Qwen3.8-27B BF16 on 1× A100 SXM 80GB (RunPod), vLLM latest, max-model-len 131072, KV cache ≈ 304K tokens.

| workload | passed/scored | avg latency (s) | notes |
|---|---|---|---|
| chat_basic | 5/5 | 1.2 | refusal case is manual-check only |
| thinking_modes | 16/16 | 3.0 | off / low / medium / xhigh all correct on 4 puzzles; xhigh used up to 221 tokens |
| multi_turn | 1/1 | 1.5 | state carried across 6 turns incl. an overwrite |
| vision_describe | 4/4 | 0.7 | shape counts exact on 3 synthetic images |
| vision_ocr | 3/3 | 2.1 | transcription similarity 1.00, JSON extraction correct |
| vision_chart | 4/4 | 3.3 | all 6 bar values read exactly |
| vision_multi_image | 1/2 | 1.2 | 2-chart comparison: model explained chart one first and ran out of tokens (prompt tightened after this run) |
| tool_calling | 5/5 | 2.7 | single, parallel, no-tool, multi-step loop, thinking+tools |
| structured_output | 5/5 | 1.4 | json_object, json_schema, choice, regex, json in thinking mode |
| coding_snippets | 4/5 | 10.8 | lru_cache failed once on a missing attribute (passed in an earlier run: sampling variance at temp 0.7) |
| long_context | 2/2 | 5.8 | needle found at 8K and 32K |
| streaming | 2/2 | 5.1 | TTFT ≈ 0.2 s |

Grader fixes made after the first pass (model was right, grader was strict): JSON-formatted shape counts, full month names in chart table, zero counts in parse_log, and the newer vLLM `structured_outputs` parameter replacing `guided_choice` / `guided_regex`.
