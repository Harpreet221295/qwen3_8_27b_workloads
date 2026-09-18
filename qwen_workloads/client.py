"""Thin wrapper over the OpenAI SDK pointed at the vLLM endpoint."""
from __future__ import annotations
import os, time
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
NOTHINK_SAMPLING = dict(temperature=0.7, top_p=0.8, extra_body={"top_k": 20, "presence_penalty": 1.5, **THINK_OFF})
THINK_SAMPLING = dict(temperature=1.0, top_p=0.95, extra_body={"top_k": 20})


def think_effort(level: str) -> dict:
    """level in {'low','medium','xhigh'} -> extra_body dict."""
    return {"chat_template_kwargs": {"reasoning_effort": level}}


@dataclass
class Result:
    content: str
    reasoning: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    raw: Any = None


class QwenClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = base_url or os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "none")
        self.model = model or os.environ.get("QWEN_MODEL", "qwen3.8-27b")
        self.sync = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600)
        self.async_ = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600)

    # ---- core ----
    def chat(self, messages: list[dict], *, thinking: bool | str = False, max_tokens: int = 1024,
             tools: list | None = None, tool_choice: str | None = None, response_format: dict | None = None,
             **overrides) -> Result:
        kw = self._kwargs(thinking, max_tokens, tools, tool_choice, response_format, overrides)
        t = time.perf_counter()
        r = self.sync.chat.completions.create(model=self.model, messages=messages, **kw)
        return self._to_result(r, time.perf_counter() - t)

    async def achat(self, messages: list[dict], *, thinking: bool | str = False, max_tokens: int = 1024,
                    tools: list | None = None, tool_choice: str | None = None, response_format: dict | None = None,
                    **overrides) -> Result:
        kw = self._kwargs(thinking, max_tokens, tools, tool_choice, response_format, overrides)
        t = time.perf_counter()
        r = await self.async_.chat.completions.create(model=self.model, messages=messages, **kw)
        return self._to_result(r, time.perf_counter() - t)

    def ask(self, prompt: str, system: str | None = None, **kw) -> Result:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw)

    def ask_image(self, prompt: str, images: list[str], **kw) -> Result:
        """images: list of data URLs or http(s) URLs."""
        parts = [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": prompt}]
        return self.chat([{"role": "user", "content": parts}], **kw)

    def stream(self, messages: list[dict], *, thinking: bool | str = False, max_tokens: int = 512, **overrides):
        """Yields (kind, text) with kind in {'reasoning','content'}; final item ('done', stats)."""
        kw = self._kwargs(thinking, max_tokens, None, None, None, overrides)
        t0 = time.perf_counter(); first = None; n = 0
        for chunk in self.sync.chat.completions.create(model=self.model, messages=messages, stream=True,
                                                       stream_options={"include_usage": True}, **kw):
            if chunk.choices:
                d = chunk.choices[0].delta
                rc = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                if rc:
                    if first is None: first = time.perf_counter() - t0
                    n += 1; yield ("reasoning", rc)
                if d.content:
                    if first is None: first = time.perf_counter() - t0
                    n += 1; yield ("content", d.content)
        yield ("done", {"ttft_s": first, "total_s": time.perf_counter() - t0, "chunks": n})

    # ---- helpers ----
    def _kwargs(self, thinking, max_tokens, tools, tool_choice, response_format, overrides):
        if thinking is False:
            kw = {k: v for k, v in NOTHINK_SAMPLING.items()}
            kw["extra_body"] = dict(NOTHINK_SAMPLING["extra_body"])
        else:
            kw = {k: v for k, v in THINK_SAMPLING.items()}
            kw["extra_body"] = dict(THINK_SAMPLING["extra_body"])
            if isinstance(thinking, str):
                kw["extra_body"].update(think_effort(thinking))
        kw["max_tokens"] = max_tokens
        if tools: kw["tools"] = tools; kw["tool_choice"] = tool_choice or "auto"
        if response_format: kw["response_format"] = response_format
        eb = overrides.pop("extra_body", None)
        if eb: kw["extra_body"].update(eb)
        kw.update(overrides)
        return kw

    @staticmethod
    def _to_result(r, latency) -> Result:
        m = r.choices[0].message
        tcs = [{"id": t.id, "name": t.function.name, "arguments": t.function.arguments} for t in (m.tool_calls or [])]
        u = r.usage
        return Result(content=m.content or "", reasoning=getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or "",
                      latency_s=latency, prompt_tokens=u.prompt_tokens if u else 0,
                      completion_tokens=u.completion_tokens if u else 0, tool_calls=tcs,
                      finish_reason=r.choices[0].finish_reason, raw=r)
