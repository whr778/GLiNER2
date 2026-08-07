"""Configurable LLM providers for synthetic generation.

One thin interface, ``LLMProvider.complete(system, user) -> str``, over the
OpenAI and Anthropic SDKs plus a keyless ``MockProvider`` for dry runs and
tests. The model, temperature, and token cap are all set from config, so the
calling model is fully swappable without touching the generator.

API keys are read from the environment only (``OPENAI_API_KEY`` /
``ANTHROPIC_API_KEY``); they are never written to disk or logged. The SDKs are
imported lazily so the module loads (and dry runs work) without them installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str = "openai"          # openai|anthropic|mock|vllm|ollama|mlx|ollama_native
    model: str = "gpt-4o"
    temperature: float = 0.9
    max_tokens: int = 2048
    base_url: str = ""                # OpenAI-compatible endpoint (local backends)
    json_object: bool = True          # request response_format=json_object;
                                      # disable for servers that reject it (some MLX builds)
    think: bool = False               # ollama_native only: enable model "thinking";
                                      # default off so reasoning models emit the answer


# OpenAI-compatible local servers reachable through the OpenAI SDK by pointing
# base_url at them: (base-url env override, default local URL, key env). Local
# servers accept any token, so a dedicated *_API_KEY env is used (default "EMPTY")
# and the real OPENAI_API_KEY is NEVER sent to a local endpoint.
LOCAL_OPENAI_BACKENDS = {
    "vllm":   ("VLLM_BASE_URL",   "http://localhost:8000/v1",  "VLLM_API_KEY"),
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434/v1", "OLLAMA_API_KEY"),
    "mlx":    ("MLX_BASE_URL",    "http://localhost:8080/v1",  "MLX_API_KEY"),
}


def build_provider(cfg: ProviderConfig) -> "LLMProvider":
    """Instantiate the provider named in ``cfg`` (case-insensitive)."""
    name = cfg.provider.lower()
    if name == "openai":
        return OpenAIProvider(cfg)
    if name == "anthropic":
        return AnthropicProvider(cfg)
    if name == "mock":
        return MockProvider(cfg)
    if name in LOCAL_OPENAI_BACKENDS:
        env_url, default_url, env_key = LOCAL_OPENAI_BACKENDS[name]
        base_url = cfg.base_url or os.environ.get(env_url) or default_url
        api_key = os.environ.get(env_key) or "EMPTY"
        return OpenAIProvider(cfg, base_url=base_url, require_key=False, api_key=api_key)
    if name in ("ollama_native", "ollama-native"):
        base = (cfg.base_url or os.environ.get("OLLAMA_BASE_URL")
                or os.environ.get("OLLAMA_HOST") or "http://localhost:11434")
        return OllamaNativeProvider(cfg, base_url=base)
    raise ValueError(
        f"unknown provider {cfg.provider!r} "
        f"(use openai|anthropic|mock|vllm|ollama|mlx|ollama_native)")


class LLMProvider:
    """Base interface. ``complete`` returns the model's raw text response."""

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def complete_batch(self, items):
        """Batch variant: items is a list of (custom_id, system, user).

        Returns {custom_id: raw_text} for the requests that succeeded. Providers
        without a batch API don't override this.
        """
        raise NotImplementedError(f"{type(self).__name__} has no batch mode")


class OpenAIProvider(LLMProvider):
    """Chat Completions with JSON-object response format.

    Drives the OpenAI API and any OpenAI-compatible local server (vLLM, Ollama,
    ``mlx_lm.server``). Pass ``base_url`` to target a local endpoint and
    ``require_key=False`` so a real ``OPENAI_API_KEY`` is optional (local servers
    accept any token; ``EMPTY`` is used when none is set).
    """

    def __init__(self, cfg: ProviderConfig, *, base_url: str | None = None,
                 require_key: bool = True, api_key: str | None = None) -> None:
        super().__init__(cfg)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai SDK missing -- run: uv add openai") from e
        url = base_url or cfg.base_url or None
        if require_key and not url:
            # Real OpenAI: needs a key (the SDK still honors OPENAI_BASE_URL if set).
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
            self._client = OpenAI()
        else:
            # OpenAI-compatible endpoint: use the explicit key when given (local
            # backends pass "EMPTY"); else fall back to OPENAI_API_KEY for a custom
            # cloud-compatible URL.
            self._client = OpenAI(
                base_url=url,
                api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY")

    def complete(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self.cfg.json_object:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class OllamaNativeProvider(LLMProvider):
    """Ollama's NATIVE /api/chat endpoint (not the OpenAI-compatible /v1 path).

    Exists so reasoning models can run with **thinking disabled** (`think: false`):
    the /v1 path ignores that toggle, so a reasoning model there burns its whole
    token budget on hidden reasoning and returns empty content. Uses Ollama's native
    JSON mode (`format: "json"`) when cfg.json_object. Stdlib HTTP -- no extra deps.
    """

    def __init__(self, cfg: ProviderConfig, *,
                 base_url: str = "http://localhost:11434") -> None:
        super().__init__(cfg)
        root = base_url.rstrip("/")
        if root.endswith("/v1"):          # tolerate an OpenAI-style base_url
            root = root[:-3].rstrip("/")
        self._url = root + "/api/chat"

    def complete(self, system: str, user: str) -> str:
        import json
        import urllib.request
        body = {
            "model": self.cfg.model,
            "stream": False,
            "think": self.cfg.think,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }
        if self.cfg.json_object:
            body["format"] = "json"
        req = urllib.request.Request(
            self._url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return (resp.get("message") or {}).get("content") or ""


class AnthropicProvider(LLMProvider):
    """Messages API; the system prompt requires a single JSON object reply."""

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError("anthropic SDK missing -- run: uv add anthropic") from e
        self._client = Anthropic()

    def _params(self, system: str, user: str) -> dict:
        # Current Claude models (Sonnet 5, Opus 4.7/4.8) reject temperature/top_p
        # with a 400; thinking is disabled so the reply is a single JSON pass that
        # stays within max_tokens and keeps per-doc cost predictable for bulk runs.
        return dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(**self._params(system, user))
        return "".join(block.text for block in resp.content if block.type == "text")

    def complete_batch(self, items):
        """Submit all requests to the Message Batches API (-50% pricing), poll
        until the batch ends, then return {custom_id: raw_text} for successes.

        Runs asynchronously on Anthropic's side (usually < 1h). The batch id is
        printed up front so a killed run can be recovered via ``fetch_batch``.
        """
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = [
            Request(custom_id=cid,
                    params=MessageCreateParamsNonStreaming(**self._params(system, user)))
            for cid, system, user in items
        ]
        batch = self._client.messages.batches.create(requests=requests)
        print(f"[batch] submitted {len(requests)} requests as {batch.id} "
              f"(-50% pricing); polling every 30s...")
        return self.fetch_batch(batch.id)

    def fetch_batch(self, batch_id):
        """Poll an already-submitted batch to completion and collect its results.
        Recovers a batch whose original poll loop died (network timeout) -- no
        resubmission, no extra spend. Transient poll errors are retried, not fatal.
        """
        import time
        from anthropic import APIConnectionError, APITimeoutError

        while True:
            try:
                b = self._client.messages.batches.retrieve(batch_id)
            except (APITimeoutError, APIConnectionError) as e:
                print(f"[batch] transient poll error, retrying: {e}")
                time.sleep(15); continue
            if b.processing_status == "ended":
                break
            c = b.request_counts
            print(f"[batch] {b.processing_status}: processing={c.processing} "
                  f"succeeded={c.succeeded} errored={c.errored}")
            time.sleep(30)

        out: dict = {}
        errored = 0
        for result in self._client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                msg = result.result.message
                out[result.custom_id] = "".join(
                    bl.text for bl in msg.content if bl.type == "text")
            else:
                errored += 1
        print(f"[batch] {batch_id} ended: {len(out)} succeeded, {errored} errored/expired")
        return out


class MockProvider(LLMProvider):
    """Keyless deterministic provider for dry runs and tests.

    Returns one small hand-built record whose spans are verbatim substrings of
    its own text, so the whole generate -> validate -> write path can run with
    no API access and no spend.
    """

    def complete_batch(self, items):
        """Deterministic batch stand-in for dry runs -- no API, no spend."""
        return {cid: self.complete(system, user) for cid, system, user in items}

    def complete(self, system: str, user: str) -> str:
        return (
            '{"text": "Acme Corp acquired Beta Inc for $2 billion on Monday in Boston. '
            'CEO Jane Doe led the deal.",'
            ' "entities": [{"type": "organization", "text": "Acme Corp"},'
            ' {"type": "organization", "text": "Beta Inc"},'
            ' {"type": "money", "text": "$2 billion"},'
            ' {"type": "geopolitical entity", "text": "Boston"},'
            ' {"type": "person", "text": "Jane Doe"},'
            ' {"type": "job title", "text": "CEO"}],'
            ' "relations": [{"type": "acquired", "head": "Acme Corp", "tail": "Beta Inc"}],'
            ' "events": [{"event_type": "Transaction.TransferOwnership",'
            ' "trigger": "acquired",'
            ' "arguments": [{"role": "Buyer", "entity": "Acme Corp"},'
            ' {"role": "Seller", "entity": "Beta Inc"},'
            ' {"role": "Price", "entity": "$2 billion"}]}],'
            ' "classifications": [{"task": "topic", "labels": ["business"]},'
            ' {"task": "sentiment", "labels": ["neutral"]}],'
            ' "structures": [{"type": "transaction",'
            ' "fields": {"item": "Beta Inc", "amount": "$2 billion",'
            ' "buyer": "Acme Corp", "seller": "Beta Inc"}}]}'
        )
