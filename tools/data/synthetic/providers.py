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
    provider: str = "openai"          # openai | anthropic | mock
    model: str = "gpt-4o"
    temperature: float = 0.9
    max_tokens: int = 2048


def build_provider(cfg: ProviderConfig) -> "LLMProvider":
    """Instantiate the provider named in ``cfg`` (case-insensitive)."""
    name = cfg.provider.lower()
    if name == "openai":
        return OpenAIProvider(cfg)
    if name == "anthropic":
        return AnthropicProvider(cfg)
    if name == "mock":
        return MockProvider(cfg)
    raise ValueError(f"unknown provider {cfg.provider!r} (use openai|anthropic|mock)")


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
    """Chat Completions with JSON-object response format (GPT-4o / GPT-4.1 / mini)."""

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai SDK missing -- run: uv add openai") from e
        self._client = OpenAI()

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


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
        printed up front so a killed run can be recovered from the Hub.
        """
        import time
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
        while True:
            b = self._client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            c = b.request_counts
            print(f"[batch] {b.processing_status}: processing={c.processing} "
                  f"succeeded={c.succeeded} errored={c.errored}")
            time.sleep(30)

        out: dict = {}
        errored = 0
        for result in self._client.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                msg = result.result.message
                out[result.custom_id] = "".join(
                    bl.text for bl in msg.content if bl.type == "text")
            else:
                errored += 1
        print(f"[batch] {batch.id} ended: {len(out)} succeeded, {errored} errored/expired")
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
