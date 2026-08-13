"""Token-and-price model for estimating synthetic-generation cost.

Prices are USD per 1,000,000 tokens (input, output), standard (non-batch) tier,
as published July 2026. The Batch API halves both; prompt caching cuts repeated
input to ~10%. Verify against current pricing before budgeting a large run --
these move. See COST_BREAKDOWN.md for the derived tables and sources.
"""

from __future__ import annotations

from dataclasses import dataclass

# model -> (input $/1M, output $/1M)
PRICES = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (5.00, 15.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),  # no bare alias exists for 4.5
}

# Default per-record token assumptions (prompt carries the full ontology).
DEFAULT_INPUT_TOKENS = 1600
DEFAULT_OUTPUT_TOKENS = 1400


@dataclass
class CostEstimate:
    model: str
    count: int
    input_tokens: int
    output_tokens: int
    batch: bool

    @property
    def _price(self):
        return PRICES.get(self.model)

    @property
    def total_input(self) -> int:
        return self.count * self.input_tokens

    @property
    def total_output(self) -> int:
        return self.count * self.output_tokens

    @property
    def usd(self) -> float | None:
        p = self._price
        if p is None:
            return None
        factor = 0.5 if self.batch else 1.0
        return factor * (self.total_input / 1e6 * p[0] + self.total_output / 1e6 * p[1])


def estimate(model: str, count: int, input_tokens: int = DEFAULT_INPUT_TOKENS,
             output_tokens: int = DEFAULT_OUTPUT_TOKENS, batch: bool = False) -> CostEstimate:
    return CostEstimate(model, count, input_tokens, output_tokens, batch)
