"""
Token counts -> estimated USD cost, via a small editable price table
(config/pricing.yaml) rather than hardcoded rates, since provider pricing
changes over time and this file is the only place that should need editing.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .schema import TokenUsage

# Sourced from Anthropic's, OpenAI's, and Groq's public pricing pages (checked 2026-07-15).
# Keep this as the fallback default; config/pricing.yaml overrides it.
# Groq rates verified against groq.com/pricing and console.groq.com/docs/models;
# double-check there before relying on this for a real budget, since GroqCloud
# has occasionally adjusted per-model rates.
DEFAULT_PRICE_TABLE = {
    # Groq (USD per 1M tokens) -- this project's primary, free-tier-friendly provider.
    # GroqCloud also offers a genuine no-credit-card free tier (rate-limited, not
    # token-limited to zero), so a modest evaluation run can cost literally $0.
    "llama-3.1-8b-instant": {"input_per_mtok": 0.05, "output_per_mtok": 0.08},
    "llama-3.3-70b-versatile": {"input_per_mtok": 0.59, "output_per_mtok": 0.79},
    "openai/gpt-oss-20b": {"input_per_mtok": 0.075, "output_per_mtok": 0.30},
    "openai/gpt-oss-120b": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
    "moonshotai/kimi-k2-instruct": {"input_per_mtok": 1.00, "output_per_mtok": 3.00},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input_per_mtok": 0.50, "output_per_mtok": 0.77},  # groq.com/newsroom launch price
    "qwen/qwen3-32b": {"input_per_mtok": 0.29, "output_per_mtok": 0.59},  # listed as preview pricing, re-verify at groq.com/pricing
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input_per_mtok": 0.11, "output_per_mtok": 0.34},  # preview pricing, re-verify
    # gemma2-9b-it: no confirmed public rate found -- omitted rather than guessed.
    # cost estimation returns None for any model not in this table, so this
    # is a safe omission, not a silent wrong number.

    # Anthropic (USD per 1M tokens) -- optional, not required by this project.
    "claude-haiku-4-5-20251001": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    "claude-sonnet-5": {"input_per_mtok": 2.00, "output_per_mtok": 10.00},  # introductory, through 2026-08-31
    "claude-opus-4-8": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
    "claude-fable-5": {"input_per_mtok": 10.00, "output_per_mtok": 50.00},
    # OpenAI (USD per 1M tokens) -- optional, not required by this project.
    "gpt-5.4": {"input_per_mtok": 2.50, "output_per_mtok": 15.00},
    "gpt-5.4-mini": {"input_per_mtok": 0.40, "output_per_mtok": 1.60},
    "gpt-5.5": {"input_per_mtok": 5.00, "output_per_mtok": 30.00},
    # Simulated provider costs nothing -- it's a local stand-in, not a paid API.
    "simulated-judge-v1": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
    "simulated-judge-alpha-family": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
    "simulated-judge-beta-family": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
    "simulated-judge-gamma-family": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
}


def load_price_table(path: str | Path | None) -> dict:
    if path is None:
        return DEFAULT_PRICE_TABLE
    p = Path(path)
    if not p.exists():
        return DEFAULT_PRICE_TABLE
    data = yaml.safe_load(p.read_text()) or {}
    merged = dict(DEFAULT_PRICE_TABLE)
    merged.update(data)
    return merged


def estimate_cost(tokens: TokenUsage, model_key: str, price_table: dict | None = None) -> float | None:
    table = price_table if price_table is not None else DEFAULT_PRICE_TABLE
    rates = table.get(model_key)
    if not rates:
        return None
    return (tokens.input_tokens / 1_000_000) * rates["input_per_mtok"] + \
           (tokens.output_tokens / 1_000_000) * rates["output_per_mtok"]
