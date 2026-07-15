"""
Robust parsing of the judge's structured verdict.

Order of attempts:
  1. json.loads on the raw text.
  2. Strip markdown code fences / leading-trailing prose, then json.loads.
  3. Regex-extract the outermost {...} block, then json.loads.
  4. Re-prompt the judge ONCE with the parse error, asking it to return
     corrected JSON only.
  5. Give up: caller logs the case as a judge error rather than crashing
     the whole suite.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional


class ParseFailure(Exception):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class ParseResult:
    data: Optional[dict]
    retries: int
    raw_response: str
    ok: bool
    error: Optional[str] = None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)

REQUIRED_KEYS_POINTWISE = {"per_criterion", "overall_score", "overall_rationale"}
REQUIRED_KEYS_PAIRWISE = {"per_criterion", "overall_score", "overall_rationale", "winner"}


def _try_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_json(text: str) -> Optional[dict]:
    text = text.strip()

    data = _try_json(text)
    if data is not None:
        return data

    fence = _FENCE_RE.search(text)
    if fence:
        data = _try_json(fence.group(1))
        if data is not None:
            return data

    brace = _BRACE_RE.search(text)
    if brace:
        data = _try_json(brace.group(0))
        if data is not None:
            return data

    return None


def validate_shape(data: dict, required_keys: set[str]) -> Optional[str]:
    missing = required_keys - data.keys()
    if missing:
        return f"Missing required key(s): {sorted(missing)}"
    if not isinstance(data.get("per_criterion"), list) or not data["per_criterion"]:
        return "per_criterion must be a non-empty list"
    for item in data["per_criterion"]:
        if not isinstance(item, dict) or not {"criterion", "score", "rationale"} <= item.keys():
            return "each per_criterion entry needs criterion, score, rationale"
    return None


def parse_verdict(
    raw_text: str,
    *,
    pairwise: bool,
    retry_fn: Optional[Callable[[str], str]] = None,
    max_retries: int = 1,
) -> ParseResult:
    """
    retry_fn(error_message) -> new_raw_text. If given, called on parse
    failure to ask the judge to fix its own output. If not given (or it
    raises), parsing just fails after the local extraction attempts.
    """
    required = REQUIRED_KEYS_PAIRWISE if pairwise else REQUIRED_KEYS_POINTWISE
    retries = 0
    text = raw_text

    while True:
        data = extract_json(text)
        error = None if data is None else validate_shape(data, required)
        if data is not None and error is None:
            return ParseResult(data=data, retries=retries, raw_response=text, ok=True)

        error = error or "Could not locate a valid JSON object in the response"
        if retries >= max_retries or retry_fn is None:
            return ParseResult(data=None, retries=retries, raw_response=text, ok=False, error=error)

        retries += 1
        try:
            text = retry_fn(error)
        except Exception as e:  # noqa: BLE001 -- any retry failure just ends the loop
            return ParseResult(data=None, retries=retries, raw_response=text, ok=False, error=str(e))
