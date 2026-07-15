"""
Core data models for the LLM-as-judge pipeline.

Everything the judge produces or consumes flows through one of these
pydantic models, so a malformed suite file or a malformed judge response
fails loudly and early instead of silently corrupting a report.
"""
from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Rubric
# --------------------------------------------------------------------------

class RubricCriterion(BaseModel):
    name: str
    description: str
    weight: float = Field(gt=0)


class ScaleAnchor(BaseModel):
    score: float
    example: str


class Rubric(BaseModel):
    criteria: list[RubricCriterion]
    scale_min: float = 1
    scale_max: float = 5
    anchors: list[ScaleAnchor] = Field(default_factory=list)
    pass_threshold: float = 3.5

    @field_validator("criteria")
    @classmethod
    def _at_least_one_criterion(cls, v: list[RubricCriterion]) -> list[RubricCriterion]:
        if not v:
            raise ValueError("Rubric must define at least one criterion")
        return v

    def weights_normalized(self) -> dict[str, float]:
        total = sum(c.weight for c in self.criteria)
        return {c.name: c.weight / total for c in self.criteria}


# --------------------------------------------------------------------------
# Test cases (the `{input, system_prompt, model_output, expected_output?,
# criteria?}` schema from the brief, extended with the bookkeeping fields
# a real pipeline needs)
# --------------------------------------------------------------------------

class TestCase(BaseModel):
    __test__ = False  # not a pytest test class, just named TestCase per the brief's schema
    id: str = ""
    input: str
    system_prompt: str = ""
    system_prompt_a: Optional[str] = None  # overrides system_prompt for output_a, if prompts differ
    system_prompt_b: Optional[str] = None  # overrides system_prompt for output_b
    model_output: Optional[str] = None
    output_a: Optional[str] = None
    output_b: Optional[str] = None
    expected_output: Optional[str] = None
    criteria: Optional[list[RubricCriterion]] = None  # per-case rubric override

    generator_model: Optional[str] = None       # BYO metadata: what produced model_output
    generator_model_a: Optional[str] = None
    generator_model_b: Optional[str] = None

    gold_score: Optional[float] = None          # human label, for validation
    gold_label: Optional[Literal["pass", "fail"]] = None
    gold_winner: Optional[Literal["a", "b", "tie"]] = None

    tags: list[str] = Field(default_factory=list)  # e.g. ["adversarial", "verbose_wrong"]

    @field_validator("id", mode="before")
    @classmethod
    def _fill_id(cls, v, info):
        if v:
            return v
        # Stable deterministic id from content, so re-running a suite
        # (or comparing across the naive/hardened configs) refers to the
        # same case even without an explicit id in the YAML.
        data = info.data
        basis = (data.get("input") or "") + (data.get("system_prompt") or "")
        return "case_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]


class TestSuite(BaseModel):
    name: str
    description: str = ""
    rubric: Optional[Rubric] = None  # falls back to config/rubric.yaml if absent
    cases: list[TestCase]


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

class CriterionScore(BaseModel):
    criterion: str
    score: float
    rationale: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class JudgeMode(str, Enum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"


class Verdict(BaseModel):
    case_id: str
    mode: JudgeMode
    per_criterion: list[CriterionScore]
    overall_score: float
    overall_rationale: str
    passed: Optional[bool] = None            # pointwise only
    winner: Optional[Literal["a", "b", "tie"]] = None  # pairwise only
    flags: list[str] = Field(default_factory=list)

    judge_model: str = ""
    order: Optional[Literal["ab", "ba"]] = None  # which order this pairwise call used
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    raw_response: str = ""
    parse_retries: int = 0
    is_judge_error: bool = False
    timestamp: float = Field(default_factory=time.time)


class PairwiseResult(BaseModel):
    """Reconciled result of running one pair in BOTH orders."""
    case_id: str
    verdict_ab: Verdict   # order: output_a shown first
    verdict_ba: Verdict   # order: output_b shown first
    winner_ab: Optional[Literal["a", "b", "tie"]]
    winner_ba: Optional[Literal["a", "b", "tie"]]
    flipped: bool
    reconciled_winner: Literal["a", "b", "tie", "inconsistent"]


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

class SuiteReport(BaseModel):
    suite_name: str
    mode: JudgeMode
    n_cases: int
    n_errors: int = 0
    pass_rate: Optional[float] = None
    mean_overall_score: Optional[float] = None
    mean_per_criterion: dict[str, float] = Field(default_factory=dict)
    score_stddev: Optional[float] = None

    win_rate_a: Optional[float] = None
    win_rate_b: Optional[float] = None
    tie_rate: Optional[float] = None
    flip_rate: Optional[float] = None
    positional_first_win_rate: Optional[float] = None
    winner: Optional[str] = None
    winner_significant: Optional[bool] = None
    winner_p_value: Optional[float] = None

    length_score_correlation: Optional[float] = None
    family_mismatch_warning: Optional[str] = None

    total_tokens: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: Optional[float] = None

    verdicts: list[dict] = Field(default_factory=list)  # serialized Verdict/PairwiseResult
