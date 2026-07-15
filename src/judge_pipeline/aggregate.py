"""Aggregate per-case verdicts into a SuiteReport."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .bias_metrics import (
    binomial_two_sided_p,
    family_mismatch_warning,
    flip_rate,
    length_score_correlation,
    positional_first_win_rate,
    score_stddev,
    win_rates,
)
from .cost import estimate_cost as _price_lookup
from .schema import JudgeMode, PairwiseResult, TestCase, TokenUsage, Verdict, SuiteReport


def _sum_tokens(verdicts: list[Verdict]) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(v.tokens.input_tokens for v in verdicts),
        output_tokens=sum(v.tokens.output_tokens for v in verdicts),
    )


def aggregate_pointwise(
    suite_name: str, cases: list[TestCase], verdicts: list[Verdict],
    *, judge_family: str = "", generator_family: Optional[str] = None,
    price_table: Optional[dict] = None,
) -> SuiteReport:
    ok = [v for v in verdicts if not v.is_judge_error]
    n_errors = len(verdicts) - len(ok)

    pass_rate = (sum(1 for v in ok if v.passed) / len(ok)) if ok else None
    mean_overall = float(np.mean([v.overall_score for v in ok])) if ok else None

    per_crit: dict[str, list[float]] = {}
    for v in ok:
        for pc in v.per_criterion:
            per_crit.setdefault(pc.criterion, []).append(pc.score)
    mean_per_criterion = {k: float(np.mean(vs)) for k, vs in per_crit.items()}

    tokens = _sum_tokens(verdicts)
    cost = estimate_cost(tokens, judge_model=verdicts[0].judge_model if verdicts else "", price_table=price_table)

    return SuiteReport(
        suite_name=suite_name, mode=JudgeMode.POINTWISE, n_cases=len(cases), n_errors=n_errors,
        pass_rate=pass_rate, mean_overall_score=mean_overall, mean_per_criterion=mean_per_criterion,
        score_stddev=score_stddev(ok),
        length_score_correlation=length_score_correlation(cases, ok),
        family_mismatch_warning=family_mismatch_warning(judge_family, generator_family),
        total_tokens=tokens, estimated_cost_usd=cost,
        verdicts=[v.model_dump(mode="json") for v in verdicts],
    )


def aggregate_pairwise(
    suite_name: str, cases: list[TestCase], results: list[PairwiseResult],
    *, judge_family: str = "", generator_family_a: Optional[str] = None,
    generator_family_b: Optional[str] = None, price_table: Optional[dict] = None,
    significance_alpha: float = 0.10,
) -> SuiteReport:
    all_verdicts = [r.verdict_ab for r in results] + [r.verdict_ba for r in results]
    n_errors = sum(1 for v in all_verdicts if v.is_judge_error)

    rates = win_rates(results)
    fr = flip_rate(results)
    pfw = positional_first_win_rate(results)

    decided = [r for r in results if r.reconciled_winner in ("a", "b")]
    winner_p = None
    winner = "tie"
    significant = False
    if decided:
        a_wins = sum(1 for r in decided if r.reconciled_winner == "a")
        n = len(decided)
        winner_p = binomial_two_sided_p(a_wins, n)
        significant = winner_p < significance_alpha
        if rates["win_rate_a"] > rates["win_rate_b"]:
            winner = "a" if significant else "tie (a ahead, not significant)"
        elif rates["win_rate_b"] > rates["win_rate_a"]:
            winner = "b" if significant else "tie (b ahead, not significant)"
        else:
            winner = "tie"

    fam_warning = None
    fw_a = family_mismatch_warning(judge_family, generator_family_a)
    fw_b = family_mismatch_warning(judge_family, generator_family_b)
    if fw_a or fw_b:
        fam_warning = " | ".join(x for x in (fw_a, fw_b) if x)

    ok_verdicts = [v for v in all_verdicts if not v.is_judge_error]
    tokens = _sum_tokens(all_verdicts)
    cost = estimate_cost(tokens, judge_model=all_verdicts[0].judge_model if all_verdicts else "",
                          price_table=price_table)

    return SuiteReport(
        suite_name=suite_name, mode=JudgeMode.PAIRWISE, n_cases=len(cases), n_errors=n_errors,
        score_stddev=score_stddev(ok_verdicts),
        win_rate_a=rates["win_rate_a"], win_rate_b=rates["win_rate_b"], tie_rate=rates["tie_rate"],
        flip_rate=fr, positional_first_win_rate=pfw,
        winner=winner, winner_significant=significant, winner_p_value=winner_p,
        family_mismatch_warning=fam_warning,
        total_tokens=tokens, estimated_cost_usd=cost,
        verdicts=[r.model_dump(mode="json") for r in results],
    )


def estimate_cost(tokens: TokenUsage, judge_model: str, price_table: Optional[dict]) -> Optional[float]:
    if not judge_model:
        return None
    # judge_model is like "anthropic:claude-opus-4-8" or "simulated:simulated-judge-v1"
    key = judge_model.split(":", 1)[-1]
    return _price_lookup(tokens, key, price_table)
