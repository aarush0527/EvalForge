"""
Turns judge output into the numbers the bias table promises:
  - position bias -> flip rate + positional first-win-rate skew
  - verbosity bias -> length/score correlation
  - self-enhancement -> judge/generator family mismatch warning
  - score clustering -> score-distribution std-dev
(sycophancy/style is measured via the adversarial probe set in validation.py,
since it needs a specifically-designed probe case, not a suite-wide stat)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .schema import PairwiseResult, TestCase, Verdict


def flip_rate(results: list[PairwiseResult]) -> Optional[float]:
    usable = [r for r in results if not (r.verdict_ab.is_judge_error or r.verdict_ba.is_judge_error)]
    if not usable:
        return None
    return sum(1 for r in usable if r.flipped) / len(usable)


def positional_first_win_rate(results: list[PairwiseResult]) -> Optional[float]:
    """
    Across all individual order-calls (2 per case), what fraction did the
    FIRST-SHOWN side win? 50% is what an unbiased judge should produce;
    higher indicates the judge favors whichever answer it sees first.
    Ties are excluded from the denominator (they carry no positional signal).
    """
    first_wins = 0
    decided = 0
    for r in results:
        for v in (r.verdict_ab, r.verdict_ba):
            if v.is_judge_error or v.winner is None or v.winner == "tie":
                continue
            decided += 1
            first_side = "a" if v.order == "ab" else "b"
            if v.winner == first_side:
                first_wins += 1
    if decided == 0:
        return None
    return first_wins / decided


def win_rates(results: list[PairwiseResult]) -> dict[str, float]:
    usable = [r for r in results if r.reconciled_winner in ("a", "b", "tie")]
    n = len(usable)
    if n == 0:
        return {"win_rate_a": 0.0, "win_rate_b": 0.0, "tie_rate": 0.0, "inconsistent_rate": 1.0}
    a = sum(1 for r in usable if r.reconciled_winner == "a")
    b = sum(1 for r in usable if r.reconciled_winner == "b")
    t = sum(1 for r in usable if r.reconciled_winner == "tie")
    inconsistent = sum(1 for r in results if r.reconciled_winner == "inconsistent")
    total = len(results)
    return {
        "win_rate_a": a / n, "win_rate_b": b / n, "tie_rate": t / n,
        "inconsistent_rate": inconsistent / total if total else 0.0,
    }


def length_score_correlation(cases: list[TestCase], verdicts: list[Verdict]) -> Optional[float]:
    by_id = {c.id: c for c in cases}
    lengths, scores = [], []
    for v in verdicts:
        if v.is_judge_error:
            continue
        case = by_id.get(v.case_id)
        if case is None or case.model_output is None:
            continue
        lengths.append(len(case.model_output.split()))
        scores.append(v.overall_score)
    if len(lengths) < 3 or len(set(lengths)) < 2:
        return None
    corr = float(np.corrcoef(lengths, scores)[0, 1])
    return corr if corr == corr else None  # filter NaN


def score_stddev(verdicts: list[Verdict]) -> Optional[float]:
    scores = [v.overall_score for v in verdicts if not v.is_judge_error]
    if len(scores) < 2:
        return None
    return float(np.std(scores, ddof=1))


def family_mismatch_warning(judge_family: str, generator_family: Optional[str]) -> Optional[str]:
    if not generator_family:
        return None
    if judge_family == generator_family:
        return (
            f"Judge and generator are both from the '{judge_family}' family. "
            "Self-enhancement bias risk: this judge may systematically favor this "
            "generator's style. Consider a cross-family judge or an ensemble."
        )
    return None


def binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided sign-test p-value using math.comb -- no scipy dependency."""
    import math
    if n == 0:
        return 1.0
    k = min(k, n - k) if k > n / 2 else k
    # sum P(X <= min(k, n-k)) over both tails
    lo = min(k, n - k)
    cum = sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(0, lo + 1))
    return min(1.0, 2 * cum)
