"""
Validates the judge itself, three independent ways:
  1. Agreement with human gold labels (Cohen's kappa + Pearson/Spearman).
  2. Test-retest consistency (rerun N times, measure verdict flips).
  3. Adversarial probes (verbose-but-wrong vs terse-but-correct pairs
     purpose-built to trip the biases in bias_metrics.py).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from pydantic import BaseModel

from .judge import Judge
from .schema import TestCase



# 1. Gold-label agreement

def cohen_kappa(judge_labels: list[str], gold_labels: list[str]) -> Optional[float]:
    """Standard Cohen's kappa for two raters over categorical labels."""
    n = len(judge_labels)
    if n == 0 or n != len(gold_labels):
        return None
    categories = sorted(set(judge_labels) | set(gold_labels))
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = np.zeros((k, k))
    for j, g in zip(judge_labels, gold_labels):
        matrix[idx[j], idx[g]] += 1

    po = np.trace(matrix) / n
    row_marg = matrix.sum(axis=1) / n
    col_marg = matrix.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1 - pe))


def pearson_spearman(judge_scores: list[float], gold_scores: list[float]) -> dict[str, Optional[float]]:
    if len(judge_scores) < 3:
        return {"pearson": None, "spearman": None}
    pearson = float(np.corrcoef(judge_scores, gold_scores)[0, 1])

    def rank(vals: list[float]) -> np.ndarray:
        order = np.argsort(vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(vals))
        return ranks

    spearman = float(np.corrcoef(rank(judge_scores), rank(gold_scores))[0, 1])
    return {
        "pearson": pearson if pearson == pearson else None,
        "spearman": spearman if spearman == spearman else None,
    }


class GoldAgreementReport(BaseModel):
    n_cases: int
    kappa_pass_fail: Optional[float] = None
    pearson_score: Optional[float] = None
    spearman_score: Optional[float] = None
    raw_agreement_rate: Optional[float] = None


def evaluate_gold_agreement(judge: Judge, cases: list[TestCase]) -> GoldAgreementReport:
    labeled = [c for c in cases if c.gold_label is not None or c.gold_score is not None]
    judge_labels, gold_labels = [], []
    judge_scores, gold_scores = [], []
    for c in labeled:
        v = judge.judge_pointwise(c)
        if v.is_judge_error:
            continue
        if c.gold_label is not None:
            judge_labels.append("pass" if v.passed else "fail")
            gold_labels.append(c.gold_label)
        if c.gold_score is not None:
            judge_scores.append(v.overall_score)
            gold_scores.append(c.gold_score)

    kappa = cohen_kappa(judge_labels, gold_labels) if judge_labels else None
    raw_agree = (
        sum(1 for j, g in zip(judge_labels, gold_labels) if j == g) / len(judge_labels)
        if judge_labels else None
    )
    corr = pearson_spearman(judge_scores, gold_scores) if judge_scores else {"pearson": None, "spearman": None}

    return GoldAgreementReport(
        n_cases=len(labeled), kappa_pass_fail=kappa, raw_agreement_rate=raw_agree,
        pearson_score=corr["pearson"], spearman_score=corr["spearman"],
    )


# 2. Test-retest consistency

class TestRetestReport(BaseModel):
    n_cases: int
    n_reruns: int
    pass_fail_flip_rate: Optional[float] = None
    mean_score_stddev_within_case: Optional[float] = None


def evaluate_test_retest(judge: Judge, cases: list[TestCase], n_reruns: int = 3) -> TestRetestReport:
    if not cases:
        return TestRetestReport(n_cases=0, n_reruns=n_reruns)

    flips = 0
    stddevs = []
    for c in cases:
        verdicts = [judge.judge_pointwise(c) for _ in range(n_reruns)]
        verdicts = [v for v in verdicts if not v.is_judge_error]
        if len(verdicts) < 2:
            continue
        pass_labels = [v.passed for v in verdicts]
        if len(set(pass_labels)) > 1:
            flips += 1
        stddevs.append(float(np.std([v.overall_score for v in verdicts], ddof=1)))

    return TestRetestReport(
        n_cases=len(cases), n_reruns=n_reruns,
        pass_fail_flip_rate=flips / len(cases) if cases else None,
        mean_score_stddev_within_case=float(np.mean(stddevs)) if stddevs else None,
    )


# 3. Adversarial probes

class AdversarialProbeReport(BaseModel):
    n_pairs: int
    n_fooled: int
    fooled_rate: Optional[float] = None
    mean_margin: Optional[float] = None  
    details: list[dict] = []


def evaluate_adversarial_probes(judge: Judge, probe_pairs: list[dict]) -> AdversarialProbeReport:
    """
    probe_pairs: list of {"verbose_wrong": TestCase, "terse_correct": TestCase}
    sharing the same underlying question. The judge is "fooled" if it scores
    the verbose-but-wrong answer >= the terse-but-correct one.

    fooled_rate is a binary count and can stay flat even when the judge's
    behavior genuinely improved -- e.g. if a mitigation widens the margin by
    which terse_correct beats verbose_wrong on every pair that was already
    resistant, fooled_rate won't move but mean_margin will. Report both.
    """
    n_fooled = 0
    margins = []
    details = []
    for pair in probe_pairs:
        vw = judge.judge_pointwise(pair["verbose_wrong"])
        tc = judge.judge_pointwise(pair["terse_correct"])
        fooled = (not vw.is_judge_error and not tc.is_judge_error
                  and vw.overall_score >= tc.overall_score)
        if fooled:
            n_fooled += 1
        if not vw.is_judge_error and not tc.is_judge_error:
            margins.append(tc.overall_score - vw.overall_score)
        details.append({
            "case_id": pair["verbose_wrong"].id,
            "verbose_wrong_score": vw.overall_score,
            "terse_correct_score": tc.overall_score,
            "margin": (tc.overall_score - vw.overall_score) if not (vw.is_judge_error or tc.is_judge_error) else None,
            "fooled": fooled,
        })
    n = len(probe_pairs)
    return AdversarialProbeReport(
        n_pairs=n, n_fooled=n_fooled,
        fooled_rate=(n_fooled / n) if n else None,
        mean_margin=(sum(margins) / len(margins)) if margins else None,
        details=details,
    )
