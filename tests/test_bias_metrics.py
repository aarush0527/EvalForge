import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge_pipeline.bias_metrics import (
    binomial_two_sided_p,
    family_mismatch_warning,
    flip_rate,
    positional_first_win_rate,
)
from judge_pipeline.judge import ensemble_reconcile
from judge_pipeline.schema import JudgeMode, PairwiseResult, TokenUsage, Verdict
from judge_pipeline.validation import cohen_kappa, pearson_spearman


def _v(order, winner):
    return Verdict(
        case_id="c", mode=JudgeMode.PAIRWISE, per_criterion=[], overall_score=3.0,
        overall_rationale="", winner=winner, order=order, judge_model="test",
        tokens=TokenUsage(),
    )


def test_flip_rate_all_consistent():
    results = [
        PairwiseResult(case_id="1", verdict_ab=_v("ab", "a"), verdict_ba=_v("ba", "a"),
                        winner_ab="a", winner_ba="a", flipped=False, reconciled_winner="a"),
        PairwiseResult(case_id="2", verdict_ab=_v("ab", "b"), verdict_ba=_v("ba", "b"),
                        winner_ab="b", winner_ba="b", flipped=False, reconciled_winner="b"),
    ]
    assert flip_rate(results) == 0.0


def test_flip_rate_all_flipped():
    results = [
        PairwiseResult(case_id="1", verdict_ab=_v("ab", "a"), verdict_ba=_v("ba", "b"),
                        winner_ab="a", winner_ba="b", flipped=True, reconciled_winner="inconsistent"),
    ]
    assert flip_rate(results) == 1.0


def test_positional_first_win_rate_unbiased():
    # order ab: first=a; order ba: first=b. If "a" always wins regardless of
    # order, that's NOT positional -- half of first-shown wins, half don't.
    results = [
        PairwiseResult(case_id="1", verdict_ab=_v("ab", "a"), verdict_ba=_v("ba", "a"),
                        winner_ab="a", winner_ba="a", flipped=False, reconciled_winner="a"),
    ]
    # order ab: first=a, winner=a -> first won. order ba: first=b, winner=a -> first did NOT win.
    assert positional_first_win_rate(results) == 0.5


def test_positional_first_win_rate_fully_biased():
    # Winner is always whichever side was shown first.
    results = [
        PairwiseResult(case_id="1", verdict_ab=_v("ab", "a"), verdict_ba=_v("ba", "b"),
                        winner_ab="a", winner_ba="b", flipped=True, reconciled_winner="inconsistent"),
    ]
    assert positional_first_win_rate(results) == 1.0


def test_binomial_two_sided_p_symmetric_is_one():
    assert binomial_two_sided_p(5, 10) == 1.0


def test_binomial_two_sided_p_extreme_is_small():
    assert binomial_two_sided_p(0, 10) < 0.01


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa(["pass", "fail", "pass"], ["pass", "fail", "pass"]) == 1.0


def test_cohen_kappa_chance_agreement_near_zero():
    # Judge always says "pass", gold alternates -- kappa should be low/zero,
    # not the ~50% raw agreement rate would naively suggest.
    judge = ["pass"] * 4
    gold = ["pass", "fail", "pass", "fail"]
    k = cohen_kappa(judge, gold)
    assert k is not None and k <= 0.01


def test_pearson_spearman_perfect_correlation():
    r = pearson_spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert abs(r["pearson"] - 1.0) < 1e-6
    assert abs(r["spearman"] - 1.0) < 1e-6


def test_family_mismatch_warning_fires_on_match():
    assert family_mismatch_warning("anthropic", "anthropic") is not None


def test_family_mismatch_warning_silent_on_difference():
    assert family_mismatch_warning("anthropic", "openai") is None


def test_family_mismatch_warning_silent_when_generator_unknown():
    assert family_mismatch_warning("anthropic", None) is None


def _pr(reconciled):
    v = _v("ab", None)
    return PairwiseResult(case_id="c", verdict_ab=v, verdict_ba=v, winner_ab=None,
                            winner_ba=None, flipped=False, reconciled_winner=reconciled)


def test_ensemble_reconcile_agreement():
    assert ensemble_reconcile([_pr("a"), _pr("a")]) == "a"


def test_ensemble_reconcile_disagreement_is_tie():
    # Two oppositely-biased judges (e.g. alpha-family and beta-family, each
    # favoring their own style) disagreeing should land on "tie", not
    # arbitrarily pick one judge's biased vote as the ensemble's answer.
    assert ensemble_reconcile([_pr("a"), _pr("b")]) == "tie"


def test_ensemble_reconcile_majority_of_three():
    assert ensemble_reconcile([_pr("a"), _pr("a"), _pr("b")]) == "a"


def test_ensemble_reconcile_empty_is_inconsistent():
    assert ensemble_reconcile([]) == "inconsistent"
