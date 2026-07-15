import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge_pipeline.aggregate import aggregate_pairwise, aggregate_pointwise
from judge_pipeline.schema import (
    CriterionScore, JudgeMode, PairwiseResult, TestCase, TokenUsage, Verdict,
)


def _pointwise_verdict(case_id, score, passed):
    return Verdict(
        case_id=case_id, mode=JudgeMode.POINTWISE,
        per_criterion=[CriterionScore(criterion="correctness", score=score, rationale="r")],
        overall_score=score, overall_rationale="", passed=passed,
        judge_model="anthropic:claude-opus-4-8", tokens=TokenUsage(input_tokens=100, output_tokens=50),
    )


def test_aggregate_pointwise_pass_rate_and_mean():
    cases = [TestCase(id=f"c{i}", input="x", model_output="y") for i in range(4)]
    verdicts = [
        _pointwise_verdict("c0", 5.0, True),
        _pointwise_verdict("c1", 4.0, True),
        _pointwise_verdict("c2", 2.0, False),
        _pointwise_verdict("c3", 1.0, False),
    ]
    report = aggregate_pointwise("suite", cases, verdicts)
    assert report.n_cases == 4
    assert report.pass_rate == 0.5
    assert report.mean_overall_score == 3.0


def test_aggregate_pointwise_judge_error_excluded_from_scores_but_counted():
    cases = [TestCase(id="c0", input="x", model_output="y"),
             TestCase(id="c1", input="x", model_output="y")]
    ok = _pointwise_verdict("c0", 5.0, True)
    err = Verdict(case_id="c1", mode=JudgeMode.POINTWISE, per_criterion=[], overall_score=0.0,
                   overall_rationale="err", passed=False, is_judge_error=True,
                   judge_model="x", tokens=TokenUsage())
    report = aggregate_pointwise("suite", cases, [ok, err])
    assert report.n_errors == 1
    assert report.mean_overall_score == 5.0  # error excluded, not counted as 0


def _pairwise_result(case_id, reconciled, flipped=False):
    v_ab = Verdict(case_id=case_id, mode=JudgeMode.PAIRWISE, per_criterion=[], overall_score=3,
                    overall_rationale="", winner=None, order="ab", judge_model="x", tokens=TokenUsage())
    v_ba = Verdict(case_id=case_id, mode=JudgeMode.PAIRWISE, per_criterion=[], overall_score=3,
                    overall_rationale="", winner=None, order="ba", judge_model="x", tokens=TokenUsage())
    return PairwiseResult(case_id=case_id, verdict_ab=v_ab, verdict_ba=v_ba,
                            winner_ab=None, winner_ba=None, flipped=flipped,
                            reconciled_winner=reconciled)


def test_aggregate_pairwise_declares_significant_winner():
    cases = [TestCase(id=f"c{i}", input="x", output_a="a", output_b="b") for i in range(10)]
    # 9 wins for b, 1 for a -> should be declared significant at default alpha 0.10
    results = [_pairwise_result(f"c{i}", "b") for i in range(9)] + [_pairwise_result("c9", "a")]
    report = aggregate_pairwise("suite", cases, results)
    assert report.winner == "b"
    assert report.winner_significant is True
    assert report.winner_p_value < 0.10


def test_aggregate_pairwise_close_result_not_significant():
    cases = [TestCase(id=f"c{i}", input="x", output_a="a", output_b="b") for i in range(10)]
    results = [_pairwise_result(f"c{i}", "b") for i in range(6)] + [_pairwise_result(f"c{i+6}", "a") for i in range(4)]
    report = aggregate_pairwise("suite", cases, results)
    assert report.winner_significant is False
    assert "not significant" in report.winner


def test_aggregate_pairwise_flip_rate_reported():
    cases = [TestCase(id=f"c{i}", input="x", output_a="a", output_b="b") for i in range(4)]
    results = [_pairwise_result("c0", "a"), _pairwise_result("c1", "inconsistent", flipped=True),
               _pairwise_result("c2", "b"), _pairwise_result("c3", "inconsistent", flipped=True)]
    report = aggregate_pairwise("suite", cases, results)
    assert report.flip_rate == 0.5
