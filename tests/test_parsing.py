import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge_pipeline.parsing import extract_json, parse_verdict, validate_shape


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    text = "Here you go:\n```json\n{\"a\": 1}\n```\nHope that helps!"
    assert extract_json(text) == {"a": 1}


def test_extract_json_with_surrounding_prose_no_fence():
    text = 'Sure, the verdict is: {"a": 1} -- let me know if you need more.'
    assert extract_json(text) == {"a": 1}


def test_extract_json_garbage_returns_none():
    assert extract_json("not json at all") is None


def test_validate_shape_missing_key():
    err = validate_shape({"overall_score": 1}, {"per_criterion", "overall_score"})
    assert err is not None and "per_criterion" in err


def test_validate_shape_ok():
    data = {
        "per_criterion": [{"criterion": "correctness", "score": 4, "rationale": "x"}],
        "overall_score": 4,
    }
    assert validate_shape(data, {"per_criterion", "overall_score"}) is None


def test_parse_verdict_succeeds_on_first_try():
    raw = '{"per_criterion": [{"criterion": "c", "score": 3, "rationale": "r"}], "overall_score": 3, "overall_rationale": "x", "passed": true, "flags": []}'
    result = parse_verdict(raw, pairwise=False)
    assert result.ok
    assert result.retries == 0


def test_parse_verdict_retries_and_recovers():
    calls = {"n": 0}

    def retry_fn(error: str) -> str:
        calls["n"] += 1
        return '{"per_criterion": [{"criterion": "c", "score": 3, "rationale": "r"}], "overall_score": 3, "overall_rationale": "x", "passed": true, "flags": []}'

    result = parse_verdict("this is not json", pairwise=False, retry_fn=retry_fn, max_retries=1)
    assert result.ok
    assert result.retries == 1
    assert calls["n"] == 1


def test_parse_verdict_gives_up_after_max_retries():
    def retry_fn(error: str) -> str:
        return "still not json"

    result = parse_verdict("not json", pairwise=False, retry_fn=retry_fn, max_retries=2)
    assert not result.ok
    assert result.retries == 2


def test_parse_verdict_pairwise_requires_winner_key():
    raw = '{"per_criterion": [{"criterion": "c", "score": 3, "rationale": "r"}], "overall_score": 3, "overall_rationale": "x"}'
    result = parse_verdict(raw, pairwise=True, max_retries=0)
    assert not result.ok  # missing "winner"
