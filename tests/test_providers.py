import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from judge_pipeline.judge import Judge
from judge_pipeline.providers import (
    GroqProvider, SimulatedProvider, _criteria_from_system, _infer_groq_family, _reasoning_kwargs,
)
from judge_pipeline.rubric import DEFAULT_RUBRIC
from judge_pipeline.schema import RubricCriterion, TestCase


def test_criteria_from_system_parses_rubric_block():
    system = (
        "Rubric criteria (score each independently on a 1-5 scale):\n"
        "- correctness (weight 0.3): blah\n"
        "- safety_only (weight 1.0): blah\n"
    )
    assert _criteria_from_system(system) == ["correctness", "safety_only"]


def test_criteria_from_system_falls_back_to_default_when_absent():
    assert _criteria_from_system("no rubric block here") == [
        "correctness", "faithfulness", "completeness", "instruction_following", "tone", "safety",
    ]


def test_per_case_criteria_override_actually_takes_effect():
    """Regression test: SimulatedProvider used to hardcode its own 6-criteria
    list and ignore whatever rubric was actually sent in the prompt, so a
    per-case override was silently dropped. This must not regress."""
    case = TestCase(
        input="Is this safe content?",
        model_output="Yes, perfectly safe.",
        criteria=[RubricCriterion(name="safety_only", description="Only check safety.", weight=1.0)],
    )
    judge = Judge(provider=SimulatedProvider(), rubric=DEFAULT_RUBRIC, mitigations=True)
    verdict = judge.judge_pointwise(case)
    assert [pc.criterion for pc in verdict.per_criterion] == ["safety_only"]


def test_groq_provider_raises_clearly_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqProvider(model="llama-3.3-70b-versatile")


@pytest.mark.parametrize("model,expected_family", [
    ("llama-3.1-8b-instant", "meta-llama"),
    ("llama-3.3-70b-versatile", "meta-llama"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama"),
    ("qwen/qwen3-32b", "qwen"),
    ("moonshotai/kimi-k2-instruct", "moonshot"),
    ("openai/gpt-oss-120b", "openai-oss"),
    ("gemma2-9b-it", "google"),
    ("some-brand-new-model", "groq-other"),
])
def test_infer_groq_family(model, expected_family):
    assert _infer_groq_family(model) == expected_family


def test_groq_provider_family_override(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key-for-construction-only")
    p = GroqProvider(model="llama-3.3-70b-versatile", family="custom-family")
    assert p.family == "custom-family"


def test_groq_provider_default_family_inferred(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key-for-construction-only")
    p = GroqProvider(model="qwen/qwen3-32b")
    assert p.family == "qwen"

def test_reasoning_kwargs_gpt_oss_gets_low_effort_and_hides_reasoning():
    assert _reasoning_kwargs("openai/gpt-oss-120b", None) == {
        "reasoning_effort": "low", "include_reasoning": False,
    }


def test_reasoning_kwargs_respects_explicit_override():
    assert _reasoning_kwargs("openai/gpt-oss-120b", "high") == {
        "reasoning_effort": "high", "include_reasoning": False,
    }


def test_reasoning_kwargs_qwen3_disables_reasoning_by_default():
    assert _reasoning_kwargs("qwen/qwen3-32b", None) == {"reasoning_effort": "none"}


def test_reasoning_kwargs_non_reasoning_model_gets_nothing():

    assert _reasoning_kwargs("llama-3.3-70b-versatile", None) == {}


def test_groq_provider_sends_max_completion_tokens_not_max_tokens(monkeypatch):
    """
    The actual bug: GroqProvider used to send `max_tokens=1024` to Groq's
    chat completions endpoint. Groq's own docs use `max_completion_tokens`
    for gpt-oss models, and reasoning tokens count against that budget --
    a small `max_tokens` value let the model spend the whole budget
    "thinking" and return empty/truncated content, breaking every pairwise
    call (which needs more budget than pointwise). This asserts the fix:
    the request must use max_completion_tokens, never max_tokens.
    """
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key-for-construction-only")
    provider = GroqProvider(model="openai/gpt-oss-120b")

    captured = {}

    class _FakeMessage:
        content = '{"per_criterion": [], "overall_score": 5, "overall_rationale": "ok", "passed": true, "flags": []}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5

    class _FakeResponse:
        choices = [_FakeChoice()]
        usage = _FakeUsage()

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    provider._client.chat.completions.create = fake_create
    provider.generate("system prompt", "user prompt", temperature=0.0, max_tokens=4096)

    assert "max_completion_tokens" in captured, "must use max_completion_tokens for gpt-oss models"
    assert captured["max_completion_tokens"] == 4096
    assert "max_tokens" not in captured, "must not also send the legacy max_tokens field"
    assert captured["reasoning_effort"] == "low"
    assert captured["include_reasoning"] is False


def test_judge_retry_doubles_token_budget_on_parse_failure(monkeypatch):
    """Regression test: retrying a truncated response with the SAME budget
    that just failed reproduces the same truncation. The retry must ask for
    a larger budget, not repeat the original one."""
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key-for-construction-only")
    provider = GroqProvider(model="openai/gpt-oss-120b")

    calls = []

    class _FakeMessage:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMessage(content)

    class _FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5

    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()

    def fake_create(**kwargs):
        calls.append(kwargs["max_completion_tokens"])
        if len(calls) == 1:
            return _FakeResponse('{"per_criterion": [{"criterion": "c", "score": 3') 
        return _FakeResponse(
            '{"per_criterion": [{"criterion": "c", "score": 3, "rationale": "r"}], '
            '"overall_score": 3, "overall_rationale": "ok", "passed": true, "flags": []}'
        )

    provider._client.chat.completions.create = fake_create
    judge = Judge(provider=provider, mitigations=True, max_tokens=1000, max_parse_retries=1)
    case = TestCase(input="x", model_output="y")
    verdict = judge.judge_pointwise(case)

    assert verdict.is_judge_error is False
    assert calls == [1000, 2000]  
