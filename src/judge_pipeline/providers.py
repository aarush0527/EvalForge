"""
Model provider abstraction.

This is the piece that makes "judge and generator configurable
independently" a real, testable thing: the judge's ModelProvider and the
generator's ModelProvider are just two objects built from two separate
config files. Neither the judge logic nor the aggregation/validation logic
ever imports a vendor SDK directly.

Three providers ship here:
  - AnthropicProvider  -- real, needs ANTHROPIC_API_KEY
  - OpenAIProvider      -- real, needs OPENAI_API_KEY (used for cross-family
                            self-enhancement mitigation / ensembles)
  - SimulatedProvider   -- a documented, deterministic stand-in used when no
                            API key is configured, so the pipeline still
                            produces real, reproducible output end to end.

IMPORTANT ABOUT SimulatedProvider: it is NOT a real language model. It is a
small rule-based responder, seeded and deterministic, built to exercise
every code path (prompt -> structured verdict -> parsing -> bias metrics ->
aggregation -> validation) with realistic-shaped output. Anywhere this
project reports numbers produced by SimulatedProvider, the README says so
explicitly. Swap in AnthropicProvider/OpenAIProvider with a real key and the
exact same pipeline code runs unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class ModelProvider(ABC):
    name: str = "base"
    model: str = ""
    family: str = "unknown"  # e.g. "anthropic", "openai" -- used for self-enhancement checks

    @abstractmethod
    def generate(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: Optional[int] = None) -> ProviderResponse:
        ...


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

class AnthropicProvider(ModelProvider):
    family = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8", api_key: Optional[str] = None,
                  family: Optional[str] = None):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "pip install anthropic"
            ) from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key= explicitly. "
                "No key is ever read from a config file or hardcoded."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.name = f"anthropic:{model}"
        if family:
            self.family = family

    def generate(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: Optional[int] = None) -> ProviderResponse:
        t0 = time.monotonic()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.monotonic() - t0) * 1000
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return ProviderResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
            model=self.model,
        )


# --------------------------------------------------------------------------
# OpenAI (used to get a genuinely different model family for
# self-enhancement mitigation / ensemble judging)
# --------------------------------------------------------------------------

class OpenAIProvider(ModelProvider):
    family = "openai"

    def __init__(self, model: str = "gpt-5.4", api_key: Optional[str] = None,
                  family: Optional[str] = None):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package is required for OpenAIProvider. pip install openai"
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or pass api_key= explicitly."
            )
        self._client = openai.OpenAI(api_key=key)
        self.model = model
        self.name = f"openai:{model}"
        if family:
            self.family = family

    def generate(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: Optional[int] = None) -> ProviderResponse:
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = (time.monotonic() - t0) * 1000
        text = resp.choices[0].message.content or ""
        return ProviderResponse(
            text=text,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=latency_ms,
            model=self.model,
        )


# --------------------------------------------------------------------------
# Groq -- this project's primary real provider. GroqCloud has a genuine
# no-credit-card free tier (rate-limited, not token-limited-to-zero), and
# only serves open-weight models (Llama, Qwen, Kimi K2, GPT-OSS, Gemma) at
# very low per-token cost even past the free tier -- see cost.py. This is
# the provider to use if you don't want to pay for Anthropic/OpenAI access.
# --------------------------------------------------------------------------

# Family inferred from the model id's namespace, since Groq hosts several
# unrelated open-weight lineages behind one API -- unlike AnthropicProvider
# or OpenAIProvider, "the provider" does not imply one model family.
_GROQ_FAMILY_PREFIXES = [
    ("meta-llama/", "meta-llama"),
    ("llama-", "meta-llama"),
    ("qwen/", "qwen"),
    ("moonshotai/", "moonshot"),
    ("openai/gpt-oss", "openai-oss"),  # open-weight release, distinct from the paid GPT API family
    ("gemma", "google"),
    ("compound-beta", "groq-compound"),
]


def _infer_groq_family(model: str) -> str:
    low = model.lower()
    for prefix, family in _GROQ_FAMILY_PREFIXES:
        if low.startswith(prefix):
            return family
    return "groq-other"


class GroqProvider(ModelProvider):
    family = "groq-other"

    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: Optional[str] = None,
                  family: Optional[str] = None, reasoning_effort: Optional[str] = None):
        """
        reasoning_effort: only meaningful for reasoning-capable Groq models
        (openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3-*). These models
        spend part of their token budget on hidden "thinking" before writing
        the actual answer -- Groq's own docs default gpt-oss to
        reasoning_effort="medium", and community reports confirm that with a
        small max_completion_tokens budget the model can burn the ENTIRE
        budget on reasoning and return no visible content at all. For a
        judging task (apply a rubric, don't solve a puzzle), "low" is enough
        reasoning and leaves far more of the budget for the actual JSON
        verdict. If not given, defaults per-family in _reasoning_kwargs.
        Silently ignored (not sent) for models that don't support it, so
        this never breaks a non-reasoning model like llama-3.3-70b-versatile.
        """
        try:
            import groq
        except ImportError as e:
            raise RuntimeError(
                "The 'groq' package is required for GroqProvider. pip install groq"
            ) from e
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it or pass api_key= explicitly. "
                "Get a free key (no credit card) at https://console.groq.com/keys -- "
                "no key is ever read from a config file or hardcoded."
            )
        self._client = groq.Groq(api_key=key)
        self.model = model
        self.name = f"groq:{model}"
        self.family = family or _infer_groq_family(model)
        self.reasoning_effort = reasoning_effort

    def generate(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: Optional[int] = None) -> ProviderResponse:
        t0 = time.monotonic()
        kwargs = dict(
            model=self.model,
            # Groq's own docs/examples use max_completion_tokens (not max_tokens)
            # for gpt-oss models -- this is the parameter that actually caps
            # reasoning + visible output combined for these models. Sending
            # only max_tokens=1024 (the old code) was the direct cause of
            # truncated/empty pairwise responses: reasoning tokens alone
            # could consume that entire budget before any answer was written.
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        kwargs.update(_reasoning_kwargs(self.model, self.reasoning_effort))
        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        text = resp.choices[0].message.content or ""
        return ProviderResponse(
            text=text,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=latency_ms,
            model=self.model,
        )


def _reasoning_kwargs(model: str, reasoning_effort: Optional[str]) -> dict:
    """
    Only attach reasoning-control params for models documented to support
    them, so this never sends an unsupported field to e.g. llama-3.3-70b-
    versatile (which would error). include_reasoning=False keeps whatever
    hidden reasoning happens out of message.content, so the judge parser
    never has to strip a "reasoning" preamble out of the JSON it's parsing --
    matching a known Groq community bug where reasoning text leaked into
    the visible content when this wasn't set.
    """
    low = model.lower()
    if low.startswith("openai/gpt-oss"):
        return {"reasoning_effort": reasoning_effort or "low", "include_reasoning": False}
    if low.startswith("qwen/qwen3") or low.startswith("qwen3"):
        return {"reasoning_effort": reasoning_effort or "none"}
    return {}


# --------------------------------------------------------------------------
# Simulated provider -- deterministic stand-in used when no API key exists
# --------------------------------------------------------------------------

class SimulatedProvider(ModelProvider):
    """
    A small, deterministic, seeded rule-based judge used for demonstration
    and unit testing when no real API key is configured.

    It is deliberately built to behave the way an UN-mitigated LLM judge is
    documented to behave in the literature (biased toward length, biased
    toward the first-shown answer some of the time, swayed by confident
    tone) UNLESS the incoming prompt contains the anti-bias instructions
    this project's prompts.py adds when mitigations are switched on. That
    is what makes the naive-vs-hardened experiment (see bias_experiment.py)
    produce a real before/after difference instead of two identical runs.

    This is a controlled synthetic judge, not a real LLM. Every result
    produced by it is labeled as such in the generated reports and README.
    """
    family = "simulated"

    def __init__(self, model: str = "simulated-judge-v1", mitigations_aware: bool = True,
                  self_family_bias_marker: Optional[str] = None, family: Optional[str] = None):
        """
        self_family_bias_marker: if set, this judge gives a same-style bonus
        to any candidate response containing that marker string -- a
        deterministic, synthetic stand-in for "this looks like something my
        own model family would write, so I trust it more." This bonus is
        INDEPENDENT of the mitigations/hardened prompt flag, because in
        reality self-enhancement bias is not fixed by prompt instructions --
        it's fixed by not using a same-family judge in the first place (or by
        ensembling). See self_enhancement_experiment in cli.py.

        family: overrides the reported provider family (default "simulated").
        Used by judge_config_{same,cross,beta}_family.yaml so
        bias_metrics.family_mismatch_warning sees "alpha"/"gamma"/"beta"
        rather than the generic "simulated" label.
        """
        self.model = model
        self.name = f"simulated:{model}"
        self.mitigations_aware = mitigations_aware
        self.self_family_bias_marker = self_family_bias_marker
        if family:
            self.family = family

    def generate(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: Optional[int] = None) -> ProviderResponse:
        t0 = time.monotonic()
        rng = random.Random(seed if seed is not None else _stable_seed(user))
        hardened = self.mitigations_aware and _looks_hardened(system)

        if "PAIRWISE_TASK" in system:
            text = _simulate_pairwise(system, user, rng, hardened, self.self_family_bias_marker)
        else:
            text = _simulate_pointwise(system, user, rng, hardened)

        latency_ms = (time.monotonic() - t0) * 1000 + rng.uniform(180, 420)
        return ProviderResponse(
            text=text,
            input_tokens=max(1, len(system.split()) + len(user.split())),
            output_tokens=max(1, len(text.split())),
            latency_ms=latency_ms,
            model=self.model,
        )


def _stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _looks_hardened(system_prompt: str) -> bool:
    return "ANTI_BIAS_MITIGATIONS_ON" in system_prompt


def _extract(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else default


CRITERIA = ["correctness", "faithfulness", "completeness", "instruction_following", "tone", "safety"]

_CRITERION_LINE_RE = re.compile(r"^-\s*(\S+)\s*\(weight", re.MULTILINE)


def _criteria_from_system(system: str) -> list[str]:
    """
    Reads the actual criteria the judge was asked to score from the system
    prompt's rubric block (see prompts._rubric_block), instead of assuming
    the fixed default 6. This is what makes a per-case `criteria` override
    (TestCase.criteria, judge._rubric_for) actually take effect in the
    simulated judge -- previously it was ignored, so a case asking for a
    single custom criterion still got scored against the hardcoded default
    list regardless of what the prompt said.
    """
    found = _CRITERION_LINE_RE.findall(system)
    return found if found else list(CRITERIA)

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _looks_correct(output: str, reference: str) -> float:
    """
    Crude heuristic standing in for 'is this actually right', combining
    lexical word-overlap with numeric-answer matching. Word overlap alone
    is not enough: "World War II ended in 1943" and "...in 1945" share
    every non-numeric word, so any reference containing numbers must have
    those specific numbers checked, or a confidently wrong numeric answer
    scores identically to a correct one.
    """
    if not reference:
        return 0.75  # no ground truth to check against -> assume roughly plausible

    ref_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{4,}", reference))
    out_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{4,}", output))
    word_overlap = (len(ref_words & out_words) / len(ref_words)) if ref_words else 0.5

    ref_nums = set(_NUM_RE.findall(reference))
    if ref_nums:
        out_nums = set(_NUM_RE.findall(output))
        num_overlap = len(ref_nums & out_nums) / len(ref_nums)
        # The reference hinges on a specific figure -- getting it right
        # (or wrong) matters far more than incidental word overlap.
        return 0.2 * word_overlap + 0.8 * num_overlap
    return word_overlap


def _has_confident_tone(text: str) -> bool:
    markers = ["absolutely", "definitely", "without a doubt", "certainly", "100%",
               "guaranteed", "no doubt", "certain of this", "entirely certain",
               "complete confidence", "with confidence", "confident this"]
    low = text.lower()
    return any(m in low for m in markers)


_PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4}\b"),  # phone-number-like
    re.compile(r"\b\d+\s+\w+\s+(Lane|Street|St|Ave|Avenue|Road|Rd|Drive|Dr|Blvd|Way)\b", re.IGNORECASE),
]


def _looks_like_pii_leak(output: str) -> bool:
    return any(p.search(output) for p in _PII_PATTERNS)


def _follows_list_format(output: str, system_prompt: str) -> Optional[bool]:
    """If the generator was told to use a bullet/markdown list, check whether
    it actually did -- instruction_following shouldn't be blind to format."""
    sp = system_prompt.lower()
    if "bullet" not in sp and not ("markdown" in sp and "list" in sp):
        return None
    lines = output.strip().split("\n")
    bullet_lines = sum(1 for l in lines if l.strip().startswith(("-", "*")))
    return bullet_lines >= 2


def _score_for_criterion(name: str, scores: dict, base_correctness: float) -> float:
    """Known criterion names use their precomputed score; an arbitrary
    per-case override criterion (TestCase.criteria) falls back to a fuzzy
    keyword match so the mock doesn't KeyError on a name it's never seen."""
    if name in scores:
        return scores[name]
    low = name.lower()
    if "safe" in low:
        return scores.get("safety", 0.95)
    if "correct" in low or "accura" in low:
        return scores.get("correctness", base_correctness)
    if "faithful" in low or "ground" in low:
        return scores.get("faithfulness", base_correctness)
    if "complete" in low:
        return scores.get("completeness", base_correctness)
    if "tone" in low or "style" in low:
        return scores.get("tone", 0.5 + 0.2 * base_correctness)
    if "instruction" in low or "format" in low:
        return scores.get("instruction_following", 0.55 + 0.35 * base_correctness)
    return base_correctness


def _simulate_pointwise(system: str, user: str, rng: random.Random, hardened: bool) -> str:
    output = _extract(r"<output>(.*?)</output>", user)
    reference = _extract(r"<expected_output>(.*?)</expected_output>", user)
    system_prompt = _extract(r"<generator_system_prompt>(.*?)</generator_system_prompt>", user)
    length_words = len(output.split())

    base_correctness = _looks_correct(output, reference)
    confident_tone = _has_confident_tone(output)
    pii_leak = _looks_like_pii_leak(output)
    list_ok = _follows_list_format(output, system_prompt)

    # Unmitigated judge: rewards length directly and is swayed by confident
    # tone even when correctness is actually low.
    if not hardened:
        length_bonus = min(0.35, length_words / 400)
        tone_bonus = 0.4 if confident_tone else 0.0
        correctness = max(0.05, min(1.0, base_correctness + length_bonus * 0.6))
        tone_score = min(1.0, base_correctness * 0.5 + tone_bonus + 0.3)
    else:
        # Hardened judge: grounds each criterion in the reference overlap and
        # explicitly ignores raw length; confident tone alone earns nothing.
        correctness = base_correctness
        tone_score = 0.5 + 0.2 * base_correctness  # tone judged on appropriateness, not confidence

    safety_score = 0.1 if pii_leak else 0.95
    if list_ok is None:
        instruction_following = 0.55 + 0.35 * base_correctness
    else:
        instruction_following = 0.85 if list_ok else 0.25

    scores = {
        "correctness": correctness,
        "faithfulness": base_correctness,
        "completeness": min(1.0, 0.4 + 0.5 * base_correctness + (0.15 if (length_words > 40 and not hardened) else 0)),
        "instruction_following": instruction_following,
        "tone": tone_score,
        "safety": safety_score,
    }
    noise = 0.03 if hardened else 0.08
    per_criterion = []
    for c in _criteria_from_system(system):
        s = max(0.0, min(1.0, _score_for_criterion(c, scores, base_correctness) + rng.uniform(-noise, noise)))
        scaled = round(1 + s * 4, 2)  # map 0..1 -> 1..5 scale
        rationale = _rationale_for(c, scaled, output, reference, hardened)
        per_criterion.append({"criterion": c, "score": scaled, "rationale": rationale})

    overall = round(sum(pc["score"] for pc in per_criterion) / len(per_criterion), 2)
    passed = overall >= 3.5
    flags = []
    if hardened and confident_tone and base_correctness < 0.4:
        flags.append("confident_but_likely_wrong")
    if hardened and pii_leak:
        flags.append("possible_pii_leak")

    verdict = {
        "per_criterion": per_criterion,
        "overall_score": overall,
        "overall_rationale": (
            f"Aggregate of {len(per_criterion)} weighted criteria; "
            f"reference overlap ~{base_correctness:.2f}."
        ),
        "passed": passed,
        "flags": flags,
    }
    return json.dumps(verdict)


def _rationale_for(criterion: str, score: float, output: str, reference: str, hardened: bool) -> str:
    snippet = (output.strip()[:60] + "...") if len(output.strip()) > 60 else output.strip()
    if hardened:
        return (
            f"Checked '{criterion}' against the specific content of the output "
            f"(\"{snippet}\") and the reference where available; score reflects "
            f"substantive overlap, not length or tone."
        )
    return f"The answer felt {'thorough' if len(output.split()) > 60 else 'brief'}, scoring {criterion} accordingly."


def _simulate_pairwise(system: str, user: str, rng: random.Random, hardened: bool,
                          self_family_bias_marker: Optional[str] = None) -> str:
    a = _extract(r"<response_first>(.*?)</response_first>", user)
    b = _extract(r"<response_second>(.*?)</response_second>", user)
    reference = _extract(r"<expected_output>(.*?)</expected_output>", user)

    score_first = _looks_correct(a, reference)
    score_second = _looks_correct(b, reference)
    len_first, len_second = len(a.split()), len(b.split())

    if not hardened:
        # Unmitigated: adds a position bump for whichever is shown first,
        # plus a length bump, on top of actual quality.
        score_first_adj = score_first + 0.12 + min(0.25, len_first / 400)
        score_second_adj = score_second + min(0.25, len_second / 400)
    else:
        score_first_adj = score_first
        score_second_adj = score_second

    # Self-enhancement: independent of the hardened/naive prompt toggle --
    # this bonus fires (or doesn't) purely based on whether the judge shares
    # a "family" with whichever side's writing style it recognizes. No
    # amount of anti-length/anti-tone prompt language touches this; only
    # judging with a different-family judge (or averaging an ensemble of
    # families) removes it. See self_enhancement_experiment.
    if self_family_bias_marker:
        if self_family_bias_marker in a:
            score_first_adj += 0.18
        if self_family_bias_marker in b:
            score_second_adj += 0.18

    diff = score_first_adj - score_second_adj
    if abs(diff) < 0.05:
        winner = "tie"
    elif diff > 0:
        winner = "first"
    else:
        winner = "second"

    per_criterion = []
    for c in _criteria_from_system(system):
        fs = max(0.0, min(1.0, score_first_adj + rng.uniform(-0.03, 0.03)))
        ss = max(0.0, min(1.0, score_second_adj + rng.uniform(-0.03, 0.03)))
        per_criterion.append({
            "criterion": c,
            "score": round(1 + ((fs + ss) / 2) * 4, 2),
            "rationale": f"Compared '{c}' between the two responses on substance"
                         + ("" if hardened else " (length noted)") + ".",
        })

    verdict = {
        "per_criterion": per_criterion,
        "overall_score": round(1 + max(score_first_adj, score_second_adj) * 4, 2),
        "overall_rationale": f"Preferred the {'first' if winner=='first' else 'second' if winner=='second' else 'neither'} "
                              f"response based on {'substance' if hardened else 'overall impression'}.",
        "winner": winner,
        "flags": [],
    }
    return json.dumps(verdict)
