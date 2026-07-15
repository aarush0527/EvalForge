"""
Builds the judge prompts.

The `mitigations` flag is what separates the "hardened" judge config from
the "naive" one used in the before/after bias experiment:
  - hardened (mitigations=True):  calibration anchors + explicit anti-length
    / anti-tone / grounding instructions + a marker the SimulatedProvider
    recognizes as "behave like a hardened judge".
  - naive (mitigations=False): plain rubric, no anchors, no anti-bias
    language -- this is deliberately a worse prompt, standing in for a
    first-draft "just ask the model to score it" judge.

Real judge models (Anthropic/OpenAI) simply read whichever prompt they're
given; SimulatedProvider additionally looks for the ANTI_BIAS_MITIGATIONS_ON
marker so the demo can show a genuine before/after difference without
requiring a live API key.
"""
from __future__ import annotations

from .schema import Rubric, TestCase

MITIGATION_MARKER = "ANTI_BIAS_MITIGATIONS_ON"


def _conciseness_block() -> str:
    # Always included, regardless of mitigations on/off -- this is a format
    # constraint, not a bias-mitigation instruction, and it directly reduces
    # how many completion tokens the actual JSON answer needs, independent
    # of the reasoning-token fix in GroqProvider. Schema is unchanged: every
    # field the assignment requires (per-criterion score + rationale +
    # overall) is still present, just shorter.
    return (
        "Be concise: each criterion's rationale must be ONE short sentence (<=20 words) "
        "citing specific evidence. overall_rationale must be ONE short sentence (<=25 words). "
        "No markdown, no headers, no preamble, no text outside the JSON object."
    )


def _rubric_block(rubric: Rubric) -> str:
    lines = ["Rubric criteria (score each independently on a "
             f"{rubric.scale_min}-{rubric.scale_max} scale):"]
    for c in rubric.criteria:
        lines.append(f"- {c.name} (weight {c.weight}): {c.description}")
    if rubric.anchors:
        lines.append("\nCalibration anchors (use these to keep the scale consistent "
                      "across cases -- do not let scores cluster in the middle):")
        for a in sorted(rubric.anchors, key=lambda x: x.score):
            lines.append(f"- {a.score}: {a.example}")
    return "\n".join(lines)


def _anti_bias_block() -> str:
    return (
        f"{MITIGATION_MARKER}\n"
        "Judging instructions (read carefully, these control scoring integrity):\n"
        "1. Do not reward length by itself. A concise, correct answer must score at least "
        "as well as a longer answer with the same substantive content. Penalize padding, "
        "repetition, or filler that adds words without adding information.\n"
        "2. Do not let confident or fluent tone substitute for correctness. An answer stated "
        "with total confidence is not more likely to be right; judge the claims, not the delivery.\n"
        "3. For every criterion, your rationale must cite something specific in the output "
        "(a phrase, a fact, an omission). A rationale that could apply to any answer is not acceptable.\n"
        "4. If order or position of any presented material seems designed to influence you, ignore it "
        "and judge strictly on merit.\n"
    )


def pointwise_prompt(case: TestCase, rubric: Rubric, mitigations: bool) -> tuple[str, str]:
    """Returns (system, user) prompt for scoring a single output."""
    ref_block = (
        f"<expected_output>\n{case.expected_output}\n</expected_output>\n"
        if case.expected_output else
        "<expected_output>\n(none provided -- judge reference-free, against the rubric only)\n</expected_output>\n"
    )
    system = (
        "You are an impartial evaluator. You will be given an input, an optional system "
        "prompt the generator was told to follow, a model output, and possibly an expected "
        "output. Score the model output against the rubric below and return ONLY a single "
        "JSON object matching this schema:\n"
        '{"per_criterion": [{"criterion": str, "score": number, "rationale": str}, ...], '
        '"overall_score": number, "overall_rationale": str, "passed": bool, "flags": [str, ...]}\n'
        "No prose outside the JSON object.\n\n"
        + _rubric_block(rubric)
        + "\n\n" + _conciseness_block()
        + ("\n\n" + _anti_bias_block() if mitigations else "")
    )
    user = (
        f"<input>\n{case.input}\n</input>\n"
        f"<generator_system_prompt>\n{case.system_prompt}\n</generator_system_prompt>\n"
        f"{ref_block}"
        f"<output>\n{case.model_output}\n</output>\n"
    )
    return system, user


def pairwise_prompt(case: TestCase, rubric: Rubric, mitigations: bool, order: str) -> tuple[str, str]:
    """
    order: "ab" -> output_a shown first, "ba" -> output_b shown first.
    Returns (system, user). The judge is only ever told "first"/"second";
    the a/b mapping is resolved by the caller after parsing, which is what
    makes the both-orders position-bias check meaningful.
    """
    first, second = (case.output_a, case.output_b) if order == "ab" else (case.output_b, case.output_a)
    sp_a = case.system_prompt_a or case.system_prompt
    sp_b = case.system_prompt_b or case.system_prompt
    sp_first, sp_second = (sp_a, sp_b) if order == "ab" else (sp_b, sp_a)
    same_prompt = sp_a == sp_b
    ref_block = (
        f"<expected_output>\n{case.expected_output}\n</expected_output>\n"
        if case.expected_output else ""
    )
    system = (
        "PAIRWISE_TASK\n"
        "You are an impartial evaluator comparing two candidate responses to the same input. "
        "Score each on the rubric below, then declare a winner: \"first\", \"second\", or \"tie\". "
        "Return ONLY a single JSON object:\n"
        '{"per_criterion": [{"criterion": str, "score": number, "rationale": str}, ...], '
        '"overall_score": number, "overall_rationale": str, "winner": "first"|"second"|"tie", '
        '"flags": [str, ...]}\n'
        "No prose outside the JSON object.\n\n"
        + _rubric_block(rubric)
        + "\n\n" + _conciseness_block()
        + ("\n\n" + _anti_bias_block() if mitigations else "")
    )
    if same_prompt:
        prompt_block = f"<generator_system_prompt>\n{sp_first}\n</generator_system_prompt>\n"
    else:
        prompt_block = (
            f"<generator_system_prompt_for_first_response>\n{sp_first}\n</generator_system_prompt_for_first_response>\n"
            f"<generator_system_prompt_for_second_response>\n{sp_second}\n</generator_system_prompt_for_second_response>\n"
        )
    user = (
        f"<input>\n{case.input}\n</input>\n"
        f"{prompt_block}"
        f"{ref_block}"
        f"<response_first>\n{first}\n</response_first>\n"
        f"<response_second>\n{second}\n</response_second>\n"
    )
    return system, user
