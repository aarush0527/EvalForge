"""
The judge itself: turns a TestCase into a structured Verdict (pointwise)
or a PairwiseResult (pairwise, both orders, reconciled).
"""
from __future__ import annotations

from collections import Counter
from typing import Literal, Optional

from .audit_log import AuditLog
from .parsing import parse_verdict
from .prompts import pairwise_prompt, pointwise_prompt
from .providers import ModelProvider
from .rubric import DEFAULT_RUBRIC
from .schema import CriterionScore, JudgeMode, PairwiseResult, Rubric, TestCase, TokenUsage, Verdict


class Judge:
    def __init__(
        self,
        provider: ModelProvider,
        rubric: Rubric = DEFAULT_RUBRIC,
        mitigations: bool = True,
        audit_log: Optional[AuditLog] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_parse_retries: int = 1,
    ):
        self.provider = provider
        self.rubric = rubric
        self.mitigations = mitigations
        self.audit_log = audit_log
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_parse_retries = max_parse_retries

    def _rubric_for(self, case: TestCase) -> Rubric:
        if case.criteria:
            return Rubric(
                criteria=case.criteria,
                scale_min=self.rubric.scale_min,
                scale_max=self.rubric.scale_max,
                anchors=self.rubric.anchors,
                pass_threshold=self.rubric.pass_threshold,
            )
        return self.rubric

    def _call_and_parse(self, *, case_id: str, mode: str, system: str, user: str,
                          pairwise: bool) -> tuple[dict | None, str, TokenUsage, float, int, bool]:
        resp = self.provider.generate(system, user, temperature=self.temperature,
                                        max_tokens=self.max_tokens)
        raw = resp.text
        tokens = TokenUsage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens)
        latency = resp.latency_ms

        def retry_fn(error: str) -> str:
            # The most common real-world cause of a parse failure is a
            # truncated response (ran out of completion budget, possibly to
            # a reasoning model's hidden "thinking" -- see GroqProvider),
            # not a formatting mistake the model can just fix in place.
            # Retrying with the SAME budget reproduces the same truncation,
            # so give the retry meaningfully more room.
            retry_max_tokens = self.max_tokens * 2
            prev_response_snippet = raw if len(raw) <= 1500 else (raw[:1500] + " ...[truncated]")
            fix_user = (
                f"{user}\n\n---\nYour previous response could not be parsed as valid JSON "
                f"matching the required schema. Parse error: {error}\n"
                f"Your previous response was:\n{prev_response_snippet}\n\n"
                f"Return ONLY the corrected JSON object, nothing else. Be concise."
            )
            r2 = self.provider.generate(system, fix_user, temperature=self.temperature,
                                          max_tokens=retry_max_tokens)
            nonlocal_tokens = TokenUsage(
                input_tokens=tokens.input_tokens + r2.input_tokens,
                output_tokens=tokens.output_tokens + r2.output_tokens,
            )
            tokens.input_tokens, tokens.output_tokens = (
                nonlocal_tokens.input_tokens, nonlocal_tokens.output_tokens,
            )
            return r2.text

        result = parse_verdict(raw, pairwise=pairwise, retry_fn=retry_fn,
                                 max_retries=self.max_parse_retries)

        if self.audit_log is not None:
            self.audit_log.log(
                case_id=case_id, mode=mode, system_prompt=system, user_prompt=user,
                raw_response=result.raw_response, judge_model=self.provider.name,
                input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens,
                latency_ms=latency, parse_retries=result.retries,
            )

        return (result.data if result.ok else None, result.raw_response, tokens, latency,
                result.retries, result.ok)

    # ---------------------------------------------------------------- #
    # Pointwise
    # ---------------------------------------------------------------- #

    def judge_pointwise(self, case: TestCase) -> Verdict:
        rubric = self._rubric_for(case)
        system, user = pointwise_prompt(case, rubric, self.mitigations)
        data, raw, tokens, latency, retries, ok = self._call_and_parse(
            case_id=case.id, mode="pointwise", system=system, user=user, pairwise=False,
        )

        if not ok:
            return Verdict(
                case_id=case.id, mode=JudgeMode.POINTWISE, per_criterion=[],
                overall_score=0.0, overall_rationale="Judge error: could not parse a valid verdict.",
                passed=False, flags=["judge_error"], judge_model=self.provider.name,
                tokens=tokens, latency_ms=latency, raw_response=raw, parse_retries=retries,
                is_judge_error=True,
            )

        per_criterion = [CriterionScore(**pc) for pc in data["per_criterion"]]
        return Verdict(
            case_id=case.id, mode=JudgeMode.POINTWISE, per_criterion=per_criterion,
            overall_score=float(data["overall_score"]),
            overall_rationale=str(data.get("overall_rationale", "")),
            passed=bool(data.get("passed", float(data["overall_score"]) >= rubric.pass_threshold)),
            flags=list(data.get("flags", [])),
            judge_model=self.provider.name, tokens=tokens, latency_ms=latency,
            raw_response=raw, parse_retries=retries,
        )

    # ---------------------------------------------------------------- #
    # Pairwise
    # ---------------------------------------------------------------- #

    def _judge_pairwise_one_order(self, case: TestCase, order: Literal["ab", "ba"]) -> Verdict:
        rubric = self._rubric_for(case)
        system, user = pairwise_prompt(case, rubric, self.mitigations, order)
        data, raw, tokens, latency, retries, ok = self._call_and_parse(
            case_id=case.id, mode=f"pairwise:{order}", system=system, user=user, pairwise=True,
        )

        if not ok:
            return Verdict(
                case_id=case.id, mode=JudgeMode.PAIRWISE, per_criterion=[],
                overall_score=0.0, overall_rationale="Judge error: could not parse a valid verdict.",
                winner=None, flags=["judge_error"], judge_model=self.provider.name,
                order=order, tokens=tokens, latency_ms=latency, raw_response=raw,
                parse_retries=retries, is_judge_error=True,
            )

        per_criterion = [CriterionScore(**pc) for pc in data["per_criterion"]]
        raw_winner = data.get("winner", "tie")  # "first" | "second" | "tie"
        # Map "first"/"second" -> a/b using the order this call used.
        if raw_winner == "first":
            winner = "a" if order == "ab" else "b"
        elif raw_winner == "second":
            winner = "b" if order == "ab" else "a"
        else:
            winner = "tie"

        return Verdict(
            case_id=case.id, mode=JudgeMode.PAIRWISE, per_criterion=per_criterion,
            overall_score=float(data["overall_score"]),
            overall_rationale=str(data.get("overall_rationale", "")),
            winner=winner, flags=list(data.get("flags", [])),
            judge_model=self.provider.name, order=order, tokens=tokens, latency_ms=latency,
            raw_response=raw, parse_retries=retries,
        )

    def judge_pairwise_both_orders(self, case: TestCase) -> PairwiseResult:
        """The position-bias control: run the SAME pair in both orders and
        reconcile. This is non-negotiable for pairwise mode."""
        v_ab = self._judge_pairwise_one_order(case, "ab")
        v_ba = self._judge_pairwise_one_order(case, "ba")

        winner_ab = v_ab.winner
        winner_ba = v_ba.winner

        if v_ab.is_judge_error or v_ba.is_judge_error:
            reconciled: str = "inconsistent"
            flipped = False
        elif winner_ab == winner_ba:
            reconciled = winner_ab or "tie"
            flipped = False
        else:
            # Orders disagree on the winner -> don't trust either single
            # order; report as inconsistent rather than picking one.
            reconciled = "inconsistent"
            flipped = True

        return PairwiseResult(
            case_id=case.id, verdict_ab=v_ab, verdict_ba=v_ba,
            winner_ab=winner_ab, winner_ba=winner_ba,
            flipped=flipped, reconciled_winner=reconciled,  # type: ignore[arg-type]
        )


def ensemble_reconcile(per_judge_results: list[PairwiseResult]) -> str:
    """
    Majority vote across MULTIPLE judges' (already both-orders-reconciled)
    winners for the same case. This is the "(or an ensemble)" mitigation for
    self-enhancement bias: if judges are drawn from different families with
    different (or no) style affinities, their individual biases don't all
    point the same direction, so voting damps out any single judge's
    self-enhancement skew rather than requiring you to trust one judge's
    family choice completely.
    """
    votes = [r.reconciled_winner for r in per_judge_results if r.reconciled_winner in ("a", "b", "tie")]
    if not votes:
        return "inconsistent"
    counts = Counter(votes)
    top_winner, top_count = counts.most_common(1)[0]
    # A genuine tie in the vote itself (e.g. 1-1 between two judges) is
    # reported as "tie" rather than arbitrarily picking the first counted.
    if list(counts.values()).count(top_count) > 1:
        return "tie"
    return top_winner


def ensemble_judge_suite(judges: list[Judge], cases: list[TestCase]) -> list[str]:
    """Runs every case through every judge (each doing its own both-orders
    reconciliation) and returns the ensemble-voted winner per case."""
    combined = []
    for case in cases:
        per_judge = [j.judge_pairwise_both_orders(case) for j in judges]
        combined.append(ensemble_reconcile(per_judge))
    return combined
