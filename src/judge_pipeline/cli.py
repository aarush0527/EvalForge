"""
Command-line entry point.

    python -m judge_pipeline run      --suite ... --judge-config ... --out ...
    python -m judge_pipeline compare  --suite ... --judge-config ... --out ...
    python -m judge_pipeline validate --gold ... --adversarial ... --judge-config ... --out ...
    python -m judge_pipeline bias-experiment --suite ... --adversarial ...
                                    --hardened-config ... --naive-config ... --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as cfgmod
from .aggregate import aggregate_pairwise, aggregate_pointwise
from .cost import load_price_table
from .suites import load_adversarial_pairs, load_suite
from .validation import evaluate_adversarial_probes, evaluate_gold_agreement, evaluate_test_retest


def _write_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def cmd_run(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    judge = cfgmod.build_judge(args.judge_config, rubric_path=args.rubric,
                                audit_log_path=args.audit_log)
    verdicts = [judge.judge_pointwise(c) for c in suite.cases]
    price_table = load_price_table(args.pricing)
    generator_family = None
    if args.generator_config:
        generator_family = cfgmod.provider_family(args.generator_config)
    report = aggregate_pointwise(
        suite.name, suite.cases, verdicts,
        judge_family=judge.provider.family, generator_family=generator_family,
        price_table=price_table,
    )
    _write_json(args.out, report.model_dump(mode="json"))
    print(f"[run] {suite.name}: n={report.n_cases} errors={report.n_errors} "
          f"pass_rate={report.pass_rate} mean_score={report.mean_overall_score} "
          f"-> {args.out}")


def cmd_compare(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    judge = cfgmod.build_judge(args.judge_config, rubric_path=args.rubric,
                                audit_log_path=args.audit_log)
    results = [judge.judge_pairwise_both_orders(c) for c in suite.cases]
    price_table = load_price_table(args.pricing)
    fam_a = cfgmod.provider_family(args.generator_config_a) if args.generator_config_a else None
    fam_b = cfgmod.provider_family(args.generator_config_b) if args.generator_config_b else None
    report = aggregate_pairwise(
        suite.name, suite.cases, results,
        judge_family=judge.provider.family, generator_family_a=fam_a, generator_family_b=fam_b,
        price_table=price_table,
    )
    _write_json(args.out, report.model_dump(mode="json"))
    print(f"[compare] {suite.name}: n={report.n_cases} win_rate_a={report.win_rate_a} "
          f"win_rate_b={report.win_rate_b} flip_rate={report.flip_rate} "
          f"winner={report.winner} (p={report.winner_p_value}) -> {args.out}")


def cmd_validate(args: argparse.Namespace) -> None:
    judge = cfgmod.build_judge(args.judge_config, rubric_path=args.rubric,
                                audit_log_path=args.audit_log)

    gold_suite = load_suite(args.gold)
    gold_report = evaluate_gold_agreement(judge, gold_suite.cases)

    retest_cases = gold_suite.cases[: args.retest_n_cases]
    retest_report = evaluate_test_retest(judge, retest_cases, n_reruns=args.retest_reruns)

    adv_report = None
    if args.adversarial:
        pairs = load_adversarial_pairs(args.adversarial)
        adv_report = evaluate_adversarial_probes(judge, pairs)

    out = {
        "judge_model": judge.provider.name,
        "mitigations": judge.mitigations,
        "gold_agreement": gold_report.model_dump(mode="json"),
        "test_retest": retest_report.model_dump(mode="json"),
        "adversarial_probes": adv_report.model_dump(mode="json") if adv_report else None,
    }
    _write_json(args.out, out)
    print(f"[validate] kappa={gold_report.kappa_pass_fail} "
          f"pearson={gold_report.pearson_score} "
          f"retest_flip_rate={retest_report.pass_fail_flip_rate} "
          f"adversarial_fooled_rate={adv_report.fooled_rate if adv_report else None} "
          f"adversarial_mean_margin={adv_report.mean_margin if adv_report else None} "
          f"-> {args.out}")


def cmd_bias_experiment(args: argparse.Namespace) -> None:
    """Runs the SAME suites through a naive judge config and a hardened one,
    and reports the before/after difference -- this is what backs the
    README's Discussion section with real numbers."""
    suite = load_suite(args.suite)
    pairs = load_adversarial_pairs(args.adversarial) if args.adversarial else []

    out = {}
    for label, cfg_path in (("naive", args.naive_config), ("hardened", args.hardened_config)):
        judge = cfgmod.build_judge(cfg_path, rubric_path=args.rubric,
                                     audit_log_path=f"logs/bias_experiment_{label}_audit.jsonl")
        results = [judge.judge_pairwise_both_orders(c) for c in suite.cases]
        report = aggregate_pairwise(suite.name, suite.cases, results, judge_family=judge.provider.family)

        adv_report = evaluate_adversarial_probes(judge, pairs) if pairs else None

        out[label] = {
            "judge_model": judge.provider.name,
            "flip_rate": report.flip_rate,
            "positional_first_win_rate": report.positional_first_win_rate,
            "win_rate_a": report.win_rate_a,
            "win_rate_b": report.win_rate_b,
            "score_stddev": report.score_stddev,
            "adversarial_fooled_rate": adv_report.fooled_rate if adv_report else None,
            "adversarial_mean_margin": adv_report.mean_margin if adv_report else None,
            "full_report": report.model_dump(mode="json"),
            "full_adversarial": adv_report.model_dump(mode="json") if adv_report else None,
        }

    _write_json(args.out, out)
    n = out["naive"]
    h = out["hardened"]
    print("[bias-experiment] naive vs hardened, same suites:")
    print(f"  flip_rate:                naive={n['flip_rate']}  hardened={h['flip_rate']}")
    print(f"  positional_first_win_rate: naive={n['positional_first_win_rate']}  hardened={h['positional_first_win_rate']}")
    print(f"  score_stddev:              naive={n['score_stddev']}  hardened={h['score_stddev']}")
    print(f"  adversarial_fooled_rate:   naive={n['adversarial_fooled_rate']}  hardened={h['adversarial_fooled_rate']}")
    print(f"  adversarial_mean_margin:   naive={n['adversarial_mean_margin']}  hardened={h['adversarial_mean_margin']}")
    print(f"  -> {args.out}")


def cmd_self_enhancement_experiment(args: argparse.Namespace) -> None:
    """
    Same-family judge vs cross-family judge vs a 2-judge ensemble, on a
    suite where output_a/output_b are quality-matched and differ only in a
    synthetic style marker. Isolates self-enhancement bias as the one
    variable under test -- see suites/self_enhancement_suite.yaml.
    """
    from .judge import ensemble_judge_suite

    suite = load_suite(args.suite)

    same_judge = cfgmod.build_judge(args.same_family_config, rubric_path=args.rubric,
                                      audit_log_path="logs/self_enhancement_same_family_audit.jsonl")
    cross_judge = cfgmod.build_judge(args.cross_family_config, rubric_path=args.rubric,
                                       audit_log_path="logs/self_enhancement_cross_family_audit.jsonl")

    same_results = [same_judge.judge_pairwise_both_orders(c) for c in suite.cases]
    cross_results = [cross_judge.judge_pairwise_both_orders(c) for c in suite.cases]

    def _rates(results):
        usable = [r for r in results if r.reconciled_winner in ("a", "b", "tie")]
        n = len(usable) or 1
        return {
            "win_rate_a": sum(1 for r in usable if r.reconciled_winner == "a") / n,
            "win_rate_b": sum(1 for r in usable if r.reconciled_winner == "b") / n,
            "tie_rate": sum(1 for r in usable if r.reconciled_winner == "tie") / n,
        }

    out = {
        "n_cases": len(suite.cases),
        "same_family": {"judge_model": same_judge.provider.name, **_rates(same_results)},
        "cross_family": {"judge_model": cross_judge.provider.name, **_rates(cross_results)},
    }

    if args.beta_family_config:
        beta_judge = cfgmod.build_judge(args.beta_family_config, rubric_path=args.rubric,
                                          audit_log_path="logs/self_enhancement_beta_family_audit.jsonl")
        ensemble_winners = ensemble_judge_suite([same_judge, beta_judge], suite.cases)
        n = len(ensemble_winners) or 1
        out["ensemble_alpha_plus_beta"] = {
            "judges": [same_judge.provider.name, beta_judge.provider.name],
            "win_rate_a": ensemble_winners.count("a") / n,
            "win_rate_b": ensemble_winners.count("b") / n,
            "tie_rate": ensemble_winners.count("tie") / n,
        }

    _write_json(args.out, out)
    print("[self-enhancement-experiment] output_a is alpha-styled, output_b is beta-styled;")
    print("both sides are quality-matched, so any skew away from 50/50 is family bias:")
    print(f"  same-family judge (family=alpha):  win_rate_a={out['same_family']['win_rate_a']:.3f}  "
          f"win_rate_b={out['same_family']['win_rate_b']:.3f}")
    print(f"  cross-family judge (family=gamma): win_rate_a={out['cross_family']['win_rate_a']:.3f}  "
          f"win_rate_b={out['cross_family']['win_rate_b']:.3f}")
    if "ensemble_alpha_plus_beta" in out:
        e = out["ensemble_alpha_plus_beta"]
        print(f"  ensemble (alpha-judge + beta-judge, voted): win_rate_a={e['win_rate_a']:.3f}  "
              f"win_rate_b={e['win_rate_b']:.3f}")
    print(f"  -> {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="judge_pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Pointwise-score a suite against the rubric.")
    r.add_argument("--suite", required=True)
    r.add_argument("--judge-config", required=True)
    r.add_argument("--rubric", default=None)
    r.add_argument("--generator-config", default=None)
    r.add_argument("--pricing", default=None)
    r.add_argument("--audit-log", default="logs/run_audit.jsonl")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="Pairwise-compare output_a vs output_b, both orders.")
    c.add_argument("--suite", required=True)
    c.add_argument("--judge-config", required=True)
    c.add_argument("--rubric", default=None)
    c.add_argument("--generator-config-a", default=None)
    c.add_argument("--generator-config-b", default=None)
    c.add_argument("--pricing", default=None)
    c.add_argument("--audit-log", default="logs/compare_audit.jsonl")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)

    v = sub.add_parser("validate", help="Gold agreement + test-retest + adversarial probes.")
    v.add_argument("--gold", required=True)
    v.add_argument("--adversarial", default=None)
    v.add_argument("--judge-config", required=True)
    v.add_argument("--rubric", default=None)
    v.add_argument("--retest-n-cases", type=int, default=5)
    v.add_argument("--retest-reruns", type=int, default=3)
    v.add_argument("--audit-log", default="logs/validate_audit.jsonl")
    v.add_argument("--out", required=True)
    v.set_defaults(func=cmd_validate)

    b = sub.add_parser("bias-experiment", help="Naive vs hardened judge config, same suites.")
    b.add_argument("--suite", required=True)
    b.add_argument("--adversarial", default=None)
    b.add_argument("--naive-config", required=True)
    b.add_argument("--hardened-config", required=True)
    b.add_argument("--rubric", default=None)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_bias_experiment)

    se = sub.add_parser("self-enhancement-experiment",
                          help="Same-family vs cross-family (vs ensemble) judge, quality-matched suite.")
    se.add_argument("--suite", required=True)
    se.add_argument("--same-family-config", required=True)
    se.add_argument("--cross-family-config", required=True)
    se.add_argument("--beta-family-config", default=None, help="Enables the ensemble arm if given.")
    se.add_argument("--rubric", default=None)
    se.add_argument("--out", required=True)
    se.set_defaults(func=cmd_self_enhancement_experiment)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
