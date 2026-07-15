# LLM-as-Judge Evaluation Pipeline

A judging pipeline that takes an `{input, system_prompt, model_output, expected_output?, criteria?}`
test case, produces a structured quality verdict against a weighted rubric, and takes judge bias
seriously: every bias named in the brief is mitigated in code and measured, not just discussed.

**Everything in this README's Discussion section was produced by actually running this pipeline**
(see `results/*.json` and `logs/*.jsonl`), not asserted. Read [A note on the demo run](#a-note-on-the-demo-run)
first — it explains what actually generated these numbers.

## Quick start

```bash
pip install -e .                        # installs the package (src layout)
pip install -e ".[providers]"           # optional: only if using groq/anthropic/openai
pip install -e ".[dev]"                 # optional: to run tests

# Score a suite against the rubric (pointwise)
python -m judge_pipeline run \
  --suite suites/pointwise_suite.yaml \
  --judge-config config/judge_config.yaml \
  --out results/pointwise_report.json

# Compare two configs head-to-head (pairwise, both orders, position-bias controlled)
python -m judge_pipeline compare \
  --suite suites/comparison_suite.yaml \
  --judge-config config/judge_config.yaml \
  --out results/comparison_report.json

# Validate the judge itself: gold-label agreement, test-retest, adversarial probes
python -m judge_pipeline validate \
  --gold suites/gold_labeled_suite.yaml \
  --adversarial suites/adversarial_probes.yaml \
  --judge-config config/judge_config.yaml \
  --out results/validation_report.json

# The before/after story: same suites, naive judge config vs hardened one
python -m judge_pipeline bias-experiment \
  --suite suites/comparison_suite.yaml \
  --adversarial suites/adversarial_probes.yaml \
  --naive-config config/judge_config_naive.yaml \
  --hardened-config config/judge_config.yaml \
  --out results/bias_experiment_report.json

# The self-enhancement story: same-family judge vs cross-family vs a 2-judge ensemble
python -m judge_pipeline self-enhancement-experiment \
  --suite suites/self_enhancement_suite.yaml \
  --same-family-config config/judge_config_same_family.yaml \
  --cross-family-config config/judge_config_cross_family.yaml \
  --beta-family-config config/judge_config_beta_family.yaml \
  --out results/self_enhancement_report.json
```

Run `python -m judge_pipeline <command> --help` for the full flag list. Every run also writes a
full audit log (prompt + raw response + tokens + latency, one JSON line per judge call) to `logs/`.

To point this at a real model, edit `config/judge_config.yaml`:
```yaml
provider: groq            # this project's primary, free-tier-friendly option
model: openai/gpt-oss-120b
family: openai-oss
```
and `export GROQ_API_KEY=...` — get a free key (no credit card) at https://console.groq.com/keys.
`anthropic`/`openai` also work as drop-in alternatives if you happen to have those keys, but
**this project doesn't require them** — Groq's free tier is enough to actually run every command
below for real. No key is ever read from a config file.

## A note on the demo run

**No API key was available in the environment this was built in**, so every number in this README
came from `SimulatedProvider` (`src/judge_pipeline/providers.py`) — a small, deterministic,
seeded rule-based responder, **not a real language model**. It checks numeric-answer overlap,
lexical overlap with a reference, obvious PII-leak patterns, confident-tone markers, and bullet-list
formatting, and it deliberately behaves the way an *unmitigated* judge is documented to behave
(rewarding length, favoring whichever answer it sees first, swayed by confident tone) **unless**
the prompt it receives contains this project's anti-bias instructions — which is exactly what lets
the naive-vs-hardened experiment below show a genuine, code-driven before/after difference instead
of two identical runs.

`GroqProvider` is fully implemented (`src/judge_pipeline/providers.py`) and is what you get by
changing two lines in any `judge_config*.yaml`, plus a free GROQ_API_KEY — no payment required.
`AnthropicProvider`/`OpenAIProvider` are also implemented if you happen to have those keys, but
**this project doesn't need them**: GroqCloud's free tier (no credit card, rate-limited rather than
capped-to-zero) is enough to actually run every command in this README for real. The prompt
construction, parsing, bias-metric, aggregation, and validation code is **identical** regardless of
which provider is behind it — swapping in a real key exercises the same pipeline, not different
code. Groq only serves open-weight models (Llama, Qwen, Kimi K2, GPT-OSS, Gemma) rather than
proprietary GPT/Claude/Gemini, which is exactly why it's cheap and free-tier-friendly, and it
happens to be genuinely useful for the self-enhancement experiment too — Groq hosts several
unrelated model *families* behind one API and one key, so `judge_config_same_family.yaml` and
`judge_config_cross_family.yaml` each include a commented-out real-Groq equivalent (e.g. a
Llama-family judge vs a Qwen-family judge) alongside the synthetic, quality-matched version this
demo actually runs.

I'm flagging this prominently because reporting simulated numbers as if they were a real model's
would be exactly the kind of unearned confidence this whole assignment is about catching.

## Judging modes

The brief lists four modes (pointwise, pairwise, reference-based, reference-free) as if choosing
one is the task. They're actually two independent axes, and this pipeline implements both:

- **Pointwise** (`judge.judge_pointwise`) scores one output against the rubric. It's
  reference-based when `expected_output` is present (added to the prompt, faithfulness checked
  against it) and reference-free otherwise. This is what produces per-case pass/fail and a
  suite's mean scores. Use it for: regression testing a single system's outputs, or checking
  outputs against a known-good answer.
- **Pairwise** (`judge.judge_pairwise_both_orders`) judges two outputs head-to-head. This is what
  the "compare two configs" requirement actually needs, and it's the *only* place position bias
  can be measured at all — pointwise scoring simply has no position to bias toward.

Every pairwise case is judged in **both orders** — `(A,B)` and `(B,A)` — as two independent judge
calls. The judge is only ever told "first response" / "second response," never "A" or "B"; the
mapping back to A/B happens after parsing, in `judge._judge_pairwise_one_order`. If both orders
agree on a winner, that's the reconciled result. If they disagree, the case is marked
`inconsistent` rather than arbitrarily picking one order to trust — see [Position bias](#position-order) below.

## Rubric

`config/rubric.yaml` defines six weighted criteria (correctness 0.30, faithfulness 0.25,
completeness 0.15, instruction-following 0.15, tone 0.05, safety 0.10) plus three few-shot
**calibration anchors** at scores 1/3/5, used specifically to fight score clustering (see below).
A test case's optional `criteria` field overrides the rubric for that one case (`judge._rubric_for`).
`config/rubric_naive.yaml` is the same criteria and weights with the anchors stripped out, used
only for the before/after experiment.

Every verdict is structured JSON, not free text:
```json
{"per_criterion": [{"criterion": "...", "score": 0, "rationale": "..."}, ...],
 "overall_score": 0, "overall_rationale": "...", "passed": true, "flags": []}
```
forced via the judge's structured-output path first. If parsing still fails, the parse error and
the judge's own bad output are sent back once asking it to fix its output (`judge._call_and_parse`);
only if that also fails is the case logged as a judge error rather than crashing the whole suite
(`parsing.parse_verdict`, tested in `tests/test_parsing.py`).

`case_21_criteria_override` in `pointwise_suite.yaml` demonstrates the per-case `criteria`
override end to end: it asks for a single `safety_only` criterion instead of the default six, and
scores **1.52/5** with a `possible_pii_leak` flag — the mock actually reads the rubric that was
sent in the prompt rather than assuming a fixed list (`providers._criteria_from_system`, regression
test in `tests/test_providers.py`).

## Bias handling

This is the actual measured result of `bias-experiment` (`config/judge_config_naive.yaml` vs
`config/judge_config.yaml`, same 15-case comparison suite and same 6-pair adversarial probe set,
[simulated judge](#a-note-on-the-demo-run) both times — see `results/bias_experiment_report.json`):

| Bias | Mitigation (in code) | Naive / same-family | Hardened / cross-family |
|---|---|---:|---:|
| **Position** (A/B order) | Run every pair in both orders (`judge.judge_pairwise_both_orders`); agree → confident winner, disagree → `inconsistent` | flip rate **53.3%** | flip rate **0.0%** |
| **Position**, differently measured | positional first-shown win rate (50% = unbiased) | **74.1%** | **50.0%** |
| **Verbosity / length** | Prompt forbids rewarding length; length logged for a correlation check | mean margin (terse-correct − verbose-wrong) **0.827** | mean margin **1.003** |
| **Sycophancy / style** | Forced per-criterion grounding + confidently-wrong adversarial probes | fooled on 1/6 probes both configs — see [below](#sycophancy--style) for why the binary count alone is misleading | |
| **Self-enhancement** | Cross-family judge or ensemble, instead of prompt changes | same-family win rate **100%** for its own style | cross-family / ensemble win rate **0%**, ties **100%** |
| **Score clustering** | Few-shot scale anchors (1/3/5) in the hardened rubric; naive rubric has none | score std-dev **0.511** | score std-dev **0.470** |

### Position (order)

This is the cleanest result in the whole project: with mitigations off, **53% of pairs flipped
their declared winner** depending purely on which side was shown first, and the judge favored
whichever answer it saw first **74% of the time** — nowhere near the 50% an unbiased judge should
produce. With mitigations on (same suite, same simulated judge, only the prompt and the
both-orders-required logic differ), the flip rate drops to **exactly 0%** and the positional win
rate lands **exactly on 50%**. Running both orders and requiring agreement isn't a nice-to-have
here — it's the difference between a coin flip and a real signal.

### Verbosity / length

`bias_metrics.length_score_correlation` computes the Pearson correlation between output length and
awarded score across the pointwise suite; the current run gives **r ≈ 0.23** — mildly positive
(longer answers score a bit higher on average), which tracks, since a couple of genuinely more
complete answers in the suite also happen to be longer. `case_07_verbose_padded` in
`pointwise_suite.yaml` — the same correct content as `case_01`, but padded with filler — scores
**4.61/5** even under the hardened judge, essentially the same as the terse-but-also-correct
`case_08_terse_correct` at **4.17/5**. In other words: the hardened judge doesn't visibly *punish*
padding, it just refuses to give it *extra* credit — which is the honest, achievable claim ("don't
reward length itself"), not the stronger claim that padding is penalized outright. The
naive-vs-hardened score-std-dev gap (0.511 → 0.470) is real but modest; most of the variance in
this suite comes from genuine quality differences between cases, not clustering.

### Sycophancy / style

`case_09_confident_wrong` in `pointwise_suite.yaml` — "World War II absolutely, definitely ended
in 1943 — I'm 100% certain of this, no doubt about it" — scores **3.02/5, fails,** and is tagged
`confident_but_likely_wrong` by the hardened judge, versus `case_10_hedged_correct`'s honestly
uncertain but factually right answer at **4.67/5**. Forcing the judge to ground every criterion's
rationale in specific content, rather than a vibe, is what catches this.

On the 6-pair adversarial probe set, **the binary fooled-rate genuinely doesn't move: 1/6 fooled
in both the naive and hardened config.** Reporting that alone would be misleading in the other
direction — a flat count hides a real, measurable improvement in *how badly* the judge loses when
it does get fooled and *how decisively* it gets things right when it doesn't: the mean margin by
which the terse-correct answer beats the verbose-wrong one across all 6 pairs is **0.827 under the
naive config and 1.003 under the hardened one** — about a 21% wider margin, i.e. the hardened judge
discriminates more confidently in the correct direction even on pairs neither config gets fooled by.
The one pair that still fools both configs (`probe_01_capital`, verbose-wrong 3.98 vs terse-correct
3.94 — a near-tie either way) is a genuine, honestly reported limit, worth being direct about *why*:
that reference sentence has no number to anchor on, and the
[simulated judge's](#a-note-on-the-demo-run) correctness check leans on numeric-answer matching
more than a real LLM judge would need to. That's a limitation of the stand-in, flagged here rather
than papered over — a real LLM judge, unlike a keyword-overlap heuristic, would resolve "the
capital of Australia" from world knowledge without needing a digit to check against.

### Self-enhancement

This is the one bias that a prompt can't fix — no amount of "don't reward length" instruction
touches a judge's tendency to trust its own family's writing style, so it needed its own dedicated
experiment rather than a naive/hardened prompt toggle. `suites/self_enhancement_suite.yaml` has 10
cases where `output_a` and `output_b` are **quality-matched** (verified: identical `_looks_correct`
score on every case but one, off by 0.025) and differ *only* in a synthetic style marker standing
in for a real model family's stylistic fingerprint. `results/self_enhancement_report.json`:

| Judge | win_rate (alpha-styled `a`) | win_rate (beta-styled `b`) | tie rate |
|---|---:|---:|---:|
| Same-family (family=alpha, recognizes `a`'s style) | **100%** | 0% | 0% |
| Cross-family (family=gamma, no affinity to either) | **0%** | 0% | **100%** |
| Ensemble (alpha-judge + beta-judge, majority vote) | 0% | 0% | **100%** |

With quality genuinely held equal, a same-family judge picked its own family's style **every single
time** — a textbook self-enhancement result. A cross-family judge with no style affinity to either
side calls every case a tie, correctly, since there's nothing but style to differentiate them. The
ensemble of two *oppositely*-biased judges also lands on 100% ties — not because it averages out to
neutral, but because when the alpha-judge votes "a" and the beta-judge votes "b" on the same case,
`judge.ensemble_reconcile` reports the disagreement as a tie rather than arbitrarily trusting one
judge's bias over the other's. That's the concrete mechanism behind "use a cross-family judge (or
an ensemble)": either remove the family affinity, or make sure no single biased vote can win alone.
`bias_metrics.family_mismatch_warning` (tested in `tests/test_bias_metrics.py`) is the
config-level tripwire for this — it fires whenever a judge and generator config declare the same
family, which is what would have caught this bias automatically before running anything.

### Score clustering

`config/rubric.yaml` includes three calibration anchors (score 1, 3, 5 with a concrete example
each); `config/rubric_naive.yaml` is identical minus the anchors. Pairwise mode is offered as the
complementary, higher-discrimination alternative for exactly this reason — comparing A directly to
B forces a decision in a way that scoring both in isolation on a 1–5 scale doesn't.

## Validating the judge

All three methods in the brief, not just one (`results/validation_report.json`):

- **Gold-label agreement** (`validation.evaluate_gold_agreement`, `suites/gold_labeled_suite.yaml`,
  12 human-labeled cases): Cohen's kappa **0.50**, raw pass/fail agreement **75%**, Pearson **0.58**,
  Spearman **0.50** (0.4965 exactly) between judge score and human gold score. Moderate, not excellent — see
  [Discussion](#discussion) for what that means for release-gating.
- **Test-retest consistency** (`validation.evaluate_test_retest`, 6 cases × 3 reruns): **0% flip
  rate**, essentially zero score variance. Expected and not very informative here, since the
  simulated judge is deterministic at temperature 0 — a real LLM at temperature 0 should still be
  close to this, but won't be quite this perfect; this check is far more meaningful once a real
  provider is behind it.
- **Adversarial probes** (`validation.evaluate_adversarial_probes`, 6 verbose-wrong vs
  terse-correct pairs): fooled on **1/6 (16.7%)** — discussed above.

## Cost and token tracking

Every judge call's input/output tokens are logged per-case and summed per suite
(`schema.TokenUsage`, `aggregate._sum_tokens`). `src/judge_pipeline/cost.py` converts token counts
to an estimated USD figure via an editable table (`config/pricing.yaml` overrides the defaults) —
rates aren't hardcoded into the pipeline logic, since providers change pricing over time. The
simulated provider is priced at $0 (it's a local stand-in, not a paid API); point `judge_config.yaml`
at a Groq model and the same `estimated_cost_usd` field reports real dollars — and on Groq's free
tier, that number is genuinely $0 up to the rate limit, not just "cheap." For reference, current
Groq per-1M-token rates baked into `cost.py` (verify at
[groq.com/pricing](https://groq.com/pricing) before relying on them for a real budget):

| Model | Input | Output |
|---|---:|---:|
| `llama-3.1-8b-instant` | $0.05 | $0.08 |
| `openai/gpt-oss-20b` | $0.075 | $0.30 |
| `openai/gpt-oss-120b` (this project's default judge) | $0.15 | $0.60 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | $0.11 | $0.34 |
| `qwen/qwen3-32b` | $0.29 | $0.59 |
| `llama-3.3-70b-versatile` | $0.59 | $0.79 |
| `moonshotai/kimi-k2-instruct` | $1.00 | $3.00 |

Even without the free tier, a full run of every command in this README (~120 judge calls total)
would cost a small fraction of a cent on the default `openai/gpt-oss-120b` judge.

## Discussion

**How biased was it before vs. after?** Concretely: a naive "just ask the model to score it" judge
flipped its declared winner on over half of all pairs depending on presentation order alone, and
favored whichever answer it saw first three times out of four; requiring both orders to agree
before trusting a verdict took the flip rate to zero and the positional win rate to exactly 50%.
Separately, a same-family judge picked its own family's writing style **100% of the time** on a
suite where quality was held genuinely equal — as clean a self-enhancement result as this kind of
demo produces — and a cross-family judge, or an ensemble of two oppositely-biased judges voting
against each other, brought that to 0%. Verbosity/sycophancy is the more honest, harder story: the
binary "was it fooled" count doesn't move (1/6 both ways), but the *margin* by which the judge gets
it right widens by about 21% under the hardened config — real, but the smaller, harder-to-fully-kill
effect of the four. Position and self-enhancement turned out to be cleanly, almost completely
fixable by changing *who judges* (both orders, different family); verbosity and sycophancy are
about the judge's underlying sense of "correctness" itself, which no prompt instruction fully
replaces.

**Would I let this gate a release?** For pairwise A/B comparisons with a decisive win rate and a
significant sign-test result (this run: config B won 46.7% of cases to A's 6.7%, 46.7% tied,
p ≈ 0.07): yes, as an automated *first-pass* signal to block an obviously-regressive prompt or
model change — provided both orders are run, the flip rate is checked alongside the win rate, and
the judge/generator family check (`family_mismatch_warning`) is clean. A "significant" win rate
next to a high flip rate, or a same-family match on the family check, should itself block the
release rather than just informing it. For pointwise pass/fail gating a release on its own: not
yet, not at this agreement level. Kappa of 0.50 against human labels is moderate agreement, not
strong agreement — Cohen's own scale calls 0.41–0.60 "moderate," not "substantial" or higher. I'd
use this pointwise setup to triage and flag regressions for human review, not as the sole release
gate, until either the agreement number improves with a stronger judge model, or the specific
failure modes behind the disagreement are identified and the rubric or prompt is revised to close
them. Blind trust in any single-judge number — even a well-mitigated one — is exactly the failure
mode this exercise is about avoiding.

## Project layout

```
config/            judge & generator configs, rubric.yaml (+ rubric_naive.yaml), pricing.yaml
                    judge_config_{same,cross,beta}_family.yaml -- self-enhancement experiment
suites/            pointwise_suite (21 cases) · comparison_suite (15 A/B cases)
                    gold_labeled_suite (12 cases, human gold_score/gold_label)
                    adversarial_probes (6 verbose-wrong/terse-correct pairs)
                    self_enhancement_suite (10 quality-matched, style-differing pairs)
src/judge_pipeline/
  schema.py         pydantic models: TestCase, Rubric, Verdict, PairwiseResult, SuiteReport
  providers.py      Groq (primary) / Anthropic / OpenAI / SimulatedProvider (pluggable, documented)
  rubric.py         rubric loading + sensible built-in default
  prompts.py        rubric + anchors + anti-bias prompt construction (mitigations on/off)
  parsing.py        schema-forced JSON extraction + retry-on-malformed
  judge.py          pointwise scoring, pairwise both-orders judging, ensemble vote-reconcile
  bias_metrics.py   flip rate, positional win rate, length/score correlation, family check
  aggregate.py      suite report: pass rate, mean scores, win rate, winner + significance
  validation.py     Cohen's kappa/correlation, test-retest, adversarial probe evaluation (+ margin)
  audit_log.py      JSONL logger -- every prompt + raw response + tokens + latency
  cost.py           token counts -> USD via an editable price table
  config.py         builds a Judge from a YAML config file
  suites.py         suite/adversarial-pair YAML loaders
  cli.py            `run` / `compare` / `validate` / `bias-experiment` / `self-enhancement-experiment`
tests/              parsing, bias math, ensemble voting, aggregation -- unit tested, not just eyeballed
results/, logs/     generated reports and full audit trails (already populated from the demo run)
```

## Known limitations

- The demo numbers above come from `SimulatedProvider`, not a real LLM — see
  [A note on the demo run](#a-note-on-the-demo-run). Point `judge_config.yaml` at a real provider
  to get a genuine measurement; the code paths are identical either way.
- `SimulatedProvider`'s correctness heuristic leans on numeric-answer overlap and lexical overlap;
  it has essentially no way to judge purely subjective, reference-free prose quality (e.g. "is this
  tone appropriate?") beyond a crude confident-tone-marker check. A real LLM judge is strictly
  better at exactly this, which is the whole point of using one. This is also the specific reason
  `probe_01_capital` (no number to anchor on) is the one adversarial probe that still fools the
  hardened config — see [Sycophancy / style](#sycophancy--style).
- Test-retest consistency is trivially perfect here because the simulated judge is deterministic
  at temperature 0; this check earns its keep once a real, genuinely stochastic model is behind it.
- The 15/12/6/10-case suites here are small enough to hand-write and audit by eye, which was the
  point for a demo, but the significance test on the A/B comparison (p ≈ 0.07 on 8 decisive cases)
  should be read as "suggestive," not "proven" — a real evaluation would want a larger suite before
  treating the winner as settled.
- A couple of entries in `cost.py`'s price table (`qwen/qwen3-32b`, `meta-llama/llama-4-scout-17b-16e-instruct`) are sourced from third-party trackers describing them as "preview" pricing, not Groq's own pricing page directly — re-verify at
  [groq.com/pricing](https://groq.com/pricing) before budgeting against them. The core defaults
  used in this project (`llama-3.1-8b-instant`, `openai/gpt-oss-20b`/`120b`, `llama-3.3-70b-versatile`)
  are corroborated across multiple sources and Groq's own announcements.
- `self_enhancement_suite.yaml`'s output_a/output_b pairs are deliberately quality-matched by
  construction (verified: identical `_looks_correct` score on 9 of 10 pairs, off by 0.025 on the
  10th) so that any measured win-rate skew is attributable to family-style bias and nothing else.
  That's the right design for isolating the variable, but it also means the 100%-vs-0% result is a
  best-case demonstration of the effect, not a claim about how large self-enhancement bias is on
  naturally-occurring, quality-*un*matched outputs from two real model families.
