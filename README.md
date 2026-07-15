# EvalForge

> A production-ready LLM evaluation framework for structured, bias-aware assessment of Large Language Model outputs.

EvalForge is a modular evaluation pipeline that automates the assessment of LLM-generated responses using configurable LLM judges, explicit rubrics, and reproducible evaluation workflows. It supports structured pointwise and pairwise judging, comprehensive validation, adversarial testing, and systematic bias analysis to measure evaluation reliability beyond simple benchmark scores.

Unlike conventional evaluation scripts that produce a single numerical score, EvalForge focuses on **evaluation quality itself** by measuring agreement, consistency, robustness, and common judge biases before relying on automated decisions.

---

## Overview

Modern AI systems are increasingly evaluated using other language models. While this approach dramatically reduces the need for manual annotation, LLM judges are themselves imperfect—they can exhibit position bias, verbosity bias, stylistic preference, score clustering, and model-family bias.

EvalForge provides a configurable evaluation pipeline that addresses these issues through structured judging prompts, explicit rubrics, validation experiments, and quantitative bias measurements.

The framework supports both simulated judges for controlled experimentation and real LLM providers for production-style evaluation.

---

# Features

## Structured Evaluation

- Pointwise quality scoring
- Pairwise A/B comparison
- Reference-based evaluation
- Reference-free evaluation
- Per-criterion scoring
- Structured JSON verdicts
- Robust JSON parsing and retry logic

---

## Configurable Rubrics

Every evaluation is performed against an explicit rubric rather than an opaque numerical score.

Example criteria include:

- Correctness
- Faithfulness
- Completeness
- Instruction Following
- Tone
- Safety

Each criterion produces:

- numerical score
- grounded rationale
- aggregated overall score

---

## Bias Mitigation

EvalForge explicitly measures and mitigates common LLM judge biases including:

- Position Bias
- Verbosity Bias
- Self-Enhancement Bias
- Style / Sycophancy Bias
- Score Clustering

Rather than simply discussing these biases, the framework implements mitigation strategies and reports quantitative measurements.

---

## Judge Validation

The framework includes multiple validation mechanisms:

- Agreement with gold labels
- Test-retest consistency
- Adversarial probe evaluation
- Reliability statistics
- Correlation metrics

This helps determine whether the judge itself is trustworthy enough for automated evaluation.

---

## Multi-Provider Architecture

Providers are interchangeable through configuration.

Supported providers include:

- Groq
- Simulated Provider
- Extensible provider interface for additional APIs

Judges and generators are configured independently.

---

## Complete Experiment Pipeline

The repository contains complete workflows for:

- Pointwise evaluation
- Pairwise comparison
- Validation experiments
- Bias experiments
- Self-enhancement experiments

Each workflow generates structured reports suitable for further analysis.

---

# Architecture

```
                  +--------------------+
                  | Test Suite         |
                  | JSON / YAML        |
                  +---------+----------+
                            |
                            |
                  +---------v----------+
                  | Prompt Builder     |
                  +---------+----------+
                            |
                            |
                  +---------v----------+
                  | Judge Provider     |
                  | Groq / Simulated   |
                  +---------+----------+
                            |
                            |
                  +---------v----------+
                  | Structured Verdict |
                  | JSON Parser        |
                  +---------+----------+
                            |
                            |
                  +---------v----------+
                  | Report Generator   |
                  +---------+----------+
                            |
          +-----------------+----------------+
          |                 |                |
          |                 |                |
  Pointwise Report   Comparison Report   Validation
```

---

# Repository Structure

```
EvalForge/

├── config/
│   ├── rubric.yaml
│   ├── judge_config.yaml
│   ├── judge_config_naive.yaml
│   └── ...
│
├── src/
│   └── judge_pipeline/
│       ├── cli.py
│       ├── judge.py
│       ├── providers.py
│       ├── prompts.py
│       ├── parser.py
│       ├── metrics.py
│       ├── schema.py
│       └── ...
│
├── suites/
│   ├── pointwise_suite.yaml
│   ├── comparison_suite.yaml
│   ├── gold_labeled_suite.yaml
│   ├── adversarial_probes.yaml
│   └── self_enhancement_suite.yaml
│
├── results/
│
├── tests/
│
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/EvalForge.git

cd EvalForge
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -e ".[providers]"
```

---

# Configuration

EvalForge uses YAML-based configuration files for all evaluation settings.

Example:

```yaml
provider: groq

model: openai/gpt-oss-120b

family: openai-oss

temperature: 0.0

max_tokens: 4096

mitigations: true

reasoning_effort: low
```

Provider credentials are supplied through environment variables.

```bash
export GROQ_API_KEY=YOUR_API_KEY
```

No API keys are stored in configuration files.

---

# Running Evaluations

## Pointwise Evaluation

```bash
python -m judge_pipeline run \
  --suite suites/pointwise_suite.yaml \
  --judge-config config/judge_config.yaml \
  --out results/pointwise_report.json
```

---

## Pairwise Comparison

```bash
python -m judge_pipeline compare \
  --suite suites/comparison_suite.yaml \
  --judge-config config/judge_config.yaml \
  --out results/comparison_report.json
```

---

## Judge Validation

```bash
python -m judge_pipeline validate \
  --gold suites/gold_labeled_suite.yaml \
  --adversarial suites/adversarial_probes.yaml \
  --judge-config config/judge_config.yaml \
  --out results/validation_report.json
```

---

## Bias Experiment

```bash
python -m judge_pipeline bias-experiment \
  --suite suites/comparison_suite.yaml \
  --adversarial suites/adversarial_probes.yaml \
  --naive-config config/judge_config_naive.yaml \
  --hardened-config config/judge_config.yaml \
  --out results/bias_experiment_report.json
```

---

## Self-Enhancement Experiment

```bash
python -m judge_pipeline self-enhancement-experiment \
  --suite suites/self_enhancement_suite.yaml \
  --same-family-config config/judge_config_same_family.yaml \
  --cross-family-config config/judge_config_cross_family.yaml \
  --beta-family-config config/judge_config_beta_family.yaml \
  --out results/self_enhancement_report.json
```

---

# Workflow

```
Test Suite
      │
      ▼
Prompt Construction
      │
      ▼
Judge Provider
      │
      ▼
Structured Verdict
      │
      ▼
Aggregation
      │
      ▼
Bias Analysis
      │
      ▼
Validation
      │
      ▼
Final Reports
```

# Evaluation Methodology

EvalForge evaluates both **model outputs** and the **judge itself**.

Rather than assuming an LLM judge is always reliable, the framework measures consistency, agreement, robustness, and common evaluation biases before trusting automated verdicts.

The evaluation pipeline consists of five complementary experiments.

---

# 1. Pointwise Evaluation

Pointwise evaluation scores each model output independently against a structured rubric.

Input:

```
{
    input,
    system_prompt,
    model_output,
    expected_output?,
    rubric
}
```

Output:

```
{
    per_criterion,
    overall_score,
    overall_rationale,
    passed,
    flags
}
```

Each criterion is scored independently before aggregation.

Advantages:

- Simple interpretation
- Stable quality scores
- Supports reference-based and reference-free judging
- Produces detailed rationales

Suitable for:

- Regression testing
- Quality monitoring
- Continuous evaluation

---

# 2. Pairwise Evaluation

Rather than assigning absolute scores, pairwise evaluation compares two candidate outputs for the same task.

Input:

```
Prompt

↓

Output A

↓

Output B
```

Judge output:

```
Winner

OR

Tie
```

Pairwise comparison is particularly useful when:

- comparing prompt versions
- comparing model checkpoints
- comparing retrieval strategies
- comparing decoding settings

Because humans are generally more consistent at comparative judgments than absolute scoring, pairwise evaluation often produces higher agreement.

---

# Structured Rubric

Instead of producing a single opaque score, EvalForge evaluates multiple dimensions independently.

| Criterion | Description |
|-----------|-------------|
| Correctness | Factual accuracy of the response |
| Faithfulness | Consistency with provided reference or context |
| Completeness | Whether important information is missing |
| Instruction Following | Compliance with user/system instructions |
| Tone | Appropriateness of writing style |
| Safety | Harmfulness, privacy, and policy compliance |

Each criterion includes:

- numerical score
- grounded rationale
- weighted contribution
- overall aggregation

This makes every evaluation transparent and auditable.

---

# Robust Structured Parsing

LLMs occasionally return malformed JSON.

EvalForge includes a structured parsing pipeline that:

- validates every response
- retries failed parses
- repairs malformed outputs
- logs raw responses
- tracks retry statistics

This allows evaluation to continue even when the underlying judge produces imperfect outputs.

---

# Auditability

Every evaluation stores:

- judge prompt
- raw judge response
- parsed verdict
- latency
- token usage
- retry count

This makes every decision reproducible and suitable for offline inspection.

Unlike black-box benchmark scores, every verdict can be replayed and inspected manually.

---

# Bias Mitigation

Automated judges are susceptible to systematic biases.

EvalForge implements both mitigation strategies and quantitative measurements for several well-known evaluation biases.

---

## Position Bias

### Problem

Many LLM judges prefer whichever answer appears first.

### Mitigation

Every comparison is evaluated twice.

```
A vs B

AND

B vs A
```

The framework then compares both verdicts and computes the **flip rate**, measuring how often the preferred answer changes when only the presentation order changes.

Lower flip rates indicate more stable evaluation.

---

## Verbosity Bias

### Problem

Longer responses often receive higher scores even when they contain redundant or unsupported information.

### Mitigation

The framework:

- includes verbose-but-wrong probes
- penalizes unsupported verbosity
- rewards concise correct answers
- explicitly instructs judges not to reward length

This helps separate informational quality from response length.

---

## Self-Enhancement Bias

### Problem

LLMs may prefer outputs that resemble responses produced by models from the same family.

### Mitigation

EvalForge supports:

- cross-family judges
- same-family judges
- ensemble judges

Synthetic experiments isolate stylistic preference while holding answer quality constant.

---

## Style / Sycophancy Bias

### Problem

Confident writing style can appear more convincing despite being incorrect.

### Mitigation

The judge must provide grounded rationales tied to specific evidence rather than confidence or fluency.

Adversarial probes intentionally contain confidently written but factually incorrect answers.

---

## Score Clustering

### Problem

LLMs often avoid using the extremes of scoring scales.

### Mitigation

Calibration anchors encourage consistent interpretation of rubric scores across evaluations.

This improves score separation and reduces central tendency bias.

---

# Judge Validation

Reliable evaluation requires validating the judge itself.

EvalForge includes three complementary validation strategies.

---

## Agreement with Gold Labels

Human-annotated evaluation cases are compared against judge verdicts.

Metrics include:

- Agreement Rate
- Cohen's Kappa
- Pearson Correlation
- Spearman Correlation

These measure alignment with expected judgments.

---

## Test–Retest Consistency

The same evaluation is repeated multiple times.

The framework reports:

- verdict flip rate
- score variance
- consistency across repeated runs

A stable judge should produce nearly identical outputs under identical conditions.

---

## Adversarial Probes

Purpose-built evaluation pairs attempt to fool the judge.

Examples include:

- verbose but incorrect
- terse but correct
- stylistically persuasive but false
- confidently incorrect answers

The framework reports:

- fooled rate
- average score margin
- per-case analysis

This provides a direct measure of judge robustness rather than relying on anecdotal observations.

# Experimental Results

The experiments below were conducted using the production judge configuration with the Groq provider and structured rubric-based evaluation.

---

# Pointwise Evaluation

Pointwise evaluation measures absolute response quality across the evaluation suite.

| Metric | Result |
|---------|--------|
| Evaluation Cases | 21 |
| Parse Errors | 0 |
| Pass Rate | 57.14% |
| Mean Overall Score | 3.64 / 5 |
| Score Standard Deviation | 1.57 |
| Mean Tone Score | 4.35 |
| Mean Safety Score | 4.80 |

The framework successfully generated structured verdicts for every evaluation without parser failures, demonstrating stable JSON generation and recovery logic.

---

# Pairwise Comparison

Pairwise evaluation compares two candidate responses for the same prompt.

| Metric | Result |
|---------|--------|
| Comparison Cases | 15 |
| A Win Rate | 23.33% |
| B Win Rate | 38.33% |
| Tie Rate | 38.33% |
| Flip Rate | 13.33% |
| Positional First Win Rate | 44.44% |
| Statistical Winner | Tie |
| p-value | 0.7266 |

The relatively low flip rate indicates that reversing presentation order has only a limited effect on evaluation outcomes.

---

# Judge Validation

The judge was validated against manually labeled examples.

## Agreement with Gold Labels

| Metric | Result |
|---------|--------|
| Cohen's Kappa | **1.00** |
| Pearson Correlation | **0.95** |
| Spearman Correlation | **0.81** |
| Raw Agreement | **100%** |

These results indicate extremely strong agreement between the automated judge and reference annotations.

---

## Test–Retest Reliability

The same evaluation cases were executed repeatedly to measure consistency.

| Metric | Result |
|---------|--------|
| Evaluation Cases | 5 |
| Re-runs per Case | 3 |
| Verdict Flip Rate | 0.00 |
| Mean Score Standard Deviation | 0.108 |

The low variance demonstrates highly repeatable scoring behavior.

---

# Adversarial Robustness

The framework evaluates whether the judge can distinguish concise correct answers from persuasive but incorrect responses.

Probe categories include:

- Verbose but incorrect
- Terse but correct
- Overconfident misinformation
- Style-based persuasion

| Metric | Result |
|---------|--------|
| Probe Pairs | 6 |
| Judge Fooled | 0 |
| Fooled Rate | **0%** |
| Mean Margin | 3.15 |

Across all adversarial probes, the judge consistently preferred the correct response despite stylistic distractions.

---

# Bias Experiments

EvalForge measures several known LLM judge biases.

## Position Bias

Every comparison is evaluated twice:

```
A → B

and

B → A
```

The disagreement between both evaluations produces the **flip rate**, which quantifies positional sensitivity.

A lower value indicates greater evaluation stability.

---

## Verbosity Bias

Longer responses are intentionally included even when factually incorrect.

The evaluation prompt explicitly instructs judges not to reward verbosity without additional information.

This discourages scoring based on response length rather than content quality.

---

## Self-Enhancement Bias

Synthetic "model families" are used to isolate stylistic preference while controlling for answer quality.

Three judge configurations are evaluated:

- Same-family judge
- Cross-family judge
- Ensemble judge

This experiment demonstrates how model-family affinity can influence automated evaluation.

---

# Engineering Highlights

EvalForge was designed with reproducibility and extensibility as primary goals.

Key implementation features include:

- Modular provider abstraction
- YAML-based configuration
- Structured prompt templates
- Automatic JSON validation
- Retry-based parser recovery
- Audit logging
- Token accounting
- Latency tracking
- Independent judge and generator configuration
- Reproducible evaluation suites

Every evaluation generates structured reports suitable for downstream analysis.

---

# Design Decisions

## Why Structured Rubrics?

Single-number evaluations provide little insight into *why* a response succeeded or failed.

EvalForge evaluates multiple quality dimensions independently before aggregation, making every verdict interpretable.

---

## Why Pairwise Evaluation?

Humans generally produce more reliable comparative judgments than absolute scores.

Supporting both pointwise and pairwise evaluation allows different benchmarking scenarios without changing the evaluation framework.

---

## Why Simulated Judges?

Certain bias experiments require controlling variables that cannot easily be isolated with real production models.

Synthetic providers make it possible to evaluate:

- position bias
- self-enhancement bias
- ensemble mitigation

under fully reproducible conditions.

---

## Why Real LLM Judges?

Production deployments require realistic evaluation behavior.

EvalForge supports production inference using Groq-hosted models while preserving the same evaluation pipeline and reporting infrastructure.

---

# Limitations

Current limitations include:

- Single primary production provider configuration
- English evaluation suites only
- Text-only evaluation
- No distributed execution
- No web dashboard
- No asynchronous batch scheduling

The architecture is intentionally modular to support future extensions.

---

# Future Work

Potential improvements include:

- Multi-judge ensemble voting
- Additional provider integrations
- Visual dashboards
- Human annotation interface
- Active learning for difficult cases
- Confidence calibration
- Multi-modal evaluation
- Parallel execution
- Distributed benchmarking
- Continuous integration pipelines

---

# Technology Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| LLM Provider | Groq |
| Models | GPT-OSS-120B |
| Configuration | YAML |
| Data Formats | JSON, YAML |
| Testing | Pytest |
| CLI | argparse |
| Logging | JSON Reports |
| Evaluation | Structured Rubrics |
| Validation | Cohen's Kappa, Pearson, Spearman |

---

# License

This project is released under the MIT License.

---

# Acknowledgements

EvalForge builds upon recent advances in LLM-as-a-Judge evaluation, structured prompting, and automated benchmarking while emphasizing reproducibility, interpretability, and bias-aware evaluation practices.
