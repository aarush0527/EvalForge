"""Turns a judge_config.yaml / generator_config.yaml into live objects."""
from __future__ import annotations

from pathlib import Path

import yaml

from .audit_log import AuditLog
from .judge import Judge
from .providers import AnthropicProvider, GroqProvider, ModelProvider, OpenAIProvider, SimulatedProvider
from .rubric import load_rubric


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def build_provider(cfg: dict) -> ModelProvider:
    kind = cfg.get("provider", "simulated")
    model = cfg.get("model")
    family = cfg.get("family")
    if kind == "groq":
        return GroqProvider(model=model or "openai/gpt-oss-120b", family=family,
                              reasoning_effort=cfg.get("reasoning_effort"))
    if kind == "anthropic":
        return AnthropicProvider(model=model or "claude-opus-4-8", family=family)
    if kind == "openai":
        return OpenAIProvider(model=model or "gpt-5.4", family=family)
    if kind == "simulated":
        return SimulatedProvider(
            model=model or "simulated-judge-v1",
            mitigations_aware=cfg.get("mitigations_aware", True),
            self_family_bias_marker=cfg.get("self_family_bias_marker"),
            family=family,
        )
    raise ValueError(f"Unknown provider kind: {kind!r}")


def build_judge(config_path: str | Path, *, rubric_path: str | Path | None = None,
                 audit_log_path: str | Path | None = None) -> Judge:
    cfg = load_yaml(config_path)
    provider = build_provider(cfg)
    rubric = load_rubric(rubric_path or cfg.get("rubric_path"))
    audit_log = AuditLog(audit_log_path) if audit_log_path else None
    return Judge(
        provider=provider,
        rubric=rubric,
        mitigations=cfg.get("mitigations", True),
        audit_log=audit_log,
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 1024),
        max_parse_retries=cfg.get("max_parse_retries", 1),
    )


def provider_family(config_path: str | Path) -> str:
    cfg = load_yaml(config_path)
    kind = cfg.get("provider", "simulated")
    return cfg.get("family", kind)
