"""Loads the different suite YAML shapes used by run / compare / validate."""
from __future__ import annotations

from pathlib import Path

import yaml

from .schema import TestCase, TestSuite


def load_suite(path: str | Path) -> TestSuite:
    data = yaml.safe_load(Path(path).read_text())
    return TestSuite.model_validate(data)


def load_adversarial_pairs(path: str | Path) -> list[dict[str, TestCase]]:
    """
    Expects:
      pairs:
        - id: probe_1
          input: "..."
          system_prompt: "..."
          expected_output: "..."
          verbose_wrong: {model_output: "..."}
          terse_correct: {model_output: "..."}
    """
    data = yaml.safe_load(Path(path).read_text())
    pairs = []
    for p in data["pairs"]:
        shared = dict(
            input=p["input"],
            system_prompt=p.get("system_prompt", ""),
            expected_output=p.get("expected_output"),
        )
        vw = TestCase(id=f"{p['id']}_verbose_wrong", tags=["adversarial", "verbose_wrong"],
                        model_output=p["verbose_wrong"]["model_output"], **shared)
        tc = TestCase(id=f"{p['id']}_terse_correct", tags=["adversarial", "terse_correct"],
                        model_output=p["terse_correct"]["model_output"], **shared)
        pairs.append({"verbose_wrong": vw, "terse_correct": tc})
    return pairs
