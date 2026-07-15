"""Append-only JSONL audit log: every judge call, in full, in order."""
from __future__ import annotations

import json
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, case_id: str, mode: str, system_prompt: str, user_prompt: str,
             raw_response: str, judge_model: str, input_tokens: int, output_tokens: int,
             latency_ms: float, parse_retries: int = 0, extra: dict | None = None) -> None:
        record = {
            "timestamp": time.time(),
            "case_id": case_id,
            "mode": mode,
            "judge_model": judge_model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "parse_retries": parse_retries,
            "extra": extra or {},
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def replay(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
