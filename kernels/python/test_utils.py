from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Env:
    ascend_home: str
    run_mode: str = "npu"
    soc: str = "a3"

