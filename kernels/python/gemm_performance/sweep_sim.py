#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    m: int
    n: int
    k: int
    grid_m: int = 1
    grid_n: int = 1
    allow_unaligned: bool = False

    def to_args(self) -> list[str]:
        args = [
            "--m",
            str(int(self.m)),
            "--n",
            str(int(self.n)),
            "--k",
            str(int(self.k)),
            "--grid-m",
            str(int(self.grid_m)),
            "--grid-n",
            str(int(self.grid_n)),
        ]
        if self.allow_unaligned:
            args.append("--allow-unaligned")
        return args


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runner() -> Path:
    return Path(__file__).resolve().parent / "run.py"


def _run(
    *,
    cmd: list[str],
    timeout_sec: float | None,
) -> tuple[int, float]:
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=False, timeout=timeout_sec)
        return (int(proc.returncode), time.time() - t0)
    except subprocess.TimeoutExpired:
        return (124, time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep kernels/python/gemm_performance in simulator mode (deadlock detection).")
    ap.add_argument("--ptoas", type=Path, default=None, help="Optional path to ptoas binary.")
    ap.add_argument("--ascend-home", type=Path, default=None, help="Optional Ascend toolkit root.")
    ap.add_argument("--soc", default="a3")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_gemm_sweep_sim"))
    ap.add_argument("--clean", action="store_true", help="Remove outdir before running.")

    ap.add_argument("--timeout-sec", type=float, default=30.0, help="Per-case runtime timeout (deadlock if exceeded).")
    ap.add_argument("--compile-timeout-sec", type=float, default=600.0, help="Per-case compile timeout.")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--check-samples", type=int, default=4)
    ap.add_argument("--no-check", action="store_true", help="Disable sampled correctness checks.")
    args = ap.parse_args()

    runner = _runner()
    if not runner.exists():
        print(f"error: runner not found: {runner}", file=sys.stderr)
        return 2

    if args.clean and args.outdir.exists():
        import shutil

        shutil.rmtree(args.outdir, ignore_errors=True)

    # Keep cases small: simulator init is expensive; avoid huge padded sizes.
    cases: list[Case] = [
        # Aligned (base tile for grid_m=1, grid_n=1: M=128, N=256, K=64)
        Case(128, 256, 64, grid_m=1, grid_n=1, allow_unaligned=False),
        # Aligned, but spread work across more blocks to keep sim runtime < 30s.
        Case(256, 256, 128, grid_m=2, grid_n=1, allow_unaligned=False),
        Case(256, 512, 128, grid_m=1, grid_n=2, allow_unaligned=False),
        Case(256, 512, 128, grid_m=2, grid_n=2, allow_unaligned=False),
        Case(384, 768, 192, grid_m=3, grid_n=3, allow_unaligned=False),
        # Unaligned (runner pads and validates only requested region)
        Case(129, 257, 65, grid_m=2, grid_n=2, allow_unaligned=True),
        Case(191, 383, 127, grid_m=2, grid_n=2, allow_unaligned=True),
        Case(255, 511, 191, grid_m=2, grid_n=2, allow_unaligned=True),
    ]

    ok = 0
    failed = 0
    timed_out = 0

    for idx, case in enumerate(cases):
        print(f"\n=== [{idx + 1}/{len(cases)}] compile: {case} ===", flush=True)
        base_cmd = [
            sys.executable,
            str(runner),
            "--run-mode",
            "sim",
            "--soc",
            str(args.soc),
            "--device",
            str(int(args.device)),
            "--outdir",
            str(args.outdir),
            "--iters",
            str(int(args.iters)),
            "--warmup",
            str(int(args.warmup)),
            "--check-samples",
            str(int(args.check_samples)),
        ]
        if args.no_check:
            base_cmd.append("--no-check")
        if args.ptoas is not None:
            base_cmd += ["--ptoas", str(args.ptoas)]
        if args.ascend_home is not None:
            base_cmd += ["--ascend-home", str(args.ascend_home)]
        base_cmd += case.to_args()

        compile_cmd = base_cmd + ["--compile-only"]
        rc, dt = _run(cmd=compile_cmd, timeout_sec=float(args.compile_timeout_sec) if args.compile_timeout_sec else None)
        if rc == 124:
            print(f"TIMEOUT: compile exceeded {args.compile_timeout_sec}s", flush=True)
            timed_out += 1
            continue
        if rc != 0:
            print(f"FAIL: compile returned {rc}", flush=True)
            failed += 1
            continue

        print(f"\n=== [{idx + 1}/{len(cases)}] run: {case} ===", flush=True)
        run_cmd = base_cmd + ["--skip-build"]
        rc, dt = _run(cmd=run_cmd, timeout_sec=float(args.timeout_sec) if args.timeout_sec else None)
        if rc == 124:
            print(f"TIMEOUT: run exceeded {args.timeout_sec}s (possible deadlock)", flush=True)
            timed_out += 1
            continue
        if rc != 0:
            print(f"FAIL: run returned {rc}", flush=True)
            failed += 1
            continue
        ok += 1

    print(f"\n=== summary ===\nOK={ok}  FAIL={failed}  TIMEOUT={timed_out}  TOTAL={len(cases)}")
    return 0 if (failed == 0 and timed_out == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
