from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ptoas.python import ast_frontend, pipeline

Target = Literal["cpu", "npu"]


@dataclass(frozen=True)
class Case:
    name: str
    desc: str
    covers: tuple[str, ...]
    make_pto: Callable[[Target], str]
    run: Callable[[Target, Path, argparse.Namespace], None]


def _case_add16() -> Case:
    def make_pto(target: Target) -> str:
        return ast_frontend.make_add16_program()

    def run(target: Target, so_or_unused: Path, args: argparse.Namespace) -> None:
        if target == "cpu":
            pipeline.run_add16_cpu_from_so(so_path=so_or_unused)
        else:
            pipeline.run_add16_from_so(so_path=so_or_unused, device_id=args.device, block_dim=args.block_dim)

    return Case(
        name="add16",
        desc="Vec add 16x16 (tload/tadd/tstore)",
        covers=("tassign", "tload", "tadd", "tstore"),
        make_pto=make_pto,
        run=run,
    )


def _case_gemm16() -> Case:
    def make_pto(target: Target) -> str:
        if target == "cpu":
            return ast_frontend.make_gemm16_cpu_program()
        return ast_frontend.make_gemm16_program()

    def run(target: Target, so_or_unused: Path, args: argparse.Namespace) -> None:
        if target == "cpu":
            pipeline.run_gemm16_cpu_from_so(so_path=so_or_unused)
        else:
            pipeline.run_gemm16_from_so(so_path=so_or_unused, device_id=args.device, block_dim=args.block_dim)

    return Case(
        name="gemm16",
        desc="Cube GEMM 16x16 (tmatmul)",
        covers=("tassign", "tload", "tmov", "tmatmul", "tstore"),
        make_pto=make_pto,
        run=run,
    )


CASES: dict[str, Case] = {
    "add16": _case_add16(),
    "gemm16": _case_gemm16(),
}


def _compile_and_run_case(*, case: Case, target: Target, args: argparse.Namespace) -> None:
    repo = pipeline.repo_root()
    ptoas_bin = args.ptoas

    with tempfile.TemporaryDirectory(prefix=f"ptoas_st_{case.name}_{target}_") as td:
        td = Path(td)
        outdir = Path(args.outdir) if args.outdir else td
        outdir.mkdir(parents=True, exist_ok=True)

        pto_path = outdir / f"{case.name}.{target}.pto"
        pto_path.write_text(case.make_pto(target), encoding="utf-8")

        if target == "cpu":
            cpp_path = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=outdir, ptoas=ptoas_bin)
            so_path = outdir / f"lib{case.name}_cpu.so"
            pipeline.build_cpu_so_from_cpp(cpp_path=cpp_path, out_so=so_path)
            case.run(target, so_path, args)
            return

        cfg = pipeline.CompileConfig(
            ptoas=ptoas_bin,
            ascend_home=args.ascend_home,
            arch=args.arch,
            insert_events=args.insert_events,
            memory_model=args.memory_model,
        )
        cce_path, _ = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=outdir, cfg=cfg)
        so_path = outdir / f"lib{case.name}_npu.so"
        pipeline.build_fatobj_so_from_cce(cce_path=cce_path, out_so=so_path, arch=cfg.arch, ascend_home=cfg.ascend_home)
        case.run(target, so_path, args)


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="Python ST runner for ptoas (CPU + NPU).")
    ap.add_argument("--list", action="store_true", help="List available cases")
    ap.add_argument("--list-instr", action="store_true", help="List PTO instruction mnemonics and coverage")
    ap.add_argument("--case", default="add16", help=f"Case name ({', '.join(sorted(CASES))})")
    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="cpu")
    ap.add_argument("--ptoas", type=Path, default=repo / "ptoas/mlir/build/bin/ptoas")
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/ptoas_py_st"))

    # NPU options
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--arch", default="dav-c220-vec")
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    args = ap.parse_args()

    if args.list:
        for k in sorted(CASES):
            c = CASES[k]
            print(f"{c.name}: {c.desc}")
        return 0

    if args.list_instr:
        import re

        text = (repo / "include/pto/common/pto_instr.hpp").read_text(encoding="utf-8", errors="ignore")
        apis: set[str] = set()
        for line in text.splitlines():
            if "PTO_INST" not in line:
                continue
            m = re.search(r"\\bPTO_INST\\b.*?\\b([A-Z][A-Z0-9_]*)\\s*\\(", line)
            if m:
                apis.add(m.group(1))
        mnemonics = sorted(a.lower() for a in apis)
        covered: dict[str, list[str]] = {}
        for c in CASES.values():
            for m in c.covers:
                covered.setdefault(m, []).append(c.name)
        for m in mnemonics:
            tags = ",".join(sorted(covered.get(m, [])))
            print(f"{m}\t{tags}")
        return 0

    if args.case not in CASES:
        print(f"error: unknown --case {args.case}", file=sys.stderr)
        return 2
    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    targets: list[Target] = []
    if args.target in ("cpu", "both"):
        targets.append("cpu")
    if args.target in ("npu", "both"):
        targets.append("npu")

    if "npu" in targets:
        if not args.ascend_home or not args.ascend_home.exists():
            print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
            return 2

    case = CASES[args.case]
    for t in targets:
        if t == "npu" and case.name == "gemm16":
            args.arch = "dav-c220-cube"
        _compile_and_run_case(case=case, target=t, args=args)

    print(f"OK: {case.name} ({args.target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
