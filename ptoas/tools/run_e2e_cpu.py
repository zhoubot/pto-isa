#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "binding" / "python"))

from ptoas.python import pipeline


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="PTO-AS -> ptoas --target cpu -> .cpp -> .so -> run on CPU (numpy check).")
    ap.add_argument("--ptoas", type=Path, default=repo / "ptoas/mlir/build/bin/ptoas")
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/ptoas_cpu_e2e"))
    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    add_pto = repo / "ptoas/examples/add16_e2e.pto"
    gemm_pto = repo / "ptoas/examples/gemm16_e2e.pto"

    add_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=add_pto, outdir=args.outdir, ptoas=args.ptoas)
    add_so = args.outdir / "libadd16_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=add_cpp, out_so=add_so)
    pipeline.run_add16_cpu_from_so(so_path=add_so)

    gemm_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=gemm_pto, outdir=args.outdir, ptoas=args.ptoas)
    gemm_so = args.outdir / "libgemm16_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=gemm_cpp, out_so=gemm_so)
    pipeline.run_gemm16_cpu_from_so(so_path=gemm_so)

    print(f"OK: CPU e2e passed (outdir: {args.outdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
