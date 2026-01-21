#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import pipeline  # noqa: E402
from gemm_pto import make_gemm16_pto  # noqa: E402


def _repo_root() -> Path:
    return _REPO_ROOT


def _default_ptoas() -> Path:
    return _repo_root() / "ptoas/mlir/build/bin/ptoas"


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/custom/gemm_python: Python -> PTO-AS -> ptoas -> run (CPU/NPU).")
    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="cpu")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_gemm_python"))

    # NPU options
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.target in ("cpu", "both"):
        pto_path = args.outdir / "gemm16.cpu.pto"
        pto_path.write_text(make_gemm16_pto(target="cpu"), encoding="utf-8")

        cpp_path = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
        so_path = args.outdir / "libgemm16_cpu.so"
        pipeline.build_cpu_so_from_cpp(cpp_path=cpp_path, out_so=so_path)
        pipeline.run_gemm16_cpu_from_so(so_path=so_path)

    if args.target in ("npu", "both"):
        if not args.ascend_home or not args.ascend_home.exists():
            print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
            return 2

        pto_path = args.outdir / "gemm16.npu.pto"
        pto_path.write_text(make_gemm16_pto(target="npu"), encoding="utf-8")

        cfg = pipeline.CompileConfig(
            ptoas=args.ptoas,
            ascend_home=args.ascend_home,
            arch="dav-c220-cube",
            memory_model=args.memory_model,
            insert_events=True,
        )
        cce_path, _ = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg)
        so_path = args.outdir / "libgemm16_npu.so"
        pipeline.build_fatobj_so_from_cce(cce_path=cce_path, out_so=so_path, arch=cfg.arch, ascend_home=cfg.ascend_home)
        pipeline.run_gemm16_from_so(so_path=so_path, device_id=args.device, block_dim=args.block_dim)

    print(f"OK: kernels/custom/gemm_python (target={args.target}) outdir={args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
