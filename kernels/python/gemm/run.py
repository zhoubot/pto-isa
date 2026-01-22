#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import pipeline  # noqa: E402
from ptoas.python.host_codegen import TensorSpec, emit_acl_host_cpp  # noqa: E402

from kernel import make_gemm16_pto  # noqa: E402


def _default_ptoas() -> Path:
    for p in (
        _REPO_ROOT / "ptoas/mlir/build-macos/bin/ptoas",
        _REPO_ROOT / "ptoas/mlir/build/bin/ptoas",
    ):
        if p.exists():
            return p
    return _REPO_ROOT / "ptoas/mlir/build-macos/bin/ptoas"


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/python/gemm: Python -> PTO-AS -> ptoas -> emit foo.cpp + host.cpp.")
    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="cpu")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_gemm"))

    # NPU options (optional; required only if you want to build/run the fatobj .so)
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Always emit device source + host source (even on macOS with no NPU).
    pto_path = args.outdir / "gemm16.pto"
    pto_path.write_text(make_gemm16_pto(target="npu"), encoding="utf-8")

    device_cpp = args.outdir / "gemm16.cpp"
    pipeline.compile_pto_to_device_cpp(
        pto_path=pto_path,
        out_cpp=device_cpp,
        ptoas=args.ptoas,
        arch="dav-c220-cube",
        memory_model=args.memory_model,
    )
    host_cpp = args.outdir / "host.cpp"
    host_cpp.write_text(
        emit_acl_host_cpp(
            so_basename="libgemm16_npu.so",
            args=[TensorSpec("f16", (16, 16)), TensorSpec("f16", (16, 16)), TensorSpec("f32", (16, 16))],
        ),
        encoding="utf-8",
    )

    if args.target in ("cpu", "both"):
        cpu_pto = args.outdir / "gemm16_cpu.pto"
        cpu_pto.write_text(make_gemm16_pto(target="cpu"), encoding="utf-8")
        cpp_path = pipeline.compile_pto_to_cpu_cpp(pto_path=cpu_pto, outdir=args.outdir, ptoas=args.ptoas)
        so_path = args.outdir / "libgemm16_cpu.so"
        pipeline.build_cpu_so_from_cpp(cpp_path=cpp_path, out_so=so_path)
        pipeline.run_gemm16_cpu_from_so(so_path=so_path)

    if args.target in ("npu", "both") and args.ascend_home and args.ascend_home.exists():
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
        print(f"built: {so_path} (run {host_cpp} on NPU env to launch)")

    print(f"OK: kernels/python/gemm (target={args.target}) outdir={args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
