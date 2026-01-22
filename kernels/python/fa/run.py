#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import pipeline  # noqa: E402
from ptoas.python.host_codegen import TensorSpec, emit_acl_host_cpp  # noqa: E402

from kernel import make_fa16_pto  # noqa: E402


def _default_ptoas() -> Path:
    for p in (
        _REPO_ROOT / "ptoas/mlir/build-macos/bin/ptoas",
        _REPO_ROOT / "ptoas/mlir/build/bin/ptoas",
    ):
        if p.exists():
            return p
    return _REPO_ROOT / "ptoas/mlir/build-macos/bin/ptoas"


def _run_cpu(*, so_path: Path) -> None:
    q = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    k = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    v = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    out = np.empty_like(q)
    expected = (q + k + v).astype(np.float16)

    lib = ctypes.CDLL(str(so_path))
    fn = lib.pto_kernel_cpu
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = None

    fn(
        ctypes.c_void_p(int(q.ctypes.data)),
        ctypes.c_void_p(int(k.ctypes.data)),
        ctypes.c_void_p(int(v.ctypes.data)),
        ctypes.c_void_p(int(out.ctypes.data)),
    )
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/python/fa: Python -> PTO-AS -> ptoas -> emit foo.cpp + host.cpp.")
    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="cpu")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_fa"))

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

    # Always emit the PTO-AS.
    pto_path = args.outdir / "fa16.pto"
    pto_path.write_text(make_fa16_pto(target="cpu" if args.target == "cpu" else "npu"), encoding="utf-8")

    # Always emit device source + host source (even on macOS with no NPU).
    device_cpp = args.outdir / "fa16.cpp"
    pipeline.compile_pto_to_device_cpp(
        pto_path=pto_path,
        out_cpp=device_cpp,
        ptoas=args.ptoas,
        arch="dav-c220-vec",
        memory_model=args.memory_model,
    )
    host_cpp = args.outdir / "host.cpp"
    host_cpp.write_text(
        emit_acl_host_cpp(
            so_basename="libfa16_npu.so",
            args=[TensorSpec("f16", (16, 16)), TensorSpec("f16", (16, 16)), TensorSpec("f16", (16, 16)), TensorSpec("f16", (16, 16))],
        ),
        encoding="utf-8",
    )

    if args.target in ("cpu", "both"):
        cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
        so_path = args.outdir / "libfa16_cpu.so"
        pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=so_path)
        _run_cpu(so_path=so_path)

    if args.target in ("npu", "both") and args.ascend_home and args.ascend_home.exists():
        cfg = pipeline.CompileConfig(
            ptoas=args.ptoas,
            ascend_home=args.ascend_home,
            arch="dav-c220-vec",
            memory_model=args.memory_model,
            insert_events=True,
        )
        cce_path, _ = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg)
        so_path = args.outdir / "libfa16_npu.so"
        pipeline.build_fatobj_so_from_cce(cce_path=cce_path, out_so=so_path, arch=cfg.arch, ascend_home=cfg.ascend_home)
        print(f"built: {so_path} (run {host_cpp} on NPU env to launch)")

    print(f"OK: kernels/python/fa (target={args.target}) outdir={args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
