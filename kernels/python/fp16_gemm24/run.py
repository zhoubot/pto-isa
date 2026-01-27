#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import binding, pipeline  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402

from kernels.python.fp16_gemm24.kernel import Fp16Gemm24Config, make_fp16_gemm24_kernel  # noqa: E402


def _default_ptoas() -> Path:
    for p in (
        _REPO_ROOT / "ptoas/mlir/build/bin/ptoas",
        _REPO_ROOT / "ptoas/mlir/build-macos/bin/ptoas",
    ):
        if p.exists():
            return p
    return _REPO_ROOT / "ptoas/mlir/build/bin/ptoas"


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/python/fp16_gemm24: Python -> PTO-AS -> ptoas -> run/compare.")
    ap.add_argument("--m", type=int, default=6144)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--grid-m", type=int, default=4)
    ap.add_argument("--grid-n", type=int, default=6)
    ap.add_argument("--base-m", type=int, default=128)
    ap.add_argument("--base-n", type=int, default=256)
    ap.add_argument("--base-k", type=int, default=64)
    ap.add_argument("--block-dim", type=int, default=24, help="Launch block_dim (use 24 to fill A3 cube).")

    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="cpu")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_fp16_gemm24"))

    # NPU options
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--bench-iters", type=int, default=50)
    ap.add_argument("--bench-warmup", type=int, default=10)
    ap.add_argument(
        "--check-samples",
        type=int,
        default=32,
        help="(NPU) Validate by sampling this many output elements against NumPy. Set 0 to skip.",
    )
    ap.add_argument("--check-rtol", type=float, default=2e-2)
    ap.add_argument("--check-atol", type=float, default=5e-2)
    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    cfg = Fp16Gemm24Config(
        m=int(args.m),
        n=int(args.n),
        k=int(args.k),
        grid_m=int(args.grid_m),
        grid_n=int(args.grid_n),
        base_m=int(args.base_m),
        base_n=int(args.base_n),
        base_k=int(args.base_k),
    )
    spec = make_fp16_gemm24_kernel(cfg=cfg)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pto_path = args.outdir / f"{spec.name}.pto"
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))
    pto_path.write_text(pto_text, encoding="utf-8")

    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=int(args.block_dim), kernel_name=host_spec.kernel_name
    )
    base = pipeline.make_host_arrays(host_spec)

    if args.target in ("cpu", "both"):
        cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
        cpu_so = args.outdir / f"lib{spec.name}_cpu.so"
        pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
        cpu_arrays = [a.copy() for a in base]
        pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

    if args.target in ("npu", "both"):
        if not args.ascend_home or not args.ascend_home.exists():
            print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
            return 2
        if args.run_mode == "sim":
            soc_full = "Ascend910B1" if args.soc == "a3" else ("Ascend910_9599" if args.soc == "a5" else args.soc)
            pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc_full)
            runtime_lib = "runtime_camodel"
        else:
            runtime_lib = "runtime"
            soc_full = None

        cfg_npu = pipeline.CompileConfig(
            ptoas=args.ptoas,
            ascend_home=args.ascend_home,
            arch="dav-c220-cube",
            memory_model=args.memory_model,
            insert_events=True,
        )
        cce_path, bin_path = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg_npu)
        npu_so = args.outdir / f"lib{spec.name}_npu.so"
        pipeline.build_fatobj_so_from_cce(
            cce_path=cce_path,
            out_so=npu_so,
            arch=cfg_npu.arch,
            ascend_home=cfg_npu.ascend_home,
            runtime_lib=runtime_lib,
            soc=soc_full,
        )

        npu_arrays = [a.copy() for a in base]
        npu_res = pipeline.run_npu_kernel_from_so(
            so_path=npu_so,
            host_spec=host_spec,
            host_arrays=npu_arrays,
            device_id=int(args.device),
            block_dim=int(args.block_dim),
            bench_iters=int(args.bench_iters) if args.run_mode == "npu" else 0,
            bench_warmup=int(args.bench_warmup) if args.run_mode == "npu" else 0,
        )
        npu_out = npu_res.outputs
        # NOTE: CPU backend currently hard-codes get_block_idx/get_block_num for cube kernels,
        # so CPU output only covers block 0. Validate against NumPy instead.
        if int(args.check_samples) > 0:
            rng = np.random.default_rng(int(getattr(host_spec, "seed", 0)))
            a = base[0]
            # DN tensor is backed by a physical [n, k] row-major buffer on host (B^T contiguous).
            b_t = base[1]
            c = npu_out[0]
            m, k = a.shape
            n = b_t.shape[0]
            for _ in range(int(args.check_samples)):
                r = int(rng.integers(0, m))
                col = int(rng.integers(0, n))
                expected = float(np.dot(a[r, :].astype(np.float32), b_t[col, :].astype(np.float32)))
                got = float(c[r, col])
                if not np.isfinite(got):
                    raise AssertionError(f"non-finite output at ({r},{col}): {got}")
                if not np.isclose(got, expected, rtol=float(args.check_rtol), atol=float(args.check_atol)):
                    raise AssertionError(f"mismatch at ({r},{col}): got={got} expected={expected}")
        if npu_res.bench:
            t = npu_res.bench
            print(
                f"timing_us: avg={t.avg_us:.2f} p50={t.p50_us:.2f} min={t.min_us:.2f} max={t.max_us:.2f} "
                f"iters={t.iters} warmup={t.warmup} method={t.method}"
            )
        if int(args.check_samples) > 0:
            print(f"OK: fp16_gemm24 matched NumPy samples (bin: {bin_path.name})")
        else:
            print(f"OK: fp16_gemm24 ran (bin: {bin_path.name})")

    print(f"OK: kernels/python/fp16_gemm24 (target={args.target}) outdir={args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
