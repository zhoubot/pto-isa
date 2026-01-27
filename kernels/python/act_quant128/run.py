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

from kernels.python.act_quant128.kernel import ActQuant128Config, make_act_quant128_kernel  # noqa: E402


def _default_ptoas() -> Path:
    for p in (
        _REPO_ROOT / "ptoas/mlir/build/bin/ptoas",
        _REPO_ROOT / "ptoas/mlir/build-macos/bin/ptoas",
    ):
        if p.exists():
            return p
    return _REPO_ROOT / "ptoas/mlir/build/bin/ptoas"


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/python/act_quant128: Python -> PTO-AS -> ptoas -> run/compare.")
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--block-m", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--block-dim", type=int, default=48, help="Launch block_dim (use 48 to fill A3 vector).")

    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="cpu")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_act_quant128"))

    # NPU options (optional; required only if you want to build/run the fatobj .so)
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC (a3|a5|Ascend910B1|...) when --run-mode=sim")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--bench-iters", type=int, default=50)
    ap.add_argument("--bench-warmup", type=int, default=10)
    ap.add_argument("--check", dest="check", action="store_true", help="Compare NPU outputs to CPU reference (default).")
    ap.add_argument("--no-check", dest="check", action="store_false", help="Skip CPU reference and output checks.")
    ap.set_defaults(check=True)
    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    cfg = ActQuant128Config(m=int(args.m), n=int(args.n), block_m=int(args.block_m), group_size=int(args.group_size))
    spec = make_act_quant128_kernel(cfg=cfg)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pto_path = args.outdir / f"{spec.name}.pto"
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))
    pto_path.write_text(pto_text, encoding="utf-8")

    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=int(args.block_dim), kernel_name=host_spec.kernel_name
    )

    base = pipeline.make_host_arrays(host_spec)
    cpu_out = None
    if args.target in ("cpu", "both") or args.check:
        cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
        cpu_so = args.outdir / f"lib{spec.name}_cpu.so"
        pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
        cpu_arrays = [a.copy() for a in base]
        cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

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
            arch="dav-c220-vec",
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
        if args.check:
            if cpu_out is None:
                raise RuntimeError("internal error: cpu_out is required when --check is enabled")
            out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
            # f16 div/cvt/clamp can differ slightly between CPU sim and NPU; use a small tolerance for f16.
            for i, (c, n, dt) in enumerate(zip(cpu_out, npu_out, out_dtypes)):
                if dt == "f16":
                    np.testing.assert_allclose(n, c, rtol=1e-3, atol=1e-3, err_msg=f"output {i} ({dt}) mismatch")
                else:
                    rtol, atol = pipeline._default_tol(dt)  # type: ignore[attr-defined]
                    np.testing.assert_allclose(n, c, rtol=rtol, atol=atol, err_msg=f"output {i} ({dt}) mismatch")
        if npu_res.bench:
            t = npu_res.bench
            print(
                f"timing_us: avg={t.avg_us:.2f} p50={t.p50_us:.2f} min={t.min_us:.2f} max={t.max_us:.2f} "
                f"iters={t.iters} warmup={t.warmup} method={t.method}"
            )
        if args.check:
            print(f"OK: act_quant128 matched CPU reference (bin: {bin_path.name})")
        else:
            print(f"OK: act_quant128 ran (bin: {bin_path.name})")

    print(f"OK: kernels/python/act_quant128 (target={args.target}) outdir={args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
