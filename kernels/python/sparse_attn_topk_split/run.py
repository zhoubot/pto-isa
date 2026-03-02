#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BINDING_PY = _REPO_ROOT / "frontend" / "python"
if str(_BINDING_PY) not in sys.path:
    sys.path.insert(0, str(_BINDING_PY))

from ptoas.python import frontend, pipeline  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402

from kernels.python.sparse_attn_topk_split.kernel import (  # noqa: E402
    SparseAttnTopKSplitConfig,
    make_sparse_attn_topk_split_kernel,
)


def _default_ptoas() -> Path:
    p = _REPO_ROOT / "bin" / "ptoas"
    if p.exists():
        return p
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description="kernels/python/sparse_attn_topk_split: compute-only sparse attention (gather moved out)."
    )
    ap.add_argument("--q", type=int, default=256, help="Number of query positions (flattened b*m).")
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--block-dim", type=int, default=24)
    ap.add_argument("--input-scale", type=float, default=0.1, help="Scale random inputs down to reduce exp drift.")

    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="cpu")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_sparse_attn_topk_split"))

    # NPU options
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--bench-iters", type=int, default=50)
    ap.add_argument("--bench-warmup", type=int, default=10)
    ap.add_argument("--check", dest="check", action="store_true", help="Compare NPU outputs to CPU reference (default).")
    ap.add_argument("--no-check", dest="check", action="store_false", help="Skip CPU reference and output checks.")
    ap.set_defaults(check=True)
    args = ap.parse_args()

    args.ptoas = pipeline.ensure_ptoas_binary(args.ptoas)
    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    cfg = SparseAttnTopKSplitConfig(q=int(args.q), h=int(args.h), d=int(args.d), topk=int(args.topk))
    spec = make_sparse_attn_topk_split_kernel(cfg=cfg)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pto_path = args.outdir / f"{spec.name}.pto"
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=frontend.default_host_spec(spec))
    pto_path.write_text(pto_text, encoding="utf-8")

    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=int(args.block_dim), kernel_name=host_spec.kernel_name
    )
    base = pipeline.make_host_arrays(host_spec)
    if float(args.input_scale) != 1.0:
        for i, a in enumerate(host_spec.args):
            if a.role == "out":
                continue
            if base[i].dtype in (np.float16, np.float32):
                base[i] = (base[i].astype(np.float32) * float(args.input_scale)).astype(base[i].dtype)

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
            arch="dav-c220",
            memory_model=args.memory_model,
            insert_events=True,
            split_kernels=True,
        )
        cce_path, bin_path = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg_npu)
        npu_so = args.outdir / f"lib{spec.name}_npu.so"
        pipeline.build_fatobj_so_from_cce(
            cce_path=cce_path,
            out_so=npu_so,
            arch=cfg_npu.arch,
            ascend_home=cfg_npu.ascend_home,
            memory_model=cfg_npu.memory_model,
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
            pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)
        if npu_res.bench:
            t = npu_res.bench
            print(
                f"timing_us: avg={t.avg_us:.2f} p50={t.p50_us:.2f} min={t.min_us:.2f} max={t.max_us:.2f} "
                f"iters={t.iters} warmup={t.warmup} method={t.method}"
            )
        artifact = f"so: {npu_so.name}" if not bin_path.exists() else f"bin: {bin_path.name}"
        if args.check:
            print(f"OK: sparse_attn_topk_split matched CPU reference ({artifact})")
        else:
            print(f"OK: sparse_attn_topk_split ran ({artifact})")

    print(f"OK: kernels/python/sparse_attn_topk_split (target={args.target}) outdir={args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
