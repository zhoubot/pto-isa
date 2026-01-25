#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ptoas.python import pipeline


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="End-to-end PTO-AS -> (CPU+NPU) -> run and compare (CPU as reference).")
    ap.add_argument("--ptoas", type=Path, default=repo / "ptoas/mlir/build/bin/ptoas", help="Path to ptoas binary")
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home(), help="ASCEND_HOME_PATH")
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC (a3|a5|Ascend910B1|...) when --run-mode=sim")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/ptoas_e2e"))
    ap.add_argument("--add-pto", type=Path, default=repo / "ptoas/examples/add16_e2e.pto")
    ap.add_argument("--gemm-pto", type=Path, default=repo / "ptoas/examples/gemm16_e2e.pto")
    args = ap.parse_args()

    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2
    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.add_pto.exists():
        print(f"error: add pto not found: {args.add_pto}", file=sys.stderr)
        return 2
    if not args.gemm_pto.exists():
        print(f"error: gemm pto not found: {args.gemm_pto}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.run_mode == "sim":
        # Match the mapping used by tests/script/run_st.py.
        soc = "Ascend910B1" if args.soc == "a3" else ("Ascend910_9599" if args.soc == "a5" else args.soc)
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc)
        runtime_lib = "runtime_camodel"
    else:
        runtime_lib = "runtime"
        soc = None

    def _run_one(*, name: str, pto: Path, arch: str) -> None:
        pto_text = pto.read_text(encoding="utf-8")
        host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
        host_spec = type(host_spec)(
            args=host_spec.args, seed=host_spec.seed, block_dim=args.block_dim, kernel_name=host_spec.kernel_name
        )

        base = pipeline.make_host_arrays(host_spec)
        cpu_arrays = [a.copy() for a in base]
        npu_arrays = [a.copy() for a in base]

        cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto, outdir=args.outdir, ptoas=args.ptoas)
        cpu_so = args.outdir / f"lib{name}_cpu.so"
        pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
        cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

        cfg = pipeline.CompileConfig(ptoas=args.ptoas, ascend_home=args.ascend_home, arch=arch)
        npu_cpp, npu_bin = pipeline.compile_pto_to_cce_and_bin(pto_path=pto, outdir=args.outdir, cfg=cfg)
        npu_so = args.outdir / f"lib{name}_npu.so"
        pipeline.build_fatobj_so_from_cce(
            cce_path=npu_cpp,
            out_so=npu_so,
            arch=arch,
            ascend_home=args.ascend_home,
            runtime_lib=runtime_lib,
            soc=soc,
        )
        npu_res = pipeline.run_npu_kernel_from_so(
            so_path=npu_so, host_spec=host_spec, host_arrays=npu_arrays, device_id=args.device, block_dim=args.block_dim
        )
        npu_out = npu_res.outputs

        out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
        pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)
        print(f"OK: {name} (bin: {npu_bin.name})")

    _run_one(name="add16", pto=args.add_pto, arch="dav-c220-vec")
    _run_one(name="gemm16", pto=args.gemm_pto, arch="dav-c220-cube")

    print(f"OK: all kernels matched CPU reference (artifacts in {args.outdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
