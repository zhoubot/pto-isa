#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ptoas.python import pipeline


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="End-to-end PTO-AS -> CCE -> BIN -> NPU run (numpy check).")
    ap.add_argument("--ptoas", type=Path, default=repo / "ptoas/mlir/build/bin/ptoas", help="Path to ptoas binary")
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home(), help="ASCEND_HOME_PATH")
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

    cfg_add = pipeline.CompileConfig(ptoas=args.ptoas, ascend_home=args.ascend_home, arch="dav-c220-vec")
    cfg_gemm = pipeline.CompileConfig(ptoas=args.ptoas, ascend_home=args.ascend_home, arch="dav-c220-cube")

    add_cce, add_bin = pipeline.compile_pto_to_cce_and_bin(pto_path=args.add_pto, outdir=args.outdir, cfg=cfg_add)
    add_so = args.outdir / "libadd16.so"
    pipeline.build_fatobj_so_from_cce(cce_path=add_cce, out_so=add_so, arch=cfg_add.arch, ascend_home=args.ascend_home)
    pipeline.run_add16_from_so(so_path=add_so, device_id=args.device, block_dim=args.block_dim)

    gemm_cce, gemm_bin = pipeline.compile_pto_to_cce_and_bin(pto_path=args.gemm_pto, outdir=args.outdir, cfg=cfg_gemm)
    gemm_so = args.outdir / "libgemm16.so"
    pipeline.build_fatobj_so_from_cce(cce_path=gemm_cce, out_so=gemm_so, arch=cfg_gemm.arch, ascend_home=args.ascend_home)
    pipeline.run_gemm16_from_so(so_path=gemm_so, device_id=args.device, block_dim=args.block_dim)

    print(f"OK: add and gemm passed (artifacts in {args.outdir}; bins: {add_bin.name}, {gemm_bin.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
