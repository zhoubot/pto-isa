#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import pipeline  # noqa: E402
from ptoas.python.host_codegen import TensorSpec  # noqa: E402


def _default_ptoas() -> Path:
    for p in (
        _REPO_ROOT / "ptoas/mlir/build/bin/ptoas",
        _REPO_ROOT / "ptoas/mlir/build-macos/bin/ptoas",
    ):
        if p.exists():
            return p
    return _REPO_ROOT / "ptoas/mlir/build/bin/ptoas"


def _soc_from_alias(alias: str) -> str:
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compile a .pto into a runnable fatobj executable (no dlopen .so) for sim/npu mode."
    )
    ap.add_argument("pto", type=Path)
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())

    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC when --run-mode=sim (a3|a5|Ascend910B1|...)")

    ap.add_argument("--arch", default="dav-c220-cube")
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    ap.add_argument("--fixed-block-dim", type=int, default=None, help="Bake blockDim into ptoas_launch (recommended).")

    ap.add_argument("--outdir", type=Path, default=_REPO_ROOT / "bin" / "pto_examples")
    ap.add_argument("--name", default=None, help="Output name prefix (default: <pto stem>)")

    ap.add_argument("--run", action="store_true", help="Run the built executable after compilation.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=None, help="Launch blockDim when running (default: from host spec).")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)

    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.pto.exists():
        print(f"error: pto not found: {args.pto}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    pto_text = args.pto.read_text(encoding="utf-8", errors="replace")
    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_specs = [TensorSpec(dtype=a.dtype, shape=(int(a.shape[0]), int(a.shape[1]))) for a in host_spec.args]
    block_dim = int(args.block_dim) if args.block_dim is not None else int(host_spec.block_dim)
    fixed_block_dim = int(args.fixed_block_dim) if args.fixed_block_dim is not None else int(block_dim)

    name = args.name or args.pto.stem
    outdir = (args.outdir / name / args.run_mode).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cce_path = outdir / f"{name}.cpp"
    exe_path = outdir / f"{name}_{args.run_mode}"

    if args.run_mode == "sim":
        soc_full = _soc_from_alias(str(args.soc))
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc_full)
        runtime_lib = "runtime_camodel"
    else:
        soc_full = None
        runtime_lib = "runtime"

    # 1) PTO -> CCE C++ (ptoas).
    pipeline.compile_pto_to_device_cpp(
        pto_path=args.pto,
        out_cpp=cce_path,
        ptoas=args.ptoas,
        arch=str(args.arch),
        memory_model=str(args.memory_model),
        insert_events=bool(args.insert_events),
        assign_tile_addrs=True,
    )

    # 2) CCE C++ -> fatobj executable (bisheng link).
    pipeline.build_fatobj_exe_from_cce(
        cce_path=cce_path,
        out_exe=exe_path,
        arch=str(args.arch),
        ascend_home=args.ascend_home,
        host_specs=host_specs,
        fixed_block_dim=fixed_block_dim,
        runtime_lib=runtime_lib,
        soc=soc_full,
        add_rpath=True,
    )

    print(f"built: {exe_path}")

    if not args.run:
        return 0

    env = dict(os.environ)
    if args.run_mode == "sim":
        log_dir = outdir / "camodel_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("CAMODEL_LOG_PATH", str(log_dir))
        env.setdefault("ASCEND_PROCESS_LOG_PATH", str(log_dir))

    cmd = [
        str(exe_path),
        "--device",
        str(int(args.device)),
        "--block-dim",
        str(int(block_dim)),
        "--warmup",
        str(int(args.warmup)),
        "--iters",
        str(int(args.iters)),
    ]
    subprocess.run(cmd, check=True, cwd=str(outdir), env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
