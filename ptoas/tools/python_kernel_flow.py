#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BINDING_PY = _REPO_ROOT / "binding" / "python"
if str(_BINDING_PY) not in sys.path:
    sys.path.insert(0, str(_BINDING_PY))

from ptoas.python import ast_frontend, pipeline  # noqa: E402
from ptoas.python.host_codegen import emit_acl_host_cpp  # noqa: E402
from ptoas.python import binding  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402


def _default_ptoas(repo: Path) -> Path:
    for p in (
        repo / "ptoas/mlir/build-macos/bin/ptoas",
        repo / "ptoas/mlir/build/bin/ptoas",
    ):
        if p.exists():
            return p
    return repo / "ptoas/mlir/build/bin/ptoas"


def _select_kernel(source: str, requested: str | None) -> str:
    names = ast_frontend.list_kernel_functions(source)
    if requested:
        if requested not in names:
            raise ValueError(f"kernel not found: {requested} (available: {', '.join(names)})")
        return requested
    if len(names) == 1:
        return names[0]
    raise ValueError(f"multiple kernels found; pass --kernel ({', '.join(names)})")


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(
        description="Python kernel -> universal PTO-AS -> ptoas -> CPU cpp + NPU cpp/bin (+ optional fatobj .so)."
    )
    ap.add_argument("py", type=Path, help="Python kernel file (defines functions like add16)")
    ap.add_argument("--kernel", help="Function name to compile (required if file has multiple defs)")
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/ptoas_py_kernel"))
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas(repo))
    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="both")
    ap.add_argument("--arch", default="dav-c220-vec")
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    ap.add_argument("--no-assign-tile-addrs", dest="assign_tile_addrs", action="store_false", default=True)
    ap.add_argument("--so-basename", default=None)
    ap.add_argument("--build-so", action="store_true", help="Build fatobj .so (requires Ascend toolkit)")

    # NPU options for building fatobj .so (optional).
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    args = ap.parse_args()

    if not args.py.exists():
        print(f"error: kernel file not found: {args.py}", file=sys.stderr)
        return 2
    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    if args.target in ("npu", "both"):
        if not args.ascend_home or not args.ascend_home.exists():
            print("error: set --ascend-home or ASCEND_HOME_PATH for NPU compilation", file=sys.stderr)
            return 2

    source = args.py.read_text(encoding="utf-8")
    try:
        kernel_name = _select_kernel(source, args.kernel)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    kernel = ast_frontend.compile_kernel_spec_from_source(source, func_name=kernel_name)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pto_path = args.outdir / f"{kernel.name}.pto"
    host_spec = binding.default_host_spec(kernel)
    pto_path.write_text(prepend_host_spec_to_pto(pto=kernel.pto, spec=host_spec), encoding="utf-8")

    cpu_cpp: Path | None = None
    npu_cpp: Path | None = None
    npu_bin: Path | None = None

    if args.target in ("cpu", "both"):
        cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)

    host_cpp: Path | None = None
    so_basename = args.so_basename or f"lib{kernel.name}_npu.so"
    if args.target in ("npu", "both"):
        npu_cpp = args.outdir / f"{kernel.name}.npu.cpp"
        npu_bin = args.outdir / f"{kernel.name}.npu.bin"
        cfg = pipeline.CompileConfig(
            ptoas=args.ptoas,
            ascend_home=args.ascend_home,
            arch=args.arch,
            memory_model=args.memory_model,
            insert_events=args.insert_events,
        )
        pipeline.compile_pto_to_cce_and_bin(
            pto_path=pto_path, outdir=args.outdir, cfg=cfg, out_cpp=npu_cpp, out_bin=npu_bin
        )

        host_cpp = args.outdir / "host.cpp"
        host_cpp.write_text(
            emit_acl_host_cpp(so_basename=so_basename, args=kernel.host_tensor_specs()),
            encoding="utf-8",
        )

    if args.build_so:
        if args.target == "cpu":
            print("error: --build-so requires --target npu|both", file=sys.stderr)
            return 2
        if not args.ascend_home or not args.ascend_home.exists():
            print("error: set --ascend-home or ASCEND_HOME_PATH to build the .so", file=sys.stderr)
            return 2
        so_path = args.outdir / so_basename
        pipeline.build_fatobj_so_from_cce(
            cce_path=npu_cpp, out_so=so_path, arch=args.arch, ascend_home=args.ascend_home
        )
        print(f"built: {so_path}")

    parts = [str(pto_path)]
    if cpu_cpp is not None:
        parts.append(cpu_cpp.name)
    if npu_cpp is not None and npu_bin is not None:
        parts.extend([npu_cpp.name, npu_bin.name])
    if host_cpp is not None:
        parts.append(host_cpp.name)
    print(f"OK: {kernel.name} -> " + " + ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
