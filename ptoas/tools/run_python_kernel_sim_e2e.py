#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
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


def _soc_from_alias(alias: str) -> str:
    # Keep consistent with tests/script/run_st.py mapping.
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="Python kernel -> PTO-AS -> (CPU+sim NPU) run and compare.")
    ap.add_argument("--py", type=Path, required=True, help="Python kernel file")
    ap.add_argument("--kernel", help="Function name to compile (required if file has multiple defs)")
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/ptoas_py_kernel_sim"))
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas(repo))

    # Build options
    ap.add_argument("--arch", default="dav-c220-vec")
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    ap.add_argument("--split-kernels", action="store_true", help="Split mixed cube/vec kernels into staged kernels")

    # Simulator runtime options
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--soc", default="a3", help="Simulator SoC (a3|a5|Ascend910B1|Ascend910_9599|...)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument(
        "--dump-instr",
        action="store_true",
        help="Also compile+run the generated host.cpp to produce simulator dumps and summarize SET_FLAG/WAIT_FLAG.",
    )
    args = ap.parse_args()

    if not args.py.exists():
        print(f"error: kernel file not found: {args.py}", file=sys.stderr)
        return 2
    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    soc = _soc_from_alias(args.soc)
    pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc)

    source = args.py.read_text(encoding="utf-8")
    kernel_name = args.kernel
    if kernel_name is None:
        names = ast_frontend.list_kernel_functions(source)
        if len(names) != 1:
            print(f"error: multiple kernels found; pass --kernel ({', '.join(names)})", file=sys.stderr)
            return 2
        kernel_name = names[0]

    kernel = ast_frontend.compile_kernel_spec_from_source(source, func_name=kernel_name)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Emit a universal `.pto` with embedded host metadata.
    pto_path = args.outdir / f"{kernel.name}.pto"
    host_spec = binding.default_host_spec(kernel)
    pto_text = prepend_host_spec_to_pto(pto=kernel.pto, spec=host_spec)
    pto_path.write_text(pto_text, encoding="utf-8")

    # Keep generating a standalone host.cpp template for reference/debugging.
    host_cpp = args.outdir / "host.cpp"
    host_cpp.write_text(
        emit_acl_host_cpp(so_basename=f"lib{kernel.name}_sim.so", args=kernel.host_tensor_specs()),
        encoding="utf-8",
    )

    cfg = pipeline.CompileConfig(
        ptoas=args.ptoas,
        ascend_home=args.ascend_home,
        arch=args.arch,
        memory_model=args.memory_model,
        insert_events=args.insert_events,
        split_kernels=args.split_kernels,
    )
    cce_path, bin_path = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg)

    so_path = args.outdir / f"lib{kernel.name}_sim.so"
    pipeline.build_fatobj_so_from_cce(
        cce_path=cce_path,
        out_so=so_path,
        arch=cfg.arch,
        ascend_home=cfg.ascend_home,
        runtime_lib="runtime_camodel",
        soc=soc,
    )

    # Compare sim-NPU output against CPU output for the same inputs.
    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=args.block_dim, kernel_name=host_spec.kernel_name
    )
    base = pipeline.make_host_arrays(host_spec)
    cpu_arrays = [a.copy() for a in base]
    npu_arrays = [a.copy() for a in base]

    cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
    cpu_so = args.outdir / f"lib{kernel.name}_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
    cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

    err: Exception | None = None
    try:
        npu_res = pipeline.run_npu_kernel_from_so(
            so_path=so_path,
            host_spec=host_spec,
            host_arrays=npu_arrays,
            device_id=args.device,
            block_dim=args.block_dim,
        )
        npu_out = npu_res.outputs
        out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
        pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)
    except Exception as e:
        err = e

    if args.dump_instr:
        # Best-effort: even if the kernel run failed, still try to run the standalone
        # host binary to collect CAMODEL instruction dumps.
        try:
            outdir = args.outdir
            sim_lib = pipeline.resolve_ascend_simulator_lib_dir(ascend_home=args.ascend_home, soc=soc)
            host_exe = outdir / f"host_sim_{kernel.name}"

            cmd = [
                "g++",
                str(outdir / "host.cpp"),
                "-o",
                str(host_exe),
                "-O2",
                "-std=c++17",
                f"-I{args.ascend_home / 'include'}",
                f"-I{args.ascend_home / 'pkg_inc'}",
                f"-I{args.ascend_home / 'pkg_inc' / 'runtime' / 'runtime'}",
                f"-I{args.ascend_home / 'pkg_inc' / 'profiling'}",
                f"-L{args.ascend_home / 'lib64'}",
                f"-L{sim_lib}",
                f"-Wl,-rpath,{args.ascend_home / 'lib64'}",
                f"-Wl,-rpath,{sim_lib}",
                "-lruntime_camodel",
                "-lnpu_drv_camodel",
                "-lascendcl",
                "-ltiling_api",
                "-lplatform",
                "-lc_sec",
                "-ldl",
                "-lm",
                "-lstdc++",
                "-lpthread",
            ]
            subprocess.run(cmd, check=True)

            dump_dir = outdir / "host_camodel_logs"
            dump_dir.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["CAMODEL_LOG_PATH"] = str(dump_dir)
            env["LD_LIBRARY_PATH"] = f"{sim_lib}:{args.ascend_home / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
            subprocess.run(
                [str(host_exe), "--so", str(so_path), "--device", str(args.device), "--block-dim", str(args.block_dim)],
                cwd=str(outdir),
                env=env,
                check=True,
            )

            set_count = 0
            wait_count = 0
            samples: list[str] = []
            for f in sorted(dump_dir.glob("*.instr_log.dump")):
                try:
                    with f.open("r", errors="ignore") as fp:
                        for line in fp:
                            if "SET_FLAG" in line:
                                set_count += 1
                                if len(samples) < 6:
                                    samples.append(f"{f.name}: {line.strip()}")
                            if "WAIT_FLAG" in line:
                                wait_count += 1
                                if len(samples) < 6:
                                    samples.append(f"{f.name}: {line.strip()}")
                except OSError:
                    continue
            print(f"SIM dumps: {dump_dir}  SET_FLAG={set_count}  WAIT_FLAG={wait_count}")
            for s in samples:
                print("  " + s)
        except Exception as dump_e:
            print(f"warning: failed to produce SIM instruction dumps: {dump_e}", file=sys.stderr)

    if err is not None:
        raise err

    print(
        f"OK: {kernel.name} (sim soc={soc}) artifacts: "
        f"{pto_path.name}, {cce_path.name}, {bin_path.name}, {so_path.name}, {cpu_so.name}, {host_cpp.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
