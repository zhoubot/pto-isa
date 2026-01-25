#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import binding, pipeline  # noqa: E402
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
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


@dataclass(frozen=True)
class _Manifest:
    pto: str
    cpu_so: str
    npu_cce: str
    npu_bin: str
    npu_so: str
    host_spec_json: str


def _write_manifest(path: Path, m: _Manifest) -> None:
    path.write_text(json.dumps(m.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _child(args: argparse.Namespace) -> int:
    repo = pipeline.repo_root()

    if not args.py.exists():
        print(f"error: kernel file not found: {args.py}", file=sys.stderr)
        return 2
    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.run_mode == "sim":
        soc_full = _soc_from_alias(str(args.soc))
        # For simulator, we want the process to start with correct LD_LIBRARY_PATH.
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc_full)
        camodel_logs = args.outdir / "camodel_logs"
        camodel_logs.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CAMODEL_LOG_PATH", str(camodel_logs))
        os.environ.setdefault("ASCEND_PROCESS_LOG_PATH", str(camodel_logs))
        runtime_lib = "runtime_camodel"
        soc_for_link = soc_full
    else:
        runtime_lib = "runtime"
        soc_for_link = None

    # Python -> PTO-AS (with embedded host spec).
    spec = binding.compile_file(args.py, kernel=args.kernel)
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))

    pto_path = args.outdir / f"{spec.name}.pto"
    pto_path.write_text(pto_text, encoding="utf-8")

    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=int(args.block_dim), kernel_name=host_spec.kernel_name
    )

    # CPU reference.
    base_arrays = pipeline.make_host_arrays(host_spec)
    cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
    cpu_so = args.outdir / f"lib{spec.name}_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
    cpu_arrays = [a.copy() for a in base_arrays]
    cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

    # NPU build.
    cfg = pipeline.CompileConfig(
        ptoas=args.ptoas,
        ascend_home=args.ascend_home,
        arch=args.arch,
        memory_model=args.memory_model,
        insert_events=args.insert_events,
    )
    npu_cce, npu_bin = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg)
    npu_so = args.outdir / f"lib{spec.name}_{args.run_mode}.so"
    pipeline.build_fatobj_so_from_cce(
        cce_path=npu_cce,
        out_so=npu_so,
        arch=cfg.arch,
        ascend_home=cfg.ascend_home,
        runtime_lib=runtime_lib,
        soc=soc_for_link,
    )

    # Best-effort: keep a compact summary of set/wait flags for deadlock debugging.
    try:
        summary = pipeline.summarize_cce_events(cce_path=npu_cce)
        (args.outdir / "event_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        snippet = pipeline.extract_cce_set_wait_lines(cce_path=npu_cce, limit=200)
        (args.outdir / "set_wait_snippet.txt").write_text("\n".join(snippet) + ("\n" if snippet else ""), encoding="utf-8")
    except Exception:
        pass

    (args.outdir / "host_spec.json").write_text(host_spec.to_json(indent=2) + "\n", encoding="utf-8")

    _write_manifest(
        args.outdir / "manifest.json",
        _Manifest(
            pto=str(pto_path),
            cpu_so=str(cpu_so),
            npu_cce=str(npu_cce),
            npu_bin=str(npu_bin),
            npu_so=str(npu_so),
            host_spec_json=str(args.outdir / "host_spec.json"),
        ),
    )

    # NPU run.
    npu_arrays = [a.copy() for a in base_arrays]
    iters = int(args.bench_iters) if args.run_mode == "npu" else 0
    warmup = int(args.bench_warmup) if args.run_mode == "npu" else 0
    if iters > 0 and int(args.bench_max_bytes) > 0:
        total_bytes = sum(int(a.nbytes) for a in npu_arrays)
        if total_bytes > int(args.bench_max_bytes):
            iters = 1
            warmup = 0

    npu_res = pipeline.run_npu_kernel_from_so(
        so_path=npu_so,
        host_spec=host_spec,
        host_arrays=npu_arrays,
        device_id=int(args.device),
        block_dim=int(args.block_dim),
        bench_iters=iters,
        bench_warmup=warmup,
    )
    npu_out = npu_res.outputs
    out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
    pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)

    bench_s = ""
    if args.run_mode == "npu" and npu_res.bench is not None:
        b = npu_res.bench
        soc = npu_res.device.soc or "unknown"
        cnt_s = f",count={npu_res.device.device_count}" if npu_res.device.device_count is not None else ""
        bench_s = (
            f" npu(dev={npu_res.device.device_id},soc={soc}{cnt_s},avg={b.avg_us:.2f}us,"
            f"p50={b.p50_us:.2f}us,min={b.min_us:.2f}us,max={b.max_us:.2f}us,iters={b.iters})"
        )

    print(
        f"OK: {args.py.name}:{spec.name}{bench_s} "
        f"(pto={pto_path.name} cce={Path(npu_cce).name} bin={Path(npu_bin).name} so={npu_so.name} outdir={args.outdir})"
    )
    return 0


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(
        description="Python kernel -> PTO-AS -> ptoas -> (CPU ref + NPU run) with optional timeout and simulator fallback."
    )
    ap.add_argument("py", type=Path, help="Python kernel file (parsed by the AST frontend)")
    ap.add_argument("--kernel", default=None, help="Function name to compile (required if file has multiple defs)")
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/ptoas_python_kernel_e2e"))
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas(repo))

    ap.add_argument("--arch", default="dav-c220-vec")
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)

    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC alias when --run-mode=sim (a3|a5|...)")
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)

    ap.add_argument("--timeout-sec", type=float, default=None, help="Kill the run if it exceeds this wall time.")
    ap.add_argument("--sim-on-timeout", action="store_true", help="If NPU run times out, rerun under simulator.")
    ap.add_argument("--sim-timeout-sec", type=float, default=120.0, help="Timeout for simulator fallback.")
    ap.add_argument("--bench-iters", type=int, default=50, help="NPU-only: measure kernel time with ACL events.")
    ap.add_argument("--bench-warmup", type=int, default=10, help="NPU-only: warmup iterations before timing.")
    ap.add_argument(
        "--bench-max-bytes",
        type=int,
        default=1 << 20,
        help="If total H2D bytes exceed this, reduce benchmark to 1 iteration (avoids slow large-kernel benches).",
    )
    ap.add_argument(
        "--show-perf-apis",
        action="store_true",
        help="Print which timing/profiling APIs are present in the Ascend toolkit headers.",
    )

    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.show_perf_apis and not args._child:
        print(json.dumps(pipeline.discover_ascend_perf_apis(ascend_home=args.ascend_home), indent=2, sort_keys=True))

    if args._child or not args.timeout_sec:
        if not os.environ.get("PTOAS_VERBOSE_RUN") and args.run_mode == "sim":
            os.environ.setdefault("PTOAS_VERBOSE_RUN", "1")
        try:
            return _child(args)
        except Exception:
            sys.stderr.write(traceback.format_exc())
            return 1

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.py),
        "--kernel",
        str(args.kernel) if args.kernel is not None else "",
        "--outdir",
        str(args.outdir),
        "--ptoas",
        str(args.ptoas),
        "--arch",
        str(args.arch),
        "--memory-model",
        str(args.memory_model),
        "--run-mode",
        str(args.run_mode),
        "--soc",
        str(args.soc),
        "--ascend-home",
        str(args.ascend_home),
        "--device",
        str(int(args.device)),
        "--block-dim",
        str(int(args.block_dim)),
        "--bench-iters",
        str(int(args.bench_iters)),
        "--bench-warmup",
        str(int(args.bench_warmup)),
        "--bench-max-bytes",
        str(int(args.bench_max_bytes)),
        "--_child",
    ]
    if args.kernel is None:
        # Remove the empty --kernel "" so binding.compile_file(kernel=None) behaves correctly.
        k = cmd.index("--kernel")
        del cmd[k : k + 2]
    if not args.insert_events:
        cmd.append("--no-insert-events")

    try:
        env = dict(os.environ)
        env["_PTOAS_PY_KERNEL_E2E_CHILD"] = "1"
        proc = subprocess.run(cmd, check=False, timeout=float(args.timeout_sec), env=env)
        if proc.returncode == 0:
            return 0
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: exceeded {args.timeout_sec}s (outdir={args.outdir})", file=sys.stderr)
        if args.run_mode == "npu" and args.sim_on_timeout:
            sim_dir = args.outdir.with_name(args.outdir.name + "_sim_timeout")
            sim_cmd = cmd[:]
            # Replace outdir + run-mode.
            sim_cmd[sim_cmd.index("--outdir") + 1] = str(sim_dir)
            sim_cmd[sim_cmd.index("--run-mode") + 1] = "sim"
            env = dict(os.environ)
            env.setdefault("PTOAS_VERBOSE_RUN", "1")
            env.setdefault("PTOAS_DISABLE_RPATH", "1")
            try:
                proc2 = subprocess.run(sim_cmd, check=False, timeout=float(args.sim_timeout_sec), env=env)
                if proc2.returncode == 0:
                    print(f"OK: simulator fallback passed (outdir={sim_dir})", file=sys.stderr)
                    return 124
                print(f"FAIL: simulator fallback returned {proc2.returncode} (outdir={sim_dir})", file=sys.stderr)
                return 124
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT: simulator fallback exceeded {args.sim_timeout_sec}s (outdir={sim_dir})", file=sys.stderr)
                return 124
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
