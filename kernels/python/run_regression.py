#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import binding, pipeline  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    py: Path
    kernel: str
    arch: str
    input_scale: float | None = None
    block_dim: int | None = None


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


def _cases() -> list[Case]:
    base = Path(__file__).resolve().parent
    return [
        # Core kernels.
        Case(name="add16", py=base / "add16.py", kernel="add16", arch="dav-c220-vec"),
        Case(name="mul16_f16", py=base / "mul16_f16.py", kernel="mul16_f16", arch="dav-c220-vec"),
        Case(name="mul16", py=base / "mul16.py", kernel="mul16", arch="dav-c220-vec"),
        Case(name="sub16", py=base / "sub16.py", kernel="sub16", arch="dav-c220-vec"),
        Case(name="neg16", py=base / "neg16.py", kernel="neg16", arch="dav-c220-vec"),
        Case(name="scale16", py=base / "scale16.py", kernel="scale16", arch="dav-c220-vec"),
        Case(name="bias16", py=base / "bias16.py", kernel="bias16", arch="dav-c220-vec"),
        Case(name="transpose16", py=base / "transpose16.py", kernel="transpose16", arch="dav-c220-vec"),
        Case(name="tiled_transpose64", py=base / "tiled_transpose64.py", kernel="tiled_transpose64", arch="dav-c220-vec"),
        Case(name="abs16", py=base / "abs16.py", kernel="abs16", arch="dav-c220-vec"),
        Case(name="abs_add16", py=base / "abs_add16.py", kernel="abs_add16", arch="dav-c220-vec"),
        Case(name="axpy16", py=base / "axpy16.py", kernel="axpy16", arch="dav-c220-vec"),
        Case(name="fma16", py=base / "fma16.py", kernel="fma16", arch="dav-c220-vec"),
        Case(name="rowmax16", py=base / "rowmax16.py", kernel="rowmax16", arch="dav-c220-vec"),
        Case(name="rowsum16", py=base / "rowsum16.py", kernel="rowsum16", arch="dav-c220-vec"),
        Case(name="tiled_rowsum64", py=base / "tiled_rowsum64.py", kernel="tiled_rowsum64", arch="dav-c220-vec"),
        Case(name="tiled_add128", py=base / "tiled_add128.py", kernel="tiled_add128", arch="dav-c220-vec"),
        # SPMD / multi-block stress tests (per-case block_dim).
        Case(
            name="spmd_tiled_add256",
            py=base / "spmd_tiled_add256.py",
            kernel="spmd_tiled_add256",
            arch="dav-c220-vec",
            block_dim=8,
        ),
        Case(
            name="spmd_tiled_transpose256",
            py=base / "spmd_tiled_transpose256.py",
            kernel="spmd_tiled_transpose256",
            arch="dav-c220-vec",
            block_dim=8,
        ),
        Case(
            name="spmd_tiled_rowsum256",
            py=base / "spmd_tiled_rowsum256.py",
            kernel="spmd_tiled_rowsum256",
            arch="dav-c220-vec",
            block_dim=8,
        ),
        # NOTE: TEXP is approx on NPU; scale inputs down to keep CPU-vs-NPU drift small.
        Case(name="sinh16", py=base / "sinh16.py", kernel="sinh16", arch="dav-c220-vec", input_scale=0.05),
        Case(name="softmax16", py=base / "softmax16.py", kernel="softmax16", arch="dav-c220-vec"),
        Case(name="softmax32x16", py=base / "softmax32x16.py", kernel="softmax32x16", arch="dav-c220-vec"),
        Case(name="gemm16", py=base / "gemm16.py", kernel="gemm16", arch="dav-c220-cube"),

        # Ported from `~/github/pto-isa/examples/*.py` (kept runnable end-to-end here).
        Case(name="pto_isa_sinh", py=base / "pto_isa_sinh.py", kernel="pto_isa_sinh", arch="dav-c220-vec", input_scale=0.05),
        Case(name="pto_fused_softmax", py=base / "pto_fused_softmax.py", kernel="pto_fused_softmax", arch="dav-c220-vec"),
        Case(name="pto_aten_ir_primitives", py=base / "pto_aten_ir_primitives.py", kernel="pto_aten_ir_primitives", arch="dav-c220-vec"),
        Case(name="pto_torch_tensor", py=base / "pto_torch_tensor.py", kernel="pto_torch_tensor", arch="dav-c220-vec"),
        Case(name="pto_torch_functional", py=base / "pto_torch_functional.py", kernel="pto_torch_functional", arch="dav-c220-vec"),
        Case(name="pto_torch_nn_operators", py=base / "pto_torch_nn_operators.py", kernel="pto_torch_nn_operators", arch="dav-c220-cube"),
        Case(name="pto_torch_flexattention", py=base / "pto_torch_flexattention.py", kernel="pto_torch_flexattention", arch="dav-c220-vec"),
        Case(name="pto_llama7B_dynamic", py=base / "pto_llama7B_dynamic.py", kernel="pto_llama7B_dynamic", arch="dav-c220-vec"),
    ]


def _run_kernel_file_e2e(
    *,
    case: Case,
    outdir: Path,
    ptoas: Path,
    ascend_home: Path,
    run_mode: str,
    soc: str,
    device: int,
    block_dim: int,
    memory_model: str,
    insert_events: bool,
) -> None:
    # Compile Python -> PTO-AS.
    spec = binding.compile_file(case.py, kernel=case.kernel)
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))

    outdir.mkdir(parents=True, exist_ok=True)
    pto_path = outdir / f"{spec.name}.pto"
    pto_path.write_text(pto_text, encoding="utf-8")

    # CPU reference.
    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=block_dim, kernel_name=host_spec.kernel_name
    )
    base_arrays = pipeline.make_host_arrays(host_spec)
    if case.input_scale is not None:
        for i, a in enumerate(host_spec.args):
            if a.role == "out":
                continue
            if base_arrays[i].dtype in (np.float16, np.float32):
                base_arrays[i] = (base_arrays[i].astype(np.float32) * float(case.input_scale)).astype(base_arrays[i].dtype)

    cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=outdir, ptoas=ptoas)
    cpu_so = outdir / f"lib{spec.name}_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
    cpu_arrays = [a.copy() for a in base_arrays]
    cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

    # NPU run (sim or real).
    if run_mode == "sim":
        camodel = outdir / "camodel_logs"
        camodel.mkdir(parents=True, exist_ok=True)
        os.environ["CAMODEL_LOG_PATH"] = str(camodel)
        soc_full = _soc_from_alias(soc)
        runtime_lib = "runtime_camodel"
    else:
        soc_full = None
        runtime_lib = "runtime"

    cfg = pipeline.CompileConfig(
        ptoas=ptoas,
        ascend_home=ascend_home,
        arch=case.arch,
        memory_model=memory_model,
        insert_events=insert_events,
    )
    cce_cpp, _bin = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=outdir, cfg=cfg)
    # Emit a small summary of inserted events for debugging deadlocks.
    try:
        summary = pipeline.summarize_cce_events(cce_path=cce_cpp)
        (outdir / "event_summary.txt").write_text(str(summary) + "\n", encoding="utf-8")
        snippet = pipeline.extract_cce_set_wait_lines(cce_path=cce_cpp, limit=200)
        (outdir / "set_wait_snippet.txt").write_text("\n".join(snippet) + ("\n" if snippet else ""), encoding="utf-8")
        if run_mode == "sim":
            sys.stdout.write(
                f"  events: set={summary.get('set_total')} wait={summary.get('wait_total')} (see {outdir / 'event_summary.txt'})\n"
            )
            sys.stdout.write(f"  set/wait snippet: {outdir / 'set_wait_snippet.txt'}\n")
    except Exception:
        pass
    npu_so = outdir / f"lib{spec.name}_{run_mode}.so"
    pipeline.build_fatobj_so_from_cce(
        cce_path=cce_cpp,
        out_so=npu_so,
        arch=cfg.arch,
        ascend_home=cfg.ascend_home,
        runtime_lib=runtime_lib,
        soc=soc_full,
    )

    npu_arrays = [a.copy() for a in base_arrays]
    npu_out = pipeline.run_npu_kernel_from_so(
        so_path=npu_so, host_spec=host_spec, host_arrays=npu_arrays, device_id=device, block_dim=block_dim
    )
    out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
    if case.name in ("sinh16", "pto_isa_sinh"):
        # `sinh(x) = (exp(x) - exp(-x)) / 2` is numerically unstable for small x, and
        # NPU vector subtraction may introduce larger relative error when subtracting
        # near-equal values. Keep this tolerance slightly looser than the default f32
        # settings so the regression is robust.
        for i, (c, n, dt) in enumerate(zip(cpu_out, npu_out, out_dtypes)):
            if dt != "f32":
                raise AssertionError(f"unexpected output dtype for {case.name}: output {i} is {dt}")
            np.testing.assert_allclose(n, c, rtol=2e-2, atol=3e-2, err_msg=f"output {i} ({dt}) mismatch")
    else:
        pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="Run kernels/python examples end-to-end with progress output.")
    ap.add_argument("--verbose-build", action="store_true", help="Print compiler commands/warnings")
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC alias when --run-mode=sim (a3|a5|...)")
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas(repo))
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernels_python_regression"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    ap.add_argument("--filter", default=None, help="Substring filter for case names")
    ap.add_argument("--cases", default=None, help="Comma-separated case names (overrides --filter)")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--timeout-sec", type=float, default=None, help="Per-case wall-time timeout (kills hung NPU runs).")
    ap.add_argument("--sim-on-timeout", action="store_true", help="If NPU run times out, rerun the same case in sim mode.")
    ap.add_argument("--retries", type=int, default=1, help="Retry a failing case up to N times (helps with flaky NPU runs).")
    args = ap.parse_args()

    if not args.verbose_build:
        os.environ.setdefault("PTOAS_QUIET", "1")

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    # In timeout mode we spawn child processes; do not re-exec the parent via ensure_ascend_sim_env().
    if args.run_mode == "sim" and not args.timeout_sec:
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=_soc_from_alias(args.soc))

    cases = _cases()
    if args.cases:
        want = [s.strip() for s in args.cases.split(",") if s.strip()]
        want_set = set(want)
        cases = [c for c in cases if c.name in want_set]
        missing = [n for n in want if n not in {c.name for c in cases}]
        if missing:
            print("error: unknown cases: " + ", ".join(missing), file=sys.stderr)
            print("available: " + ", ".join(c.name for c in _cases()), file=sys.stderr)
            return 2
    elif args.filter:
        cases = [c for c in cases if args.filter in c.name]
    if not cases:
        print("error: no cases selected", file=sys.stderr)
        return 2

    passed: list[str] = []
    failed: list[str] = []
    timed_out: list[str] = []

    print(f"ptoas: {args.ptoas}")
    print(f"ascend_home: {args.ascend_home}")
    print(f"run_mode: {args.run_mode}  device: {args.device}  block_dim: {args.block_dim}")
    print(f"outdir: {args.outdir}")
    print(f"cases: {', '.join(c.name for c in cases)}\n")

    t0 = time.perf_counter()
    for i, c in enumerate(cases, start=1):
        case_outdir = args.outdir / c.name
        case_block_dim = int(c.block_dim) if c.block_dim is not None else int(args.block_dim)
        prefix = f"[{i:>2}/{len(cases)}] {c.name:<20} (block_dim={case_block_dim})"
        case_t0 = time.perf_counter()
        try:
            if args.timeout_sec and float(args.timeout_sec) > 0:
                # Run each case in a child process so we can time out hung NPU runs.
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--run-mode",
                    str(args.run_mode),
                    "--soc",
                    str(args.soc),
                    "--ascend-home",
                    str(args.ascend_home),
                    "--ptoas",
                    str(args.ptoas),
                    "--outdir",
                    str(args.outdir),
                    "--device",
                    str(int(args.device)),
                    "--block-dim",
                    str(int(case_block_dim)),
                    "--memory-model",
                    str(args.memory_model),
                    "--retries",
                    "0",
                    "--cases",
                    c.name,
                ]
                if args.verbose_build:
                    cmd.append("--verbose-build")
                if args.keep_going:
                    cmd.append("--keep-going")
                if not args.insert_events:
                    cmd.append("--no-insert-events")

                retries = max(0, int(args.retries))
                attempt = 0
                while True:
                    sys.stdout.write(
                        f"{prefix} RUN (timeout={float(args.timeout_sec)}s"
                        + (f", attempt={attempt + 1}/{retries + 1}" if retries else "")
                        + ")\n"
                    )
                    sys.stdout.flush()
                    try:
                        proc = subprocess.run(cmd, check=False, timeout=float(args.timeout_sec))
                    except subprocess.TimeoutExpired:
                        timed_out.append(c.name)
                        case_dt = time.perf_counter() - case_t0
                        sys.stdout.write(f"{prefix} TIMEOUT ({case_dt:.1f}s)\n")
                        sys.stdout.write(f"outdir: {case_outdir}\n")
                        sys.stdout.flush()

                        if args.run_mode == "npu" and args.sim_on_timeout:
                            sim_outdir = args.outdir.with_name(args.outdir.name + "_timeout_sim")
                            sim_cmd = list(cmd)
                            sim_cmd[sim_cmd.index("--run-mode") + 1] = "sim"
                            sim_cmd[sim_cmd.index("--outdir") + 1] = str(sim_outdir)
                            env = dict(os.environ)
                            env.setdefault("PTOAS_VERBOSE_RUN", "1")
                            env.setdefault("PTOAS_DISABLE_RPATH", "1")
                            try:
                                subprocess.run(sim_cmd, check=False, timeout=120.0, env=env)
                            except subprocess.TimeoutExpired:
                                pass
                        break

                    if int(proc.returncode) == 0:
                        case_dt = time.perf_counter() - case_t0
                        total_dt = time.perf_counter() - t0
                        sys.stdout.write(f"{prefix} OK   ({case_dt:.1f}s, {total_dt:.1f}s total)\n")
                        passed.append(c.name)
                        break

                    attempt += 1
                    if attempt > retries:
                        raise RuntimeError(f"child returned {proc.returncode}")
                    sys.stdout.write(f"{prefix} RETRY (child returned {proc.returncode})\n")
                    sys.stdout.flush()
                    time.sleep(0.5)

                if c.name in passed or c.name in timed_out:
                    continue

            retries = max(0, int(args.retries))
            attempt = 0
            while True:
                sys.stdout.write(
                    f"{prefix} RUN"
                    + (f" (attempt={attempt + 1}/{retries + 1})" if retries else "")
                    + "\n"
                )
                sys.stdout.flush()
                try:
                    _run_kernel_file_e2e(
                        case=c,
                        outdir=case_outdir,
                        ptoas=args.ptoas,
                        ascend_home=args.ascend_home,
                        run_mode=args.run_mode,
                        soc=args.soc,
                        device=args.device,
                        block_dim=case_block_dim,
                        memory_model=args.memory_model,
                        insert_events=args.insert_events,
                    )
                    case_dt = time.perf_counter() - case_t0
                    total_dt = time.perf_counter() - t0
                    sys.stdout.write(f"{prefix} OK   ({case_dt:.1f}s, {total_dt:.1f}s total)\n")
                    passed.append(c.name)
                    break
                except Exception:
                    attempt += 1
                    if attempt > retries:
                        raise
                    sys.stdout.write(f"{prefix} RETRY\n")
                    sys.stdout.flush()
                    time.sleep(0.5)
        except Exception:
            case_dt = time.perf_counter() - case_t0
            failed.append(c.name)
            sys.stdout.write(f"{prefix} FAIL ({case_dt:.1f}s)\n")
            sys.stdout.write(f"outdir: {case_outdir}\n")
            sys.stdout.write(traceback.format_exc())
            sys.stdout.flush()
            if not args.keep_going:
                break

    total_dt = time.perf_counter() - t0
    print(f"\nSummary: {len(passed)} passed, {len(failed)} failed, {len(timed_out)} timed out, total {total_dt:.1f}s")
    if failed:
        print("Failed cases: " + ", ".join(failed))
        return 1
    if timed_out:
        print("Timed out cases: " + ", ".join(timed_out))
        return 124
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
