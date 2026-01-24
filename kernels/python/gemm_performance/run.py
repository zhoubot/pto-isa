#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import pipeline  # noqa: E402
from ptoas.python import binding  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402

from kernels.python.gemm_performance.kernel import make_gemm_performance_kernel  # noqa: E402


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


def _summarize_camodel_set_wait_flags(*, log_dir: Path) -> tuple[int, int, list[str]]:
    """
    Best-effort scan of Ascend simulator dumps for SET_FLAG / WAIT_FLAG.
    """
    set_count = 0
    wait_count = 0
    samples: list[str] = []
    if not log_dir.exists():
        return (0, 0, [])
    for f in sorted(log_dir.rglob("*.instr_log.dump")):
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
    return (set_count, wait_count, samples)


def _read_device_f32(acl, *, dev_ptr: int, offset_bytes: int) -> float:
    import ctypes
    import struct

    buf = (ctypes.c_ubyte * 4)()
    host_ptr = ctypes.addressof(buf)
    src_ptr = int(dev_ptr) + int(offset_bytes)
    ret = acl.rt.memcpy(int(host_ptr), 4, int(src_ptr), 4, 2)  # D2H
    if int(ret) != 0:
        raise RuntimeError(f"acl.rt.memcpy(D2H, 4B) failed (ret={ret})")
    return float(struct.unpack("<f", bytes(buf))[0])


def _check_samples_device(
    *,
    acl,
    c_dev: int,
    a: np.ndarray,
    b_t: np.ndarray,
    samples: int,
    seed: int,
    rtol: float,
    atol: float,
) -> None:
    rng = np.random.default_rng(int(seed))
    m, k = a.shape
    n = b_t.shape[0]
    rs = rng.integers(0, m, size=(samples,), dtype=np.int64)
    cs = rng.integers(0, n, size=(samples,), dtype=np.int64)
    a32 = a.astype(np.float32, copy=False)
    b32 = b_t.astype(np.float32, copy=False)
    for r, col in zip(rs, cs):
        r_i = int(r)
        c_i = int(col)
        expected = float(np.dot(a32[r_i, :], b32[c_i, :]))
        offset = (r_i * n + c_i) * 4
        got = _read_device_f32(acl, dev_ptr=int(c_dev), offset_bytes=int(offset))
        if not np.isfinite(got):
            raise AssertionError(f"non-finite output at ({r_i},{c_i}): {got}")
        if not np.isclose(got, expected, rtol=float(rtol), atol=float(atol)):
            raise AssertionError(f"mismatch at ({r_i},{c_i}): got={got} expected={expected}")


def _benchmark_so(
    *,
    so_path: Path,
    device_id: int,
    block_dim: int,
    a: np.ndarray,
    b_t: np.ndarray,
    iters: int,
    warmup: int,
) -> tuple[float, float, float]:
    import ctypes
    import acl

    def _recent() -> str:
        try:
            return str(acl.get_recent_err_msg())
        except Exception:
            return ""

    def _check(ret: int, what: str) -> None:
        if int(ret) == 0:
            return
        msg = _recent()
        raise RuntimeError(f"{what} failed (ret={ret})" + (f": {msg}" if msg else ""))

    m, k = a.shape
    n = b_t.shape[0]
    c_nbytes = int(m) * int(n) * 4

    acl.init()
    acl.rt.set_device(int(device_id))
    stream, ret = acl.rt.create_stream()
    _check(ret, "acl.rt.create_stream")

    a_dev, ret = acl.rt.malloc(int(a.nbytes), 0)
    _check(ret, "acl.rt.malloc(a)")
    b_dev, ret = acl.rt.malloc(int(b_t.nbytes), 0)
    _check(ret, "acl.rt.malloc(b)")
    c_dev, ret = acl.rt.malloc(int(c_nbytes), 0)
    _check(ret, "acl.rt.malloc(c)")

    _check(acl.rt.memcpy(int(a_dev), int(a.nbytes), int(a.ctypes.data), int(a.nbytes), 1), "acl.rt.memcpy(a H2D)")
    _check(acl.rt.memcpy(int(b_dev), int(b_t.nbytes), int(b_t.ctypes.data), int(b_t.nbytes), 1), "acl.rt.memcpy(b H2D)")

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None

    def _launch():
        launch(
            ctypes.c_void_p(int(stream)),
            ctypes.c_uint32(int(block_dim)),
            ctypes.c_void_p(int(c_dev)),
            ctypes.c_void_p(int(a_dev)),
            ctypes.c_void_p(int(b_dev)),
        )

    for _ in range(int(warmup)):
        _launch()
    _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream(warmup)")

    start, ret = acl.rt.create_event()
    _check(ret, "acl.rt.create_event(start)")
    end, ret = acl.rt.create_event()
    _check(ret, "acl.rt.create_event(end)")

    _check(acl.rt.record_event(start, stream), "acl.rt.record_event(start)")
    for _ in range(int(iters)):
        _launch()
    _check(acl.rt.record_event(end, stream), "acl.rt.record_event(end)")
    _check(acl.rt.synchronize_event(end), "acl.rt.synchronize_event(end)")
    elapsed_ms, ret = acl.rt.event_elapsed_time(start, end)
    _check(ret, "acl.rt.event_elapsed_time")

    avg_ms = float(elapsed_ms) / float(max(1, int(iters)))

    # Keep output on device for sampled validation.
    acl.rt.destroy_event(start)
    acl.rt.destroy_event(end)

    return avg_ms, int(c_dev), (lambda: (acl, stream, a_dev, b_dev, c_dev))()


def _cleanup_npu(acl, stream, a_dev, b_dev, c_dev, device_id: int):
    acl.rt.free(int(a_dev))
    acl.rt.free(int(b_dev))
    acl.rt.free(int(c_dev))
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(int(device_id))
    acl.finalize()


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/python/gemm_performance: PTO-AS + ptoas --insert-events + NPU TFLOPS.")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_gemm_performance"))

    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3")
    ap.add_argument("--device", type=int, default=7)

    ap.add_argument("--m", type=int, default=6144)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--block-dim", type=int, default=24)

    ap.add_argument("--emit-bin", action="store_true", help="Also emit *.bin via ptoas (slower; not needed to benchmark).")
    ap.add_argument("--skip-build", action="store_true", help="Reuse existing built .so if present in outdir.")
    ap.add_argument("--compile-only", action="store_true", help="Only build artifacts, do not run the kernel.")

    ap.add_argument(
        "--camodel-log-path",
        type=Path,
        default=None,
        help="Simulator CAMODEL_LOG_PATH (default: <outdir>/<case>/camodel_logs when --run-mode=sim).",
    )

    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-check", dest="check", action="store_false", default=True)
    ap.add_argument("--check-samples", type=int, default=16)
    ap.add_argument("--check-rtol", type=float, default=2e-2)
    ap.add_argument("--check-atol", type=float, default=5e-2)
    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    if args.run_mode == "sim":
        soc_full = _soc_from_alias(str(args.soc))
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc_full)
        runtime_lib = "runtime_camodel"
    else:
        runtime_lib = "runtime"
        soc_full = None

    case_dir = args.outdir / f"m{int(args.m)}_n{int(args.n)}_k{int(args.k)}_bd{int(args.block_dim)}_{args.run_mode}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Build PTO-AS from Python.
    spec = make_gemm_performance_kernel(m=int(args.m), k=int(args.k), n=int(args.n))
    pto_path = case_dir / "gemm_performance.pto"
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))
    pto_path.write_text(pto_text, encoding="utf-8")

    # Compile via ptoas with --insert-events (set/wait flags insertion).
    cfg = pipeline.CompileConfig(
        ptoas=args.ptoas,
        ascend_home=args.ascend_home,
        arch="dav-c220-cube",
        memory_model="MEMORY_BASE",
        insert_events=True,
    )

    cce_path = case_dir / "gemm_performance.cpp"
    so_path = case_dir / f"libgemm_performance_{args.run_mode}.so"
    bin_path = case_dir / "gemm_performance.bin"

    if not (args.skip_build and so_path.exists()):
        # Emit CCE source (and optionally *.bin) first.
        pipeline.compile_pto_to_device_cpp(
            pto_path=pto_path,
            out_cpp=cce_path,
            ptoas=cfg.ptoas,
            arch=cfg.arch,
            memory_model=cfg.memory_model,
            insert_events=cfg.insert_events,
            assign_tile_addrs=True,
        )
        if args.emit_bin:
            pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=case_dir, cfg=cfg, out_cpp=cce_path, out_bin=bin_path)

        # Build the fatobj shared library used for launch.
        pipeline.build_fatobj_so_from_cce(
            cce_path=cce_path,
            out_so=so_path,
            arch=cfg.arch,
            ascend_home=cfg.ascend_home,
            fixed_block_dim=int(args.block_dim),
            runtime_lib=runtime_lib,
            soc=soc_full,
            cce_extra_flags=[
                "-mllvm",
                "-cce-aicore-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-function-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-record-overflow=true",
                "-mllvm",
                "-cce-aicore-addr-transform",
                "-mllvm",
                "-cce-aicore-dcci-insert-for-scalar=false",
            ],
        )

    # Keep a quick summary of inserted set/wait flags for debugging.
    try:
        summary = pipeline.summarize_cce_events(cce_path=cce_path)
        import json

        (case_dir / "event_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        snippet = pipeline.extract_cce_set_wait_lines(cce_path=cce_path, limit=200)
        (case_dir / "set_wait_snippet.txt").write_text("\n".join(snippet) + ("\n" if snippet else ""), encoding="utf-8")
    except Exception:
        pass

    if args.compile_only:
        print(f"OK: built so={so_path} outdir={case_dir}")
        return 0

    log_dir: Path | None = None
    if args.run_mode == "sim":
        log_dir = args.camodel_log_path or (case_dir / "camodel_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CAMODEL_LOG_PATH"] = str(log_dir)
        os.environ.setdefault("ASCEND_PROCESS_LOG_PATH", os.environ["CAMODEL_LOG_PATH"])

    # Host inputs.
    rng = np.random.default_rng(19)
    a_i16 = rng.integers(-1000, 1000, size=(int(args.m), int(args.k)), dtype=np.int16)
    a = a_i16.astype(np.float16, copy=False)
    a = (a / np.float16(256.0)).astype(np.float16, copy=False)
    del a_i16

    # DN tensor is backed by a physical [n, k] row-major buffer (host passes B^T contiguous).
    b_t_i16 = rng.integers(-1000, 1000, size=(int(args.n), int(args.k)), dtype=np.int16)
    b_t = b_t_i16.astype(np.float16, copy=False)
    b_t = (b_t / np.float16(256.0)).astype(np.float16, copy=False)
    del b_t_i16

    # Benchmark + optional sampled validation.
    t0 = time.time()
    avg_ms, c_dev, state = _benchmark_so(
        so_path=so_path,
        device_id=int(args.device),
        block_dim=int(args.block_dim),
        a=a,
        b_t=b_t,
        iters=int(args.iters),
        warmup=int(args.warmup),
    )
    acl, stream, a_dev, b_dev, c_dev = state
    try:
        flops = 2.0 * float(args.m) * float(args.n) * float(args.k)
        tflops = flops / ((avg_ms / 1e3) * 1.0e12)
        print(f"avg_time_ms: {avg_ms:.4f}  tflops: {tflops:.2f}  (m={args.m} n={args.n} k={args.k})")

        if args.check:
            _check_samples_device(
                acl=acl,
                c_dev=int(c_dev),
                a=a,
                b_t=b_t,
                samples=int(args.check_samples),
                seed=20,
                rtol=float(args.check_rtol),
                atol=float(args.check_atol),
            )
            print(f"check: OK (samples={int(args.check_samples)})")
    finally:
        _cleanup_npu(acl, stream, a_dev, b_dev, c_dev, device_id=int(args.device))

    if args.run_mode == "sim":
        assert log_dir is not None
        s, w, samples = _summarize_camodel_set_wait_flags(log_dir=log_dir)
        (case_dir / "camodel_set_wait_summary.txt").write_text(
            f"CAMODEL_LOG_PATH={log_dir}\nSET_FLAG={s}\nWAIT_FLAG={w}\n"
            + ("\n".join(samples) + ("\n" if samples else "")),
            encoding="utf-8",
        )
        print(f"sim_log: CAMODEL_LOG_PATH={log_dir}  SET_FLAG={s}  WAIT_FLAG={w}")

    extra = f" bin={bin_path.name}" if args.emit_bin else ""
    print(f"OK: outdir={case_dir}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
