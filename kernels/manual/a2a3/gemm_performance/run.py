#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BINDING_PY = _REPO_ROOT / "frontend" / "python"
if str(_BINDING_PY) not in sys.path:
    sys.path.insert(0, str(_BINDING_PY))

from ptoas.python import pipeline  # noqa: E402


def _soc_from_alias(alias: str) -> str:
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


def _default_ascend_home() -> Path:
    p = os.environ.get("ASCEND_HOME_PATH", "").strip()
    if p:
        return Path(p)
    return pipeline.default_ascend_home()


def _run(cmd: list[str], *, timeout_sec: int = 60) -> str:
    p = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    return p.stdout or ""


def _source_ascend_env(ascend_home: Path) -> dict[str, str]:
    """
    Source Ascend `setenv.bash` and return the resulting environment.

    This matches the behavior of repo runners under `tests/script/` but is kept local
    to this example so it can be invoked directly.
    """
    setenv = ascend_home / "bin" / "setenv.bash"
    if not setenv.exists():
        # In some environments vars are already set (e.g., container images).
        return dict(os.environ)
    out = _run(
        ["bash", "-lc", f"source {setenv} >/dev/null 2>&1 && env -0"],
        timeout_sec=60,
    )
    env = dict(os.environ)
    for item in out.split("\0"):
        if not item:
            continue
        k, v = item.split("=", 1)
        env[k] = v
    return env


def _filter_ld_library_path(ld: str, *, drop_substrs: tuple[str, ...]) -> str:
    parts = [p for p in (ld or "").split(":") if p and all(s not in p for s in drop_substrs)]
    return ":".join(parts)


def _desired_env(*, run_mode: str, ascend_home: Path, soc_version: str) -> dict[str, str]:
    """
    Prepare process env so `import acl` + `ctypes.CDLL(...)` work in the selected mode.

    - npu: keep default runtime libs (device execution).
    - sim: prefer runtime stub + simulator libs (model execution).

    Important: `LD_LIBRARY_PATH` is read by the dynamic loader at process start.
    Updating `os.environ["LD_LIBRARY_PATH"]` at runtime does not reliably affect
    `dlopen()` search paths, so the runner re-execs itself with this env.
    """
    env = _source_ascend_env(ascend_home)
    env["ASCEND_HOME_PATH"] = str(ascend_home)

    if run_mode == "npu":
        # Ensure we do not accidentally pick simulator/camodel libraries from the parent shell.
        ld = env.get("LD_LIBRARY_PATH", "")
        ld = _filter_ld_library_path(
            ld,
            drop_substrs=(
                "/runtime/lib64/stub",
                "/tools/simulator/",
                "/simulator/",
            ),
        )
        env["LD_LIBRARY_PATH"] = ld
        return env
    if run_mode != "sim":
        raise ValueError(f"unsupported run_mode: {run_mode}")

    ld = env.get("LD_LIBRARY_PATH", "")
    # Avoid accidentally picking non-stub runtime libs when sim is requested.
    ld = _filter_ld_library_path(ld, drop_substrs=("/runtime/lib64",))
    runtime_stub = ascend_home / "runtime" / "lib64" / "stub"
    sim_lib = ascend_home / "tools" / "simulator" / soc_version / "lib"
    if not runtime_stub.exists():
        raise FileNotFoundError(f"runtime stub not found: {runtime_stub}")
    if not sim_lib.exists():
        raise FileNotFoundError(f"simulator lib not found: {sim_lib}")
    env["LD_LIBRARY_PATH"] = f"{runtime_stub}:{sim_lib}:{ld}"
    return env


def _run_subprocess(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout_sec: int) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=int(timeout_sec),
    )
    return p.stdout or ""


def run_sim_via_cmake(*, soc_version: str, ascend_home: Path, work_dir: Path, timeout_sec: int = 7200) -> bool:
    """
    Run the example in CA model (sim mode) via the CMake-built binary.

    The Python `acl` bindings are usually built against the real runtime and can
    fail to import under camodel/stub library paths. For sim correctness, using
    the repo's CMake flow is the most robust option.
    """
    src_root = Path(__file__).resolve().parent
    # Match `run.sh` for this example: just prepend simulator libs.
    env = _source_ascend_env(ascend_home)
    env["ASCEND_HOME_PATH"] = str(ascend_home)
    sim_lib = ascend_home / "tools" / "simulator" / soc_version / "lib"
    if not sim_lib.exists():
        raise FileNotFoundError(f"simulator lib not found: {sim_lib}")
    env["LD_LIBRARY_PATH"] = f"{sim_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    # Default: keep simulator quiet (no heavy dumps / waves).
    for k in (
        "CAMODEL_LOG_PATH",
        "PTO_ST_LOGS",
        "PTO_ST_VERBOSE",
        "PTO_ST_LOGS_ON_PASS",
    ):
        env.pop(k, None)

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] sim: generating input+golden ...")
    _run_subprocess([sys.executable, str((src_root / "tests/scripts" / "gen_data.py").resolve())], cwd=work_dir, env=env, timeout_sec=600)

    build_dir = work_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] sim: configuring (cmake) ...")
    _run_subprocess(
        ["cmake", "-DRUN_MODE=sim", f"-DSOC_VERSION={soc_version}", str(src_root)],
        cwd=build_dir,
        env=env,
        timeout_sec=1800,
    )
    jobs = str(min(os.cpu_count() or 4, 16))
    print(f"[INFO] sim: building (make -j{jobs}) ...")
    _run_subprocess(["make", "-j", jobs], cwd=build_dir, env=env, timeout_sec=3600)

    print("[INFO] sim: running ./gemm_performance (may take a long time) ...")
    t0 = time.perf_counter()
    try:
        out = _run_subprocess(["./gemm_performance"], cwd=build_dir, env=env, timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired:
        elapsed_s = time.perf_counter() - t0
        print(f"[WARN] sim timed out after {int(timeout_sec)}s (wall={elapsed_s:.2f}s)")
        print(f"[INFO] sim artifacts dir: {work_dir}")
        return False
    elapsed_s = time.perf_counter() - t0
    print(out.strip())
    print(f"sim_wall_time_sec: {elapsed_s:.2f}")
    print(f"[INFO] sim artifacts dir: {work_dir}")
    if "test success" not in out:
        raise RuntimeError("sim run finished but did not report `test success`")
    return True


def _emit_cce_wrapper(
    *,
    kernel_cpp: Path,
    block_dim: int,
    m: int,
    k: int,
    n: int,
    single_core_m: int,
    single_core_k: int,
    single_core_n: int,
    base_m: int,
    base_k: int,
    base_n: int,
    step_m: int,
    step_ka: int,
    step_kb: int,
    step_n: int,
) -> str:
    # Keep defaults identical to `main.cpp` / `LaunchGEMME2E` in this example.
    return (
        "#define MEMORY_BASE\n"
        "#include \"kernel_operator.h\"\n"
        f'#include "{kernel_cpp.as_posix()}"\n\n'
        'extern "C" __global__ AICORE void pto_kernel_gemm_performance(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)\n'
        "{\n"
        f"    constexpr uint32_t blockDim = {int(block_dim)};\n"
        f"    constexpr uint32_t m = {int(m)};\n"
        f"    constexpr uint32_t k = {int(k)};\n"
        f"    constexpr uint32_t n = {int(n)};\n"
        f"    constexpr uint32_t singleCoreM = {int(single_core_m)};\n"
        f"    constexpr uint32_t singleCoreK = {int(single_core_k)};\n"
        f"    constexpr uint32_t singleCoreN = {int(single_core_n)};\n"
        f"    constexpr uint32_t baseM = {int(base_m)};\n"
        f"    constexpr uint32_t baseK = {int(base_k)};\n"
        f"    constexpr uint32_t baseN = {int(base_n)};\n"
        f"    constexpr uint32_t stepM = {int(step_m)};\n"
        f"    constexpr uint32_t stepKa = {int(step_ka)};\n"
        f"    constexpr uint32_t stepKb = {int(step_kb)};\n"
        f"    constexpr uint32_t stepN = {int(step_n)};\n"
        "    RunGemmE2E<float, half, half, float, blockDim, m, k, n, m, k, n,\n"
        "        singleCoreM, singleCoreK, singleCoreN,\n"
        "        baseM, baseK, baseN,\n"
        "        stepM, stepKa, stepKb, stepN>(\n"
        "        reinterpret_cast<__gm__ float *>(out),\n"
        "        reinterpret_cast<__gm__ half *>(src0),\n"
        "        reinterpret_cast<__gm__ half *>(src1));\n"
        "}\n"
    )


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_meta(path: Path, meta: dict[str, Any]) -> None:
    import json

    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _wrapper_hash(wrapper_src: str) -> str:
    import hashlib

    return hashlib.sha256(wrapper_src.encode("utf-8")).hexdigest()


def _ensure_so_matches(*, out_so: Path, expected: dict[str, Any]) -> None:
    meta_path = out_so.with_suffix(out_so.suffix + ".meta.json")
    meta = _load_meta(meta_path)
    for k, v in expected.items():
        if str(meta.get(k, "")) != str(v):
            raise RuntimeError(
                f"kernel SO metadata mismatch for {out_so} ({k} expected={v} got={meta.get(k, '')}); "
                "rebuild without --skip-build (or use a different --out-so)"
            )

def build_so(
    *,
    out_so: Path,
    ascend_home: Path,
    arch: str,
    runtime_lib: str,
    fixed_block_dim: int,
    wrapper_src: str,
    soc: str | None,
    force: bool,
) -> None:
    wrapper_hash = _wrapper_hash(wrapper_src)
    if out_so.exists() and not force:
        meta_path = out_so.with_suffix(out_so.suffix + ".meta.json")
        meta = _load_meta(meta_path)
        if meta.get("runtime_lib") == runtime_lib and meta.get("arch") == arch and meta.get("fixed_block_dim") == int(
            fixed_block_dim
        ):
            # Best-effort: if wrapper content didn't change, skip rebuild.
            if meta.get("wrapper_hash") == wrapper_hash and (meta.get("soc") or "") == (str(soc) if soc else ""):
                return

    # Keep builds quiet by default (print only on failure).
    os.environ.setdefault("PTOAS_QUIET", "1")

    kernel_cpp = Path(__file__).resolve().parent / "gemm_performance_kernel.cpp"
    if not kernel_cpp.exists():
        raise FileNotFoundError(f"kernel not found: {kernel_cpp}")

    out_so.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gemm_perf_py_") as td:
        td_path = Path(td)
        cce_path = td_path / "gemm_performance.cce.cpp"
        cce_path.write_text(wrapper_src, encoding="utf-8")
        pipeline.build_fatobj_so_from_cce(
            cce_path=cce_path,
            out_so=out_so,
            arch=arch,
            ascend_home=ascend_home,
            fixed_block_dim=int(fixed_block_dim),
            runtime_lib=str(runtime_lib),
            soc=str(soc) if soc is not None else None,
            # Mirror `kernels/manual/a2a3/gemm_performance/CMakeLists.txt` defaults.
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
        meta_path = out_so.with_suffix(out_so.suffix + ".meta.json")
        _save_meta(
            meta_path,
            {
                "arch": str(arch),
                "runtime_lib": str(runtime_lib),
                "fixed_block_dim": int(fixed_block_dim),
                "wrapper_hash": wrapper_hash,
                "soc": str(soc) if soc is not None else "",
            },
        )


def _make_inputs(*, m: int, k: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Use integer RNG -> f16 cast (fast and deterministic, avoids large float32 temporaries).
    rng = np.random.default_rng(int(seed))
    a_i16 = rng.integers(low=-1000, high=1000, size=(m, k), dtype=np.int16)
    b_i16 = rng.integers(low=-1000, high=1000, size=(n, k), dtype=np.int16)  # stored as B^T (n,k)
    a = (a_i16.astype(np.float16) / np.float16(256.0)).astype(np.float16, copy=False)
    b_t = (b_i16.astype(np.float16) / np.float16(256.0)).astype(np.float16, copy=False)
    return a, b_t


def _sample_check_host(*, a: np.ndarray, b_t: np.ndarray, c: np.ndarray, samples: int, seed: int, rtol: float, atol: float) -> None:
    rng = np.random.default_rng(int(seed))
    m, k = a.shape
    n = b_t.shape[0]
    # Random (row, col) pairs; compute dot(A[row], B_T[col]) in float32.
    rs = rng.integers(0, m, size=(samples,), dtype=np.int64)
    cs = rng.integers(0, n, size=(samples,), dtype=np.int64)
    a32 = a.astype(np.float32, copy=False)
    b32 = b_t.astype(np.float32, copy=False)
    for r, col in zip(rs, cs):
        expected = float(np.dot(a32[int(r), :], b32[int(col), :]))
        got = float(c[int(r), int(col)])
        # Heuristic tolerance: fp16 inputs accumulated to fp32.
        # Keep conservative to avoid false failures across toolkit versions.
        if not np.isfinite(got):
            raise AssertionError(f"non-finite output at ({int(r)},{int(col)}): {got}")
        if not np.isclose(got, expected, rtol=float(rtol), atol=float(atol)):
            raise AssertionError(f"mismatch at ({int(r)},{int(col)}): got={got} expected={expected}")


def _read_device_f32(acl, *, dev_ptr: int, offset_bytes: int) -> float:
    import ctypes
    import struct

    buf = (ctypes.c_ubyte * 4)()
    host_ptr = ctypes.addressof(buf)
    src_ptr = int(dev_ptr) + int(offset_bytes)
    # 2 == ACL_MEMCPY_DEVICE_TO_HOST
    ret = acl.rt.memcpy(int(host_ptr), 4, int(src_ptr), 4, 2)
    if int(ret) != 0:
        raise RuntimeError(f"acl.rt.memcpy(D2H, 4B) failed (ret={ret})")
    return float(struct.unpack("<f", bytes(buf))[0])


def _sample_check_device(
    *,
    acl,
    out_dev: int,
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
        offset = (r_i * n + c_i) * 4  # float32
        got = _read_device_f32(acl, dev_ptr=int(out_dev), offset_bytes=int(offset))
        if not np.isfinite(got):
            raise AssertionError(f"non-finite output at ({r_i},{c_i}): {got}")
        if not np.isclose(got, expected, rtol=float(rtol), atol=float(atol)):
            raise AssertionError(f"mismatch at ({r_i},{c_i}): got={got} expected={expected}")


def _time_kernel_ms(
    *,
    acl,
    stream,
    launch,
    block_dim: int,
    out_dev: int,
    a_dev: int,
    b_dev: int,
    iters: int,
) -> list[float]:
    """
    Measure kernel time using ACL events (device timestamps).

    This avoids host-side sync overhead dominating small iteration counts and is
    closer to what instruction-level profilers report.
    """
    start_evt, ret = acl.rt.create_event()
    if int(ret) != 0:
        raise RuntimeError(f"acl.rt.create_event(start) failed (ret={ret})")
    end_evt, ret = acl.rt.create_event()
    if int(ret) != 0:
        raise RuntimeError(f"acl.rt.create_event(end) failed (ret={ret})")
    try:
        times_ms: list[float] = []
        for _ in range(int(iters)):
            ret = acl.rt.record_event(start_evt, stream)
            if int(ret) != 0:
                raise RuntimeError(f"acl.rt.record_event(start) failed (ret={ret})")
            launch(stream, int(block_dim), out_dev, a_dev, b_dev)
            ret = acl.rt.record_event(end_evt, stream)
            if int(ret) != 0:
                raise RuntimeError(f"acl.rt.record_event(end) failed (ret={ret})")
            ret = acl.rt.synchronize_event(end_evt)
            if int(ret) != 0:
                raise RuntimeError(f"acl.rt.synchronize_event failed (ret={ret})")
            ms, ret = acl.rt.event_elapsed_time(start_evt, end_evt)
            if int(ret) != 0:
                raise RuntimeError(f"acl.rt.event_elapsed_time failed (ret={ret})")
            times_ms.append(float(ms))
        return times_ms
    finally:
        acl.rt.destroy_event(start_evt)
        acl.rt.destroy_event(end_evt)


def run_acl(
    *,
    so_path: Path,
    run_mode: str,
    soc_version: str,
    ascend_home: Path,
    device: int,
    warmup: int,
    iters: int,
    check: bool,
    check_samples: int,
    seed: int,
    m: int,
    k: int,
    n: int,
    block_dim: int,
    timing: str,
    check_rtol: float,
    check_atol: float,
    check_full: bool,
) -> None:
    import ctypes
    import acl

    a, b_t = _make_inputs(m=m, k=k, n=n, seed=seed)
    out = np.empty((m, n), dtype=np.float32)

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

    acl.init()
    acl.rt.set_device(int(device))
    stream, ret = acl.rt.create_stream()
    _check(ret, "acl.rt.create_stream")

    a_dev, ret = acl.rt.malloc(int(a.nbytes), 0)
    _check(ret, "acl.rt.malloc(a)")
    b_dev, ret = acl.rt.malloc(int(b_t.nbytes), 0)
    _check(ret, "acl.rt.malloc(b)")
    out_dev, ret = acl.rt.malloc(int(out.nbytes), 0)
    _check(ret, "acl.rt.malloc(out)")

    _check(acl.rt.memcpy(a_dev, int(a.nbytes), int(a.ctypes.data), int(a.nbytes), 1), "acl.rt.memcpy(a H2D)")
    _check(acl.rt.memcpy(b_dev, int(b_t.nbytes), int(b_t.ctypes.data), int(b_t.nbytes), 1), "acl.rt.memcpy(b H2D)")

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None
    launch_wrapped = lambda s, bd, o, aa, bb: launch(
        ctypes.c_void_p(int(s)),
        ctypes.c_uint32(int(bd)),
        ctypes.c_void_p(int(o)),
        ctypes.c_void_p(int(aa)),
        ctypes.c_void_p(int(bb)),
    )

    # Warmup.
    for _ in range(int(warmup)):
        launch_wrapped(stream, block_dim, out_dev, a_dev, b_dev)
        _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")

    # Timed.
    times_ms: list[float]
    if timing == "event":
        try:
            times_ms = _time_kernel_ms(
                acl=acl,
                stream=stream,
                launch=launch_wrapped,
                block_dim=int(block_dim),
                out_dev=int(out_dev),
                a_dev=int(a_dev),
                b_dev=int(b_dev),
                iters=int(iters),
            )
        except Exception:
            # Fallback to host timing if events are not supported in the current runtime.
            timing = "host"
            times_ms = []
    if timing == "host":
        times_ms = []
        for _ in range(int(iters)):
            t0 = time.perf_counter()
            launch_wrapped(stream, block_dim, out_dev, a_dev, b_dev)
            _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")
            times_ms.append((time.perf_counter() - t0) * 1e3)

    flops = 2.0 * float(m) * float(n) * float(k)

    ts = sorted(float(x) for x in times_ms if float(x) > 0.0)
    avg_ms = float(sum(ts) / max(1, len(ts))) if ts else float("nan")
    min_ms = float(min(ts)) if ts else float("nan")
    med_ms = float(ts[len(ts) // 2]) if ts else float("nan")
    p90_ms = float(ts[int(0.9 * (len(ts) - 1))]) if len(ts) >= 2 else med_ms

    def _tflops(ms: float) -> float:
        if not np.isfinite(ms) or ms <= 0:
            return float("nan")
        return flops / ((ms / 1e3) * 1.0e12)

    print(f"timing: {timing}  warmup: {warmup}  iters: {iters}")
    print(f"time_ms: avg={avg_ms:.4f}  min={min_ms:.4f}  p50={med_ms:.4f}  p90={p90_ms:.4f}")
    print(f"tflops:  avg={_tflops(avg_ms):.2f}  max={_tflops(min_ms):.2f}  (m={m} n={n} k={k}, fp16*fp16->fp32)")

    if check:
        if bool(check_full):
            # Full numpy validation is only practical for small sizes.
            # Guard against accidentally doing a 6144^3 numpy matmul.
            max_flops = float(os.environ.get("PTO_GEMM_PERF_FULLCHECK_MAX_FLOPS", "2e9"))
            flops = 2.0 * float(m) * float(n) * float(k)
            if flops > max_flops:
                raise RuntimeError(
                    f"refusing full numpy check for large GEMM (flops={flops:.3e} > {max_flops:.3e}); "
                    "use --check-samples instead or lower m/k/n"
                )
            _check(acl.rt.memcpy(int(out.ctypes.data), int(out.nbytes), out_dev, int(out.nbytes), 2), "acl.rt.memcpy(out D2H)")
            # b_t is stored as B^T (n,k), so numpy reference is A @ B = A @ (B^T)^T.
            ref = (a.astype(np.float32, copy=False) @ b_t.astype(np.float32, copy=False).T).astype(np.float32, copy=False)
            if not np.allclose(out, ref, rtol=float(check_rtol), atol=float(check_atol)):
                diff = np.max(np.abs(out - ref))
                raise AssertionError(f"full check failed: max_abs_diff={float(diff)}")
            print("check: OK (full)")
        else:
            # Sample-based check reads only a few float32s from device to keep performance runs fast.
            _sample_check_device(
                acl=acl,
                out_dev=int(out_dev),
                a=a,
                b_t=b_t,
                samples=int(check_samples),
                seed=int(seed) + 1,
                rtol=float(check_rtol),
                atol=float(check_atol),
            )
            print(f"check: OK (samples={int(check_samples)})")

    acl.rt.free(a_dev)
    acl.rt.free(b_dev)
    acl.rt.free(out_dev)
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(int(device))
    acl.finalize()


def main() -> int:
    default_work_dir = Path(os.environ.get("PTO_GEMM_PERF_WORK_DIR", "/tmp/pto-isa-gemm-performance")).resolve()

    ap = argparse.ArgumentParser(
        description="Python runner for kernels/manual/a2a3/gemm_performance (sim + NPU)."
    )
    ap.add_argument("-r", "--run-mode", choices=["sim", "npu"], default="npu")
    ap.add_argument("-v", "--soc-version", default="a3",
                    help="SOC version (e.g. Ascend910B1) or alias {a3,a5}; used for simulator libs in sim mode")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--arch", default="dav-c220-cube")
    ap.add_argument("--ascend-home", type=Path, default=_default_ascend_home())
    ap.add_argument("--work-dir", type=Path, default=default_work_dir,
                    help="artifact root dir (default: /tmp/...); writes to <work-dir>/{npu,sim}/")
    ap.add_argument("--out-so", type=Path, default=None,
                    help="output kernel .so; default depends on --run-mode")
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="skip kernel SO build (assumes --out-so exists)")
    ap.add_argument("--timing", choices=["event", "host"], default="event",
                    help="timing source: device events (recommended) or host wall time")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-check", dest="check", action="store_false", default=True)
    ap.add_argument("--check-samples", type=int, default=16)
    ap.add_argument("--check-rtol", type=float, default=2e-2)
    ap.add_argument("--check-atol", type=float, default=5e-2)
    ap.add_argument("--check-full", action="store_true",
                    help="full numpy matmul validation (only feasible for small m/k/n)")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--m", type=int, default=6144)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--grid-m", type=int, default=4, help="core partitioning along M (default: 4)")
    ap.add_argument("--grid-n", type=int, default=6, help="core partitioning along N (default: 6)")
    ap.add_argument("--sim-timeout-sec", type=int, default=int(os.environ.get("PTO_GEMM_PERF_SIM_TIMEOUT_SEC", "7200")),
                    help="sim mode run timeout in seconds (CA model can be very slow)")
    args = ap.parse_args()

    if not args.ascend_home.exists():
        print(f"error: ascend home not found: {args.ascend_home}", file=sys.stderr)
        return 2

    soc_version = _soc_from_alias(str(args.soc_version).strip())
    work_root = Path(args.work_dir).resolve()
    sim_work_dir = (work_root / "sim").resolve()
    npu_work_dir = (work_root / "npu").resolve()
    if args.run_mode == "sim":
        ok = run_sim_via_cmake(
            soc_version=str(soc_version),
            ascend_home=args.ascend_home,
            work_dir=sim_work_dir,
            timeout_sec=int(args.sim_timeout_sec),
        )
        return 0 if ok else 3

    # NPU path: ensure dynamic-loader paths are applied at process start (prevents
    # parent-shell simulator paths from interfering with `acl` bindings).
    if os.environ.get("PTO_GEMM_PERF_ENV_READY", "") != "1":
        env = _desired_env(run_mode="npu", ascend_home=args.ascend_home, soc_version=str(soc_version))
        env["PTO_GEMM_PERF_ENV_READY"] = "1"
        argv = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]
        os.execve(sys.executable, argv, env)

    m = int(args.m)
    k = int(args.k)
    n = int(args.n)
    grid_m = int(args.grid_m)
    grid_n = int(args.grid_n)
    block_dim = grid_m * grid_n

    if m <= 0 or k <= 0 or n <= 0:
        print("error: m/k/n must be positive", file=sys.stderr)
        return 2
    if (m % grid_m) != 0 or (n % grid_n) != 0:
        print(f"error: m must be divisible by grid-m and n by grid-n (m={m} grid-m={grid_m}, n={n} grid-n={grid_n})",
              file=sys.stderr)
        return 2

    # Default tiling matches the tuned configuration in README/main.cpp.
    base_m, base_k, base_n = 128, 64, 256
    step_m, step_ka, step_kb, step_n = 1, 4, 4, 1
    single_core_m = m // grid_m
    single_core_n = n // grid_n
    single_core_k = k

    if single_core_m % base_m != 0 or single_core_n % base_n != 0 or single_core_k % base_k != 0:
        print(
            "error: derived single-core shapes must be multiples of base tile sizes "
            f"(singleCoreM={single_core_m}, singleCoreN={single_core_n}, singleCoreK={single_core_k})",
            file=sys.stderr,
        )
        return 2

    npu_work_dir.mkdir(parents=True, exist_ok=True)

    out_so = args.out_so
    if out_so is None:
        out_so = (npu_work_dir / "libgemm_performance.so").resolve()

    kernel_cpp = Path(__file__).resolve().parent / "gemm_performance_kernel.cpp"
    wrapper_src = _emit_cce_wrapper(
        kernel_cpp=kernel_cpp,
        block_dim=int(block_dim),
        m=int(m),
        k=int(k),
        n=int(n),
        single_core_m=int(single_core_m),
        single_core_k=int(single_core_k),
        single_core_n=int(single_core_n),
        base_m=int(base_m),
        base_k=int(base_k),
        base_n=int(base_n),
        step_m=int(step_m),
        step_ka=int(step_ka),
        step_kb=int(step_kb),
        step_n=int(step_n),
    )
    runtime_lib = "runtime" if args.run_mode == "npu" else "runtime_camodel"
    expected_meta = {
        "arch": str(args.arch),
        "runtime_lib": str(runtime_lib),
        "fixed_block_dim": str(int(block_dim)),
        "soc": str(soc_version),
        "wrapper_hash": _wrapper_hash(wrapper_src),
    }

    if not args.skip_build:
        build_so(
            out_so=out_so,
            ascend_home=args.ascend_home,
            arch=str(args.arch),
            runtime_lib=str(runtime_lib),
            fixed_block_dim=int(block_dim),
            wrapper_src=wrapper_src,
            soc=str(soc_version),
            force=bool(args.force_build),
        )
    else:
        if not out_so.exists():
            print(f"error: kernel SO not found: {out_so}", file=sys.stderr)
            return 2
        _ensure_so_matches(out_so=out_so, expected=expected_meta)

    run_acl(
        so_path=out_so,
        run_mode=str(args.run_mode),
        soc_version=str(soc_version),
        ascend_home=args.ascend_home,
        device=int(args.device),
        warmup=int(args.warmup),
        iters=int(args.iters),
        check=bool(args.check),
        check_samples=int(args.check_samples),
        seed=int(args.seed),
        m=int(m),
        k=int(k),
        n=int(n),
        block_dim=int(block_dim),
        timing=str(args.timing),
        check_rtol=float(args.check_rtol),
        check_atol=float(args.check_atol),
        check_full=bool(args.check_full),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
