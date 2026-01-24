from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .host_spec import HostSpec, HostTensorArg, infer_host_spec_from_pto, parse_host_spec_from_pto


_EVENT_RE = re.compile(
    r"\b(?P<kind>set_flag|wait_flag)\(\s*(?P<src>PIPE_[A-Z0-9_]+)\s*,\s*(?P<dst>PIPE_[A-Z0-9_]+)\s*,\s*"
    r"static_cast<event_t>\(\s*(?P<tok>\d+)\s*\)\s*\)"
)


def summarize_cce_events(*, cce_path: Path) -> dict[str, object]:
    """
    Best-effort summary of inserted event ops from generated CCE C++.

    This is meant as a quick sanity check for simulator/NPU runs where missing or
    mismatched set/wait pairs can lead to hangs.
    """
    text = cce_path.read_text(encoding="utf-8", errors="replace")
    matches = list(_EVENT_RE.finditer(text))
    by_edge: dict[tuple[str, str], dict[str, set[int] | int]] = {}
    total_set = 0
    total_wait = 0
    for m in matches:
        kind = m.group("kind")
        src = m.group("src")
        dst = m.group("dst")
        tok = int(m.group("tok"))
        key = (src, dst)
        st = by_edge.setdefault(key, {"set": set(), "wait": set(), "set_n": 0, "wait_n": 0})
        if kind == "set_flag":
            total_set += 1
            st["set_n"] = int(st["set_n"]) + 1
            cast = st["set"]
            assert isinstance(cast, set)
            cast.add(tok)
        else:
            total_wait += 1
            st["wait_n"] = int(st["wait_n"]) + 1
            cast = st["wait"]
            assert isinstance(cast, set)
            cast.add(tok)

    edges: list[dict[str, object]] = []
    for (src, dst), st in sorted(by_edge.items()):
        edges.append(
            {
                "src": src,
                "dst": dst,
                "set_n": int(st["set_n"]),
                "wait_n": int(st["wait_n"]),
                "set_tokens": sorted(int(x) for x in (st["set"] or [])),
                "wait_tokens": sorted(int(x) for x in (st["wait"] or [])),
            }
        )

    return {
        "set_total": total_set,
        "wait_total": total_wait,
        "edges": edges,
    }


def extract_cce_set_wait_lines(*, cce_path: Path, limit: int = 200) -> list[str]:
    """
    Extract a few `set_flag(...)` / `wait_flag(...)` lines from generated CCE.
    """
    if limit <= 0:
        return []
    out: list[str] = []
    for ln in cce_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if "set_flag(" in s or "wait_flag(" in s:
            out.append(s)
            if len(out) >= limit:
                break
    return out

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ascend_include_dirs(ascend_home: Path) -> list[str]:
    candidates = [
        ascend_home / "compiler/ascendc/include/basic_api",
        ascend_home / "compiler/ascendc/include/basic_api/impl",
        ascend_home / "compiler/asc/include/basic_api",
        ascend_home / "compiler/asc/include/interface",
        ascend_home / "compiler/asc",
        ascend_home / "include/ascendc",
        ascend_home / "include",
        ascend_home / "runtime/include",
    ]
    return [str(p) for p in candidates if p.exists()]


def _run(cmd: list[str], *, cwd: Path) -> None:
    quiet = os.environ.get("PTOAS_QUIET", "") in ("1", "true", "True", "yes", "YES")
    if not quiet:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode == 0:
        return
    if p.stdout:
        sys.stderr.write(p.stdout)
    if p.stderr:
        sys.stderr.write(p.stderr)
    raise subprocess.CalledProcessError(p.returncode, cmd)


def _resolve_bisheng(ascend_home: Path) -> str:
    for p in (
        ascend_home / "compiler/ccec_compiler/bin/bisheng",
        ascend_home / "compiler/bin/bisheng",
        ascend_home / "bin/bisheng",
    ):
        if p.exists():
            return str(p)
    return "bisheng"


@dataclass(frozen=True)
class CompileConfig:
    ptoas: Path
    ascend_home: Path
    arch: str
    memory_model: str = "MEMORY_BASE"
    insert_events: bool = True


def _source_setenv_bash(setenv_path: Path) -> dict[str, str]:
    # Capture the environment after sourcing setenv.bash. Use -0 for robust parsing.
    if not setenv_path.exists():
        return {}
    cmd = f"source {setenv_path} >/dev/null 2>&1 && env -0"
    p = subprocess.run(
        ["bash", "-lc", cmd],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = p.stdout.split(b"\x00")
    env: dict[str, str] = {}
    for entry in out:
        if not entry:
            continue
        k, sep, v = entry.partition(b"=")
        if sep != b"=":
            continue
        env[k.decode("utf-8", errors="ignore")] = v.decode("utf-8", errors="ignore")
    return env


def ensure_ascend_sim_env(*, ascend_home: Path, soc: str) -> None:
    """
    Ensure the *current process* is started with a simulator-friendly environment.

    Important:
    - glibc's dynamic loader reads `LD_LIBRARY_PATH` at process start; mutating
      `os.environ['LD_LIBRARY_PATH']` inside Python is not sufficient for later
      `dlopen()` calls (e.g. `ctypes.CDLL`, `import acl`).
    - Therefore, when running on simulator from Python, we re-exec the current
      process once with the desired environment.
    """
    if os.environ.get("_PTOAS_SIM_ENV_READY", "") == "1":
        return

    env = dict(os.environ)
    env["_PTOAS_SIM_ENV_READY"] = "1"

    # Keep consistent toolchain/root.
    env["ASCEND_HOME_PATH"] = str(ascend_home)

    # Pull in toolkit defaults (compiler/runtime/HCCL/OPP paths, etc.)
    setenv_path = ascend_home / "bin" / "setenv.bash"
    try:
        sourced = _source_setenv_bash(setenv_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to source {setenv_path}: {e.stderr.decode('utf-8', errors='ignore')}") from e
    env.update(sourced)
    env["ASCEND_HOME_PATH"] = str(ascend_home)

    # Prefer LD_LIBRARY_PATH over any embedded RPATH in fatobj .so's.
    env.setdefault("PTOAS_DISABLE_RPATH", "1")

    # Default to quiet runs (show build/run output only on failure).
    env.setdefault("PTOAS_QUIET", "1")

    # More useful progress logs when debugging under simulator.
    env.setdefault("PTOAS_VERBOSE_RUN", "0")

    # Simulator logs: use CAMODEL_LOG_PATH if available (used by Ascend simulator runtime).
    env.setdefault("CAMODEL_LOG_PATH", "/tmp/camodel_logs")
    try:
        Path(env["CAMODEL_LOG_PATH"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Ascend logging runtime (libascendalog.so): default to file logs only.
    verbose_run = env.get("PTOAS_VERBOSE_RUN", "") in ("1", "true", "True", "yes", "YES")
    if verbose_run:
        env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
        env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "2")  # 0=trace, 2=info
    else:
        env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    env.setdefault("ASCEND_PROCESS_LOG_PATH", env["CAMODEL_LOG_PATH"])
    try:
        Path(env["ASCEND_PROCESS_LOG_PATH"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    sim_lib = resolve_ascend_simulator_lib_dir(ascend_home=ascend_home, soc=soc)
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{sim_lib}:{ld}" if ld else str(sim_lib)

    if env.get("PTO_USE_RUNTIME_STUB", "0") == "1":
        stub = ascend_home / "runtime/lib64/stub"
        if stub.exists():
            env["LD_LIBRARY_PATH"] = f"{stub}:{env.get('LD_LIBRARY_PATH', '')}"

    # Re-exec under the new environment (so the dynamic loader sees LD_LIBRARY_PATH).
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def configure_ascend_sim_env(*, ascend_home: Path, soc: str) -> None:
    """
    Configure process env vars for Ascend simulator runs.

    This is intended to mirror `tests/script/run_st.py`:
      - source `$ASCEND_HOME_PATH/bin/setenv.bash` (if present)
      - add `$ASCEND_HOME_PATH/tools/simulator/<soc>/lib` (or arch-specific equivalents)

    Optional:
      - if `PTO_USE_RUNTIME_STUB=1`, also prepend `$ASCEND_HOME_PATH/runtime/lib64/stub`.
        Some toolkit builds may crash if runtime stubs take precedence, so this is
        opt-in instead of default.

    Note: this function mutates `os.environ` in-process. For Python flows that use
    `dlopen()` (ctypes / `import acl`), prefer `ensure_ascend_sim_env()` so the
    process starts with the desired `LD_LIBRARY_PATH`.
    """
    if not ascend_home or not ascend_home.exists():
        raise ValueError("ascend_home must exist")

    # Many scripts key off ASCEND_HOME_PATH; ensure it is set consistently.
    os.environ["ASCEND_HOME_PATH"] = str(ascend_home)

    # Prefer LD_LIBRARY_PATH (stub + simulator libs) over any embedded RPATH in fatobj .so's.
    os.environ.setdefault("PTOAS_DISABLE_RPATH", "1")

    # More useful progress logs when running under simulator.
    os.environ.setdefault("PTOAS_VERBOSE_RUN", "1")

    # Simulator logs: use CAMODEL_LOG_PATH if available (used by Ascend simulator runtime).
    # Keep a stable default so users get logs even if callers don't set it.
    os.environ.setdefault("CAMODEL_LOG_PATH", "/tmp/camodel_logs")
    try:
        Path(os.environ["CAMODEL_LOG_PATH"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Ascend logging: prefer printing + per-process log directory.
    # These env vars are parsed by Ascend logging runtime (libascendalog.so).
    os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
    os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "2")  # 0=trace, 2=info (matches simulator toml defaults)
    os.environ.setdefault("ASCEND_PROCESS_LOG_PATH", os.environ["CAMODEL_LOG_PATH"])
    try:
        Path(os.environ["ASCEND_PROCESS_LOG_PATH"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # setenv.bash may set ASCEND_HOME_PATH and a bunch of runtime/compiler vars (including LD_LIBRARY_PATH).
    setenv_path = ascend_home / "bin/setenv.bash"
    try:
        sourced = _source_setenv_bash(setenv_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to source {setenv_path}: {e.stderr.decode('utf-8', errors='ignore')}") from e
    for k, v in sourced.items():
        os.environ[k] = v

    # Ensure ASCEND_HOME_PATH survives setenv overrides.
    os.environ["ASCEND_HOME_PATH"] = str(ascend_home)

    sim_lib = resolve_ascend_simulator_lib_dir(ascend_home=ascend_home, soc=soc)

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{sim_lib}:{ld}" if ld else str(sim_lib)

    if os.environ.get("PTO_USE_RUNTIME_STUB", "0") == "1":
        stub = ascend_home / "runtime/lib64/stub"
        if stub.exists():
            os.environ["LD_LIBRARY_PATH"] = f"{stub}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def resolve_ascend_simulator_lib_dir(*, ascend_home: Path, soc: str) -> Path:
    sim_lib_candidates = (
        ascend_home / "tools" / "simulator" / soc / "lib",
        ascend_home / "tools" / "simulator" / soc / "lib64",
        ascend_home / "simulator" / soc / "lib",
        ascend_home / "simulator" / soc / "lib64",
        ascend_home / "aarch64-linux" / "simulator" / soc / "lib",
        ascend_home / "aarch64-linux" / "simulator" / soc / "lib64",
        ascend_home / "x86_64-linux" / "simulator" / soc / "lib",
        ascend_home / "x86_64-linux" / "simulator" / soc / "lib64",
    )
    sim_lib = next((p for p in sim_lib_candidates if p.exists()), None)
    if sim_lib is None:
        raise RuntimeError(
            "simulator lib dir not found (check --soc and your toolkit install). Tried:\n"
            + "\n".join(f"  - {p}" for p in sim_lib_candidates)
        )
    return sim_lib


def _sanitize_kernel_name(name: str) -> str:
    """
    Convert an arbitrary stem into a C identifier suitable for `--kernel-name`.
    """
    out: list[str] = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    if not s:
        return "pto_kernel"
    if s[0].isdigit():
        s = "_" + s
    return s


def compile_pto_to_cce_and_bin(
    *, pto_path: Path, outdir: Path, cfg: CompileConfig, out_cpp: Path | None = None, out_bin: Path | None = None
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    # This is still CCE source compiled via `bisheng -xcce`; we use `.cpp` for editor compatibility.
    cce_path = out_cpp or (outdir / (pto_path.stem + ".cpp"))
    bin_path = out_bin or (outdir / (pto_path.stem + ".bin"))

    args = [
        str(cfg.ptoas),
        str(pto_path),
        "--target",
        "npu",
        "-o",
        str(cce_path),
        "--kernel-name",
        _sanitize_kernel_name(f"pto_kernel_{pto_path.stem}"),
        "--arch",
        cfg.arch,
        "--memory-model",
        cfg.memory_model,
        "--repo-root",
        str(repo_root()),
        "--ascend-home",
        str(cfg.ascend_home),
        f"--emit-bin={bin_path}",
    ]
    if not cfg.insert_events:
        args.append("--no-insert-events")
    _run(args, cwd=repo_root())
    return cce_path, bin_path


def compile_pto_to_cpu_cpp(*, pto_path: Path, outdir: Path, ptoas: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cpp_path = outdir / (pto_path.stem + ".cpu.cpp")
    _run(
        [
            str(ptoas),
            str(pto_path),
            "--target",
            "cpu",
            "-o",
            str(cpp_path),
            "--kernel-name",
            _sanitize_kernel_name(f"pto_kernel_{pto_path.stem}"),
            "--repo-root",
            str(repo_root()),
        ],
        cwd=repo_root(),
    )
    return cpp_path


def compile_pto_to_device_cpp(
    *,
    pto_path: Path,
    out_cpp: Path,
    ptoas: Path,
    arch: str,
    memory_model: str = "MEMORY_BASE",
    insert_events: bool = True,
    assign_tile_addrs: bool = True,
) -> Path:
    out_cpp.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(ptoas),
        str(pto_path),
        "--target",
        "npu",
        "-o",
        str(out_cpp),
        "--kernel-name",
        _sanitize_kernel_name(f"pto_kernel_{pto_path.stem}"),
        "--arch",
        arch,
        "--memory-model",
        memory_model,
        "--repo-root",
        str(repo_root()),
    ]
    if not insert_events:
        args.append("--no-insert-events")
    if assign_tile_addrs:
        args.append("--assign-tile-addrs")
    _run(args, cwd=repo_root())
    return out_cpp


def build_cpu_so_from_cpp(*, cpp_path: Path, out_so: Path) -> None:
    def _host_cxx() -> tuple[str, str]:
        # Keep this robust on older distros (e.g. GCC 10 doesn't accept `-std=c++23`).
        cxx = os.environ.get("CXX", "").strip()
        if cxx:
            return cxx, "-std=c++23"
        if shutil.which("clang++"):
            return "clang++", "-std=c++23"
        if shutil.which("g++"):
            return "g++", "-std=gnu++2b"
        return "c++", "-std=gnu++2b"

    out_so.parent.mkdir(parents=True, exist_ok=True)
    cxx, std = _host_cxx()
    _run(
        [
            cxx,
            "-shared",
            "-fPIC",
            "-O2",
            std,
            "-Wno-unknown-attributes",
            "-Wno-ignored-attributes",
            f"-I{repo_root() / 'include'}",
            str(cpp_path),
            "-o",
            str(out_so),
        ],
        cwd=repo_root(),
    )


def build_fatobj_so_from_cce(
    *,
    cce_path: Path,
    out_so: Path,
    arch: str,
    ascend_home: Path,
    fixed_block_dim: int | None = None,
    runtime_lib: str = "runtime",
    soc: str | None = None,
    cce_extra_flags: list[str] | None = None,
) -> None:
    include_dirs = ascend_include_dirs(ascend_home) + [str(repo_root() / "include")]

    kernel_src = cce_path.read_text(encoding="utf-8")

    def strip_outer_cce_guard(text: str) -> str:
        # Newer `ptoas` emits CCE sources wrapped in:
        #   #if defined(__CCE_AICORE__)
        #   ...
        #   #endif
        #
        # For fatobj compilation, we compile with `bisheng -xcce` already; stripping the outer
        # guard avoids host-side compilation variants where the kernel body is skipped, which
        # would otherwise leave an unresolved `pto_kernel` reference in the linked `.so`.
        lines = text.splitlines()
        # Find first non-empty, non-comment line.
        first = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if not s or s.startswith("//"):
                continue
            first = i
            break
        if first is None:
            return text
        # Find last non-empty line.
        last = None
        for i in range(len(lines) - 1, -1, -1):
            s = lines[i].strip()
            if not s:
                continue
            last = i
            break
        if last is None:
            return text
        if lines[first].strip() != "#if defined(__CCE_AICORE__)":
            return text
        if lines[last].strip() != "#endif":
            return text
        new_lines = lines[:first] + lines[first + 1 : last] + lines[last + 1 :]
        return "\n".join(new_lines).rstrip() + "\n"

    kernel_src = strip_outer_cce_guard(kernel_src)

    m = re.search(r'extern\s+"C"\s+__global__\s+AICORE\s+void\s+(\w+)\s*\(([^)]*)\)', kernel_src)
    if not m:
        raise RuntimeError(f"failed to infer kernel signature from: {cce_path}")
    kernel_name = m.group(1)
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    arg_count = len(params)

    host_params = ", ".join([f"void *arg{i}" for i in range(arg_count)])
    kernel_args = ", ".join([f"(GM_ADDR)arg{i}" for i in range(arg_count)])

    if fixed_block_dim is not None:
        if fixed_block_dim <= 0:
            raise ValueError("fixed_block_dim must be > 0")
        block_expr = str(int(fixed_block_dim))
        block_param = "uint32_t blockDim"
        block_unused = "    (void)blockDim;\n"
    else:
        block_expr = "blockDim"
        block_param = "uint32_t blockDim"
        block_unused = ""

    combined = (
        "#include \"kernel.cpp\"\n"
        "#include <cstdint>\n\n"
        f"extern \"C\" void ptoas_launch(void *stream, {block_param}{', ' if arg_count else ''}{host_params})\n"
        "{\n"
        f"{block_unused}"
        f"    {kernel_name}<<<{block_expr}, nullptr, stream>>>({kernel_args});\n"
        "}\n"
    )

    with tempfile.TemporaryDirectory(prefix="ptoas_so_") as td:
        td_path = Path(td)
        (td_path / "kernel.cpp").write_text(kernel_src, encoding="utf-8")
        combined_path = td_path / "combined.cpp"
        combined_path.write_text(combined, encoding="utf-8")
        combined_o = td_path / "combined.o"

        bisheng = _resolve_bisheng(ascend_home)
        # Match CMake examples: optimize by default for performance-sensitive kernels.
        common = [bisheng, "-xcce", f"--cce-aicore-arch={arch}", "-std=c++17", "-fPIC", "-O2"]
        if cce_extra_flags:
            common += list(cce_extra_flags)
        incs = [f"-I{d}" for d in include_dirs]
        _run(common + incs + ["-c", str(combined_path), "-o", str(combined_o)], cwd=td_path)

        out_so.parent.mkdir(parents=True, exist_ok=True)
        link = [bisheng, "-shared", "--cce-fatobj-link", "-o", str(out_so), str(combined_o)]
        lib64 = ascend_home / "lib64"
        if lib64.exists():
            link += [f"-L{lib64}"]
            # For simulator runs we usually want the stub runtime to win via LD_LIBRARY_PATH.
            # Some linkers emit DT_RPATH (higher priority than LD_LIBRARY_PATH), which can
            # accidentally force loading the real runtime and hang/crash under simulator.
            disable_rpath = os.environ.get("PTOAS_DISABLE_RPATH", "") in ("1", "true", "True", "yes", "YES")
            if not disable_rpath:
                link += [f"-Wl,-rpath,{lib64}"]
        if runtime_lib not in ("runtime", "runtime_camodel", "runtime_cmodel"):
            raise ValueError(f"unsupported runtime_lib: {runtime_lib}")

        if runtime_lib in ("runtime_camodel", "runtime_cmodel"):
            if not soc:
                raise ValueError(f"soc must be provided when runtime_lib={runtime_lib}")
            sim_lib = resolve_ascend_simulator_lib_dir(ascend_home=ascend_home, soc=soc)
            link += [f"-L{sim_lib}"]

        link += [
            f"-l{runtime_lib}",
            "-lascendcl",
            "-ltiling_api",
            "-lplatform",
            "-lc_sec",
            "-ldl",
            "-lm",
            "-lstdc++",
        ]
        _run(link, cwd=td_path)


def _acl_h2d() -> int:
    return 1  # ACL_MEMCPY_HOST_TO_DEVICE


def _acl_d2h() -> int:
    return 2  # ACL_MEMCPY_DEVICE_TO_HOST


def _np_dtype(dtype: str) -> np.dtype:
    if dtype == "f16":
        return np.dtype(np.float16)
    if dtype == "f32":
        return np.dtype(np.float32)
    if dtype == "i32":
        return np.dtype(np.int32)
    if dtype == "u32":
        return np.dtype(np.uint32)
    raise ValueError(f"unsupported dtype: {dtype}")


def _default_tol(dtype: str) -> tuple[float, float]:
    if dtype == "f16":
        return (0.0, 0.0)
    if dtype == "f32":
        return (1e-2, 2e-2)
    return (0.0, 0.0)


def make_host_arrays(spec: HostSpec) -> list[np.ndarray]:
    rng = np.random.default_rng(int(spec.seed))
    arrays: list[np.ndarray] = []
    for a in spec.args:
        dt = _np_dtype(a.dtype)
        if a.role == "out":
            arr = np.zeros(a.shape, dtype=dt)
        else:
            # Keep values small to reduce numerical drift across backends.
            if dt == np.float16 or dt == np.float32:
                arr = (rng.random(a.shape, dtype=np.float32) - 0.5).astype(dt)
            else:
                arr = rng.integers(low=0, high=7, size=a.shape, dtype=dt)
        arrays.append(arr)
    return arrays


def parse_or_default_host_spec(*, pto_text: str) -> HostSpec:
    spec = parse_host_spec_from_pto(pto_text)
    if spec is not None:
        return spec
    return infer_host_spec_from_pto(pto=pto_text)


def run_cpu_kernel_from_so(*, so_path: Path, host_spec: HostSpec, host_arrays: list[np.ndarray]) -> list[np.ndarray]:
    import ctypes

    lib = ctypes.CDLL(str(so_path))
    fn = lib.pto_kernel_cpu
    fn.argtypes = [ctypes.c_void_p] * len(host_arrays)
    fn.restype = None

    args = [ctypes.c_void_p(int(a.ctypes.data)) for a in host_arrays]
    fn(*args)

    out: list[np.ndarray] = []
    for i in host_spec.output_indices():
        out.append(np.array(host_arrays[i], copy=True))
    return out


def run_npu_kernel_from_so(
    *,
    so_path: Path,
    host_spec: HostSpec,
    host_arrays: list[np.ndarray],
    device_id: int,
    block_dim: int,
) -> list[np.ndarray]:
    import ctypes
    import acl
    import time

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

    verbose = os.environ.get("PTOAS_VERBOSE_RUN", "") in ("1", "true", "True", "yes", "YES")

    def _log(msg: str) -> None:
        if not verbose:
            return
        sys.stderr.write(f"[ptoas][run] {msg}\n")
        sys.stderr.flush()

    stream = None
    dev_ptrs: list[int] = []
    try:
        t0 = time.perf_counter()
        _log("acl.init() ...")
        acl.init()
        _log(f"acl.init() OK ({time.perf_counter() - t0:.2f}s)")
        _log(f"acl.rt.set_device({device_id}) ...")
        acl.rt.set_device(device_id)
        _log("acl.rt.set_device OK")
        _log("acl.rt.create_stream() ...")
        stream, ret = acl.rt.create_stream()
        _check(ret, "acl.rt.create_stream")
        _log("acl.rt.create_stream OK")

        for i, a in enumerate(host_arrays):
            _log(f"acl.rt.malloc(arg{i}, {int(a.nbytes)} bytes) ...")
            p, r = acl.rt.malloc(int(a.nbytes), 0)
            _check(r, f"acl.rt.malloc(arg{i})")
            dev_ptrs.append(int(p))
            _log(f"acl.rt.malloc(arg{i}) OK (ptr=0x{int(p):x})")

        for i, (a, dev) in enumerate(zip(host_arrays, dev_ptrs)):
            if host_spec.args[i].role == "out":
                continue
            _log(f"acl.rt.memcpy(arg{i} H2D, {int(a.nbytes)} bytes) ...")
            _check(
                acl.rt.memcpy(dev, int(a.nbytes), int(a.ctypes.data), int(a.nbytes), _acl_h2d()),
                f"acl.rt.memcpy(arg{i} H2D)",
            )
            _log(f"acl.rt.memcpy(arg{i} H2D) OK")

        _log(f"ctypes.CDLL({so_path}) ...")
        lib = ctypes.CDLL(str(so_path))
        launch = lib.ptoas_launch
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * len(dev_ptrs)
        launch.restype = None
        _log(f"launch(block_dim={block_dim}, argc={len(dev_ptrs)}) ...")
        launch(ctypes.c_void_p(stream), int(block_dim), *[ctypes.c_void_p(p) for p in dev_ptrs])
        _log("launch OK")

        _log("acl.rt.synchronize_stream() ...")
        _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")
        _log("acl.rt.synchronize_stream OK")

        out: list[np.ndarray] = []
        for i in host_spec.output_indices():
            a = host_arrays[i]
            tmp = np.empty_like(a)
            _log(f"acl.rt.memcpy(arg{i} D2H, {int(tmp.nbytes)} bytes) ...")
            _check(
                acl.rt.memcpy(int(tmp.ctypes.data), int(tmp.nbytes), dev_ptrs[i], int(tmp.nbytes), _acl_d2h()),
                f"acl.rt.memcpy(arg{i} D2H)",
            )
            _log(f"acl.rt.memcpy(arg{i} D2H) OK")
            out.append(tmp)
        return out
    finally:
        for p in dev_ptrs:
            try:
                acl.rt.free(int(p))
            except Exception:
                pass
        if stream is not None:
            try:
                acl.rt.destroy_stream(stream)
            except Exception:
                pass
        try:
            acl.rt.reset_device(device_id)
        except Exception:
            pass
        try:
            acl.finalize()
        except Exception:
            pass


def compare_cpu_and_npu_outputs(
    *,
    cpu_out: list[np.ndarray],
    npu_out: list[np.ndarray],
    out_dtypes: list[str],
) -> None:
    if len(cpu_out) != len(npu_out) or len(cpu_out) != len(out_dtypes):
        raise ValueError("output list length mismatch")
    for i, (c, n, dt) in enumerate(zip(cpu_out, npu_out, out_dtypes)):
        rtol, atol = _default_tol(dt)
        np.testing.assert_allclose(n, c, rtol=rtol, atol=atol, err_msg=f"output {i} ({dt}) mismatch")

def run_add16_from_so(*, so_path: Path, device_id: int, block_dim: int) -> None:
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

    x = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    y = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    expected = (x + y).astype(np.float16)
    out = np.empty_like(x)

    acl.init()
    acl.rt.set_device(device_id)
    stream, ret = acl.rt.create_stream()
    _check(ret, "acl.rt.create_stream")

    x_dev, ret = acl.rt.malloc(x.nbytes, 0)
    _check(ret, "acl.rt.malloc(x)")
    y_dev, ret = acl.rt.malloc(y.nbytes, 0)
    _check(ret, "acl.rt.malloc(y)")
    out_dev, ret = acl.rt.malloc(out.nbytes, 0)
    _check(ret, "acl.rt.malloc(out)")

    _check(acl.rt.memcpy(x_dev, x.nbytes, int(x.ctypes.data), x.nbytes, _acl_h2d()), "acl.rt.memcpy(x H2D)")
    _check(acl.rt.memcpy(y_dev, y.nbytes, int(y.ctypes.data), y.nbytes, _acl_h2d()), "acl.rt.memcpy(y H2D)")

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None
    launch(ctypes.c_void_p(stream), block_dim, ctypes.c_void_p(x_dev), ctypes.c_void_p(y_dev), ctypes.c_void_p(out_dev))

    _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")
    _check(acl.rt.memcpy(int(out.ctypes.data), out.nbytes, out_dev, out.nbytes, _acl_d2h()), "acl.rt.memcpy(out D2H)")

    np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    acl.rt.free(x_dev)
    acl.rt.free(y_dev)
    acl.rt.free(out_dev)
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(device_id)
    acl.finalize()


def run_gemm16_from_so(*, so_path: Path, device_id: int, block_dim: int) -> None:
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

    a = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    b = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    # Avoid relying on NumPy's matmul/dot (can be broken depending on the local build/toolchain).
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    expected = (a32[:, :, None] * b32[None, :, :]).sum(axis=1).astype(np.float32)
    out = np.empty((16, 16), dtype=np.float32)

    acl.init()
    acl.rt.set_device(device_id)
    stream, ret = acl.rt.create_stream()
    _check(ret, "acl.rt.create_stream")

    a_dev, ret = acl.rt.malloc(a.nbytes, 0)
    _check(ret, "acl.rt.malloc(a)")
    b_dev, ret = acl.rt.malloc(b.nbytes, 0)
    _check(ret, "acl.rt.malloc(b)")
    out_dev, ret = acl.rt.malloc(out.nbytes, 0)
    _check(ret, "acl.rt.malloc(out)")

    _check(acl.rt.memcpy(a_dev, a.nbytes, int(a.ctypes.data), a.nbytes, _acl_h2d()), "acl.rt.memcpy(a H2D)")
    _check(acl.rt.memcpy(b_dev, b.nbytes, int(b.ctypes.data), b.nbytes, _acl_h2d()), "acl.rt.memcpy(b H2D)")

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None
    launch(ctypes.c_void_p(stream), block_dim, ctypes.c_void_p(a_dev), ctypes.c_void_p(b_dev), ctypes.c_void_p(out_dev))

    _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")
    _check(acl.rt.memcpy(int(out.ctypes.data), out.nbytes, out_dev, out.nbytes, _acl_d2h()), "acl.rt.memcpy(out D2H)")

    np.testing.assert_allclose(out, expected, rtol=1e-2, atol=2e-2)

    acl.rt.free(a_dev)
    acl.rt.free(b_dev)
    acl.rt.free(out_dev)
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(device_id)
    acl.finalize()


def run_add16_cpu_from_so(*, so_path: Path) -> None:
    import ctypes

    x = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    y = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    expected = (x + y).astype(np.float16)
    out = np.empty_like(x)

    lib = ctypes.CDLL(str(so_path))
    fn = lib.pto_kernel_cpu
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = None

    fn(ctypes.c_void_p(int(x.ctypes.data)), ctypes.c_void_p(int(y.ctypes.data)), ctypes.c_void_p(int(out.ctypes.data)))
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def run_gemm16_cpu_from_so(*, so_path: Path) -> None:
    import ctypes

    a = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    b = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    # Avoid relying on NumPy's matmul/dot (can be broken depending on the local build/toolchain).
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    expected = (a32[:, :, None] * b32[None, :, :]).sum(axis=1).astype(np.float32)
    out = np.empty((16, 16), dtype=np.float32)

    lib = ctypes.CDLL(str(so_path))
    fn = lib.pto_kernel_cpu
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = None

    fn(ctypes.c_void_p(int(a.ctypes.data)), ctypes.c_void_p(int(b.ctypes.data)), ctypes.c_void_p(int(out.ctypes.data)))
    np.testing.assert_allclose(out, expected, rtol=1e-2, atol=2e-2)


def default_ascend_home() -> Path:
    env = os.environ.get("ASCEND_HOME_PATH", "")
    if env:
        p = Path(env)
        if p.exists():
            return p

    home = Path.home()
    candidates = [
        home / "Ascend/ascend-toolkit/latest",
        home / "Ascend/ascend-toolkit",
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
        Path("/usr/local/Ascend/ascend-toolkit"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path()
