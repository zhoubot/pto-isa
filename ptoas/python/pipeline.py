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


def configure_ascend_sim_env(*, ascend_home: Path, soc: str) -> None:
    """
    Configure process env vars for Ascend simulator runs.

    This mirrors the behavior in `tests/script/run_st.py`:
      - remove any existing `/runtime/lib64` entries from LD_LIBRARY_PATH
      - add `$ASCEND_HOME_PATH/runtime/lib64/stub`
      - source `$ASCEND_HOME_PATH/bin/setenv.bash` (if present)
      - add `$ASCEND_HOME_PATH/tools/simulator/<soc>/lib`
    """
    if not ascend_home or not ascend_home.exists():
        raise ValueError("ascend_home must exist")

    # Many scripts key off ASCEND_HOME_PATH; ensure it is set consistently.
    os.environ.setdefault("ASCEND_HOME_PATH", str(ascend_home))

    # Start from a "sim-friendly" LD_LIBRARY_PATH (avoid mixing stub/real runtime).
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ld:
        filtered = [p for p in ld.split(":") if "/runtime/lib64" not in p]
        os.environ["LD_LIBRARY_PATH"] = ":".join(filtered)

    # setenv.bash may set ASCEND_HOME_PATH and a bunch of runtime/compiler vars (including LD_LIBRARY_PATH).
    setenv_path = ascend_home / "bin/setenv.bash"
    try:
        sourced = _source_setenv_bash(setenv_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to source {setenv_path}: {e.stderr.decode('utf-8', errors='ignore')}") from e
    for k, v in sourced.items():
        os.environ[k] = v

    # Ensure stub + simulator libs are present after sourcing.
    stub = ascend_home / "runtime/lib64/stub"
    if stub.exists():
        os.environ["LD_LIBRARY_PATH"] = f"{stub}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    sim_lib = ascend_home / "tools/simulator" / soc / "lib"
    if sim_lib.exists():
        os.environ["LD_LIBRARY_PATH"] = f"{sim_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    else:
        raise RuntimeError(f"simulator lib dir not found: {sim_lib} (check --soc and your toolkit install)")


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
    if cfg.insert_events:
        args.append("--insert-events")
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
    if insert_events:
        args.append("--insert-events")
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
    *, cce_path: Path, out_so: Path, arch: str, ascend_home: Path, fixed_block_dim: int | None = None
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
        common = [bisheng, "-xcce", f"--cce-aicore-arch={arch}", "-std=c++17", "-fPIC"]
        incs = [f"-I{d}" for d in include_dirs]
        _run(common + incs + ["-c", str(combined_path), "-o", str(combined_o)], cwd=td_path)

        out_so.parent.mkdir(parents=True, exist_ok=True)
        link = [bisheng, "-shared", "--cce-fatobj-link", "-o", str(out_so), str(combined_o)]
        lib64 = ascend_home / "lib64"
        if lib64.exists():
            link += [f"-L{lib64}", f"-Wl,-rpath,{lib64}"]
        link += [
            "-lruntime",
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

    stream = None
    dev_ptrs: list[int] = []
    try:
        acl.init()
        acl.rt.set_device(device_id)
        stream, ret = acl.rt.create_stream()
        _check(ret, "acl.rt.create_stream")

        for i, a in enumerate(host_arrays):
            p, r = acl.rt.malloc(int(a.nbytes), 0)
            _check(r, f"acl.rt.malloc(arg{i})")
            dev_ptrs.append(int(p))

        for i, (a, dev) in enumerate(zip(host_arrays, dev_ptrs)):
            if host_spec.args[i].role == "out":
                continue
            _check(
                acl.rt.memcpy(dev, int(a.nbytes), int(a.ctypes.data), int(a.nbytes), _acl_h2d()),
                f"acl.rt.memcpy(arg{i} H2D)",
            )

        lib = ctypes.CDLL(str(so_path))
        launch = lib.ptoas_launch
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * len(dev_ptrs)
        launch.restype = None
        launch(ctypes.c_void_p(stream), int(block_dim), *[ctypes.c_void_p(p) for p in dev_ptrs])

        _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")

        out: list[np.ndarray] = []
        for i in host_spec.output_indices():
            a = host_arrays[i]
            tmp = np.empty_like(a)
            _check(
                acl.rt.memcpy(int(tmp.ctypes.data), int(tmp.nbytes), dev_ptrs[i], int(tmp.nbytes), _acl_d2h()),
                f"acl.rt.memcpy(arg{i} D2H)",
            )
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
