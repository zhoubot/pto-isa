from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    subprocess.run(cmd, cwd=str(cwd), check=True)


@dataclass(frozen=True)
class CompileConfig:
    ptoas: Path
    ascend_home: Path
    arch: str
    memory_model: str = "MEMORY_BASE"
    insert_events: bool = True


def compile_pto_to_cce_and_bin(*, pto_path: Path, outdir: Path, cfg: CompileConfig) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    # This is still CCE source compiled via `bisheng -xcce`; we use `.cpp` for editor compatibility.
    cce_path = outdir / (pto_path.stem + ".cpp")
    bin_path = outdir / (pto_path.stem + ".bin")

    args = [
        str(cfg.ptoas),
        str(pto_path),
        "-o",
        str(cce_path),
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
    out_so.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "clang++",
            "-shared",
            "-fPIC",
            "-O2",
            "-std=c++23",
            "-Wno-unknown-attributes",
            "-Wno-ignored-attributes",
            f"-I{repo_root() / 'include'}",
            str(cpp_path),
            "-o",
            str(out_so),
        ],
        cwd=repo_root(),
    )


def build_fatobj_so_from_cce(*, cce_path: Path, out_so: Path, arch: str, ascend_home: Path) -> None:
    include_dirs = ascend_include_dirs(ascend_home) + [str(repo_root() / "include")]

    kernel_src = cce_path.read_text(encoding="utf-8")

    m = re.search(r"\bpto_kernel\s*\(([^)]*)\)", kernel_src)
    if not m:
        raise RuntimeError(f"failed to infer pto_kernel(...) signature from: {cce_path}")
    params = [p.strip() for p in m.group(1).split(",") if p.strip()]
    arg_count = len(params)

    host_params = ", ".join([f"void *arg{i}" for i in range(arg_count)])
    kernel_args = ", ".join([f"(GM_ADDR)arg{i}" for i in range(arg_count)])

    combined = (
        "#include \"kernel.cpp\"\n"
        "#include <cstdint>\n\n"
        f"extern \"C\" void ptoas_launch(void *stream, uint32_t blockDim{', ' if arg_count else ''}{host_params})\n"
        "{\n"
        f"    pto_kernel<<<blockDim, nullptr, stream>>>({kernel_args});\n"
        "}\n"
    )

    with tempfile.TemporaryDirectory(prefix="ptoas_so_") as td:
        td_path = Path(td)
        (td_path / "kernel.cpp").write_text(kernel_src, encoding="utf-8")
        combined_path = td_path / "combined.cpp"
        combined_path.write_text(combined, encoding="utf-8")
        combined_o = td_path / "combined.o"

        common = ["bisheng", "-xcce", f"--cce-aicore-arch={arch}", "-std=c++17", "-fPIC"]
        incs = [f"-I{d}" for d in include_dirs]
        _run(common + incs + ["-c", str(combined_path), "-o", str(combined_o)], cwd=td_path)

        out_so.parent.mkdir(parents=True, exist_ok=True)
        link = ["bisheng", "-shared", "--cce-fatobj-link", "-o", str(out_so), str(combined_o)]
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


def run_add16_from_so(*, so_path: Path, device_id: int, block_dim: int) -> None:
    import ctypes
    import acl

    x = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    y = (np.random.rand(16, 16).astype(np.float16) - 0.5)
    expected = (x + y).astype(np.float16)
    out = np.empty_like(x)

    acl.init()
    acl.rt.set_device(device_id)
    stream, ret = acl.rt.create_stream()
    assert ret == 0, ret

    x_dev, ret = acl.rt.malloc(x.nbytes, 0)
    assert ret == 0, ret
    y_dev, ret = acl.rt.malloc(y.nbytes, 0)
    assert ret == 0, ret
    out_dev, ret = acl.rt.malloc(out.nbytes, 0)
    assert ret == 0, ret

    assert acl.rt.memcpy(x_dev, x.nbytes, int(x.ctypes.data), x.nbytes, _acl_h2d()) == 0
    assert acl.rt.memcpy(y_dev, y.nbytes, int(y.ctypes.data), y.nbytes, _acl_h2d()) == 0

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None
    launch(ctypes.c_void_p(stream), block_dim, ctypes.c_void_p(x_dev), ctypes.c_void_p(y_dev), ctypes.c_void_p(out_dev))

    assert acl.rt.synchronize_stream(stream) == 0
    assert acl.rt.memcpy(int(out.ctypes.data), out.nbytes, out_dev, out.nbytes, _acl_d2h()) == 0

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
    assert ret == 0, ret

    a_dev, ret = acl.rt.malloc(a.nbytes, 0)
    assert ret == 0, ret
    b_dev, ret = acl.rt.malloc(b.nbytes, 0)
    assert ret == 0, ret
    out_dev, ret = acl.rt.malloc(out.nbytes, 0)
    assert ret == 0, ret

    assert acl.rt.memcpy(a_dev, a.nbytes, int(a.ctypes.data), a.nbytes, _acl_h2d()) == 0
    assert acl.rt.memcpy(b_dev, b.nbytes, int(b.ctypes.data), b.nbytes, _acl_h2d()) == 0

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None
    launch(ctypes.c_void_p(stream), block_dim, ctypes.c_void_p(a_dev), ctypes.c_void_p(b_dev), ctypes.c_void_p(out_dev))

    assert acl.rt.synchronize_stream(stream) == 0
    assert acl.rt.memcpy(int(out.ctypes.data), out.nbytes, out_dev, out.nbytes, _acl_d2h()) == 0

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
    return Path(env) if env else Path()
