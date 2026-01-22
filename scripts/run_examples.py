#!/usr/bin/env python3
import argparse
import fnmatch
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE_SCRIPTS = [
    "pto_isa_sinh.py",
    "pto_fused_softmax.py",
    "pto_aten_ir_primitives.py",
    "pto_torch_tensor.py",
    "pto_torch_functional.py",
    "pto_torch_nn_operators.py",
    "pto_torch_flexattention.py",
    "pto_llama7B_dynamic.py",
]


def run(cmd, env=None):
    print(f"+ {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=env)


def list_subdirs(base_dir, skip_llama):
    if not os.path.isdir(base_dir):
        return []
    subdirs = []
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        if skip_llama and name == "llama7b":
            continue
        subdirs.append(name)
    return subdirs


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_generate(skip_llama):
    scripts = EXAMPLE_SCRIPTS
    if skip_llama:
        scripts = [s for s in scripts if s != "pto_llama7B_dynamic.py"]
    for script in scripts:
        run(f"python {os.path.join(ROOT, 'examples', script)}")


def run_cpu(subdirs, c_glob):
    build_dir = os.path.join(ROOT, "build", "cpu")
    kernels_dir = os.path.join(build_dir, "kernels")
    logs_dir = os.path.join(build_dir, "logs")
    ensure_dir(kernels_dir)
    ensure_dir(logs_dir)

    runner = os.path.join(build_dir, "pto_cpu_runner")
    run(f"g++ -O2 -std=c++17 {os.path.join(ROOT, 'scripts/cpu/pto_cpu_runner.cpp')} -ldl -o {runner}")

    for subdir in subdirs:
        base = os.path.join(ROOT, "examples", "output_arm64", subdir)
        if not os.path.isdir(base):
            print(f"[skip] missing CPU outputs: {base}")
            continue
        so_paths = []
        for root, _, files in os.walk(base):
            for name in sorted(files):
                if not fnmatch.fnmatch(name, c_glob):
                    continue
                if not name.endswith(".c"):
                    continue
                c_path = os.path.join(root, name)
                rel = os.path.relpath(c_path, os.path.join(ROOT, "examples", "output_arm64"))
                out_dir = os.path.join(kernels_dir, os.path.dirname(rel))
                ensure_dir(out_dir)
                so_path = os.path.join(out_dir, os.path.splitext(os.path.basename(name))[0] + ".so")
                run(
                    " ".join(
                        [
                            "gcc -shared -fPIC -O2 -std=c11 -DPTO_CPU_SMOKE_RUNNER",
                            f"-I{ROOT}",
                            f"-I{os.path.join(ROOT, 'include')}",
                            c_path,
                            "-o",
                            so_path,
                        ]
                    )
                )
                so_paths.append(so_path)
        if not so_paths:
            print(f"[skip] no CPU kernels found in {base}")
            continue
        log_path = os.path.join(logs_dir, f"{subdir}_cpu.log")
        run(f"{runner} " + " ".join(so_paths) + f" |& tee {log_path}")


def run_npu(mode, subdirs, cpp_glob, soc_version):
    if not os.environ.get("ASCEND_HOME_PATH"):
        raise RuntimeError("ASCEND_HOME_PATH must be set for sim/npu runs.")

    for subdir in subdirs:
        env = os.environ.copy()
        env["PTO_RUN_MODE"] = mode
        env["PTO_SKIP_GENERATE"] = "1"
        env["PTO_SUBDIRS"] = subdir
        env["PTO_CPP_GLOB"] = cpp_glob
        if soc_version:
            env["PTO_SOC_VERSION"] = soc_version
        run(f"{os.path.join(ROOT, 'scripts/a3/build_and_run_examples.sh')}", env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Run PTO examples on CPU, simulator, or NPU one subdir at a time."
    )
    parser.add_argument(
        "--mode",
        default="npu",
        choices=["cpu", "sim", "npu", "all"],
        help="Run mode (default: npu)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip Python example generation",
    )
    parser.add_argument(
        "--skip-llama",
        action="store_true",
        help="Skip llama7b outputs",
    )
    parser.add_argument(
        "--subdirs",
        default="",
        help="Comma-separated output subdirs to run (default: all found)",
    )
    parser.add_argument(
        "--cpp-glob",
        default="*.cpp",
        help="C++ filename glob for NPU/sim (default: *.cpp)",
    )
    parser.add_argument(
        "--c-glob",
        default="*.c",
        help="C filename glob for CPU (default: *.c)",
    )
    parser.add_argument(
        "--soc-version",
        default="",
        help="Override SOC version (default: Ascend910B1 for sim, Ascend910B for npu)",
    )
    args = parser.parse_args()

    if not args.skip_generate:
        run_generate(skip_llama=args.skip_llama)

    modes = [args.mode]
    if args.mode == "all":
        modes = ["cpu", "sim", "npu"]

    cpu_subdirs = list_subdirs(os.path.join(ROOT, "examples", "output_arm64"), args.skip_llama)
    npu_subdirs = list_subdirs(os.path.join(ROOT, "examples", "output_ascend_a2a3"), args.skip_llama)

    if args.subdirs:
        requested = [s.strip() for s in args.subdirs.split(",") if s.strip()]
        cpu_subdirs = [s for s in cpu_subdirs if s in requested]
        npu_subdirs = [s for s in npu_subdirs if s in requested]

    for mode in modes:
        if mode == "cpu":
            run_cpu(cpu_subdirs, args.c_glob)
        else:
            soc = args.soc_version
            if not soc:
                soc = "Ascend910B1" if mode == "sim" else "Ascend910B"
            run_npu(mode, npu_subdirs, args.cpp_glob, soc)


if __name__ == "__main__":
    main()
