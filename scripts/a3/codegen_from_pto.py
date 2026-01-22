#!/usr/bin/env python3
import argparse
import os
import re
import sys
import types


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _sanitize_identifier(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not cleaned:
        return "parsed_module"
    if cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned


def _load_module_from_pto(pto_path: str):
    root = _repo_root()
    sys.path.insert(0, root)

    from pto_parser import PTOParser, PythonCodeGenerator

    parser = PTOParser()
    module = parser.parse_file(pto_path)

    if module.name == "parsed_module":
        base = os.path.splitext(os.path.basename(pto_path))[0]
        module.name = _sanitize_identifier(base)
    else:
        module.name = _sanitize_identifier(module.name)

    code = PythonCodeGenerator(module).generate()
    mod = types.ModuleType("pto_from_pto")
    mod.__file__ = pto_path

    # Ensure import pto_compile resolves to this repo root.
    if "pto_compile" not in sys.modules:
        sys.path.insert(0, root)

    exec(compile(code, pto_path, "exec"), mod.__dict__)
    return module, mod


def _emit_outputs(obj, backend: str, output_prefix: str, output_base_dir: str, module_func: str, enable_fusion: bool):
    from pto_compile import MultiBackendCodeGenerator, BACKENDS, PTOCompiler, PTOModule

    if backend not in BACKENDS and backend != "all":
        raise ValueError(f"Unsupported backend: {backend}")

    gen = MultiBackendCodeGenerator(enable_fusion=enable_fusion)

    def _emit_one(prog, backend_key: str):
        backend_info = BACKENDS[backend_key]
        out_dir = os.path.join(output_base_dir, f"output{backend_info['suffix']}", output_prefix)
        os.makedirs(out_dir, exist_ok=True)
        if backend_key == "arm64":
            code = gen.generate_arm64(prog)
        elif backend_key == "cuda":
            code = gen.generate_cuda(prog)
        elif backend_key == "ascend_a2a3":
            code = gen.generate_ascend_a2a3(prog)
        elif backend_key == "ascend_a5":
            code = gen.generate_ascend_a5(prog)
        else:
            raise ValueError(f"Unsupported backend: {backend_key}")
        out_file = os.path.join(out_dir, f"{prog.name}{backend_info['extension']}")
        with open(out_file, "w") as f:
            f.write(code)
        print(f"  [{backend_info['name']}] -> {out_file}")
        return out_file

    def _emit_pto(prog):
        compiler = PTOCompiler()
        pto_asm = compiler.compile(prog)
        pto_dir = os.path.join(output_base_dir, "output_pto", output_prefix)
        os.makedirs(pto_dir, exist_ok=True)
        pto_file = os.path.join(pto_dir, f"{prog.name}.pto")
        with open(pto_file, "w") as f:
            f.write(pto_asm)
        print(f"  [PTO-AS] -> {pto_file}")
        return pto_file

    if isinstance(obj, PTOModule):
        fn_names = obj.get_function_names() if module_func == "__all__" else [module_func]
        for name in fn_names:
            prog = obj.get_function(name)
            if prog is None:
                raise ValueError(f"Module function not found: {name}")
            if backend == "all":
                for bk in ("arm64", "cuda", "ascend_a2a3", "ascend_a5"):
                    _emit_one(prog, bk)
                _emit_pto(prog)
            else:
                _emit_one(prog, backend)
                _emit_pto(prog)
    else:
        prog = obj
        if backend == "all":
            for bk in ("arm64", "cuda", "ascend_a2a3", "ascend_a5"):
                _emit_one(prog, bk)
            _emit_pto(prog)
        else:
            _emit_one(prog, backend)
            _emit_pto(prog)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate backend code directly from a .pto file"
    )
    parser.add_argument("--pto", required=True, help="Path to .pto file")
    parser.add_argument(
        "--backend",
        default="ascend_a2a3",
        choices=["arm64", "cuda", "ascend_a2a3", "ascend_a5", "all"],
        help="Backend to generate (default: ascend_a2a3)",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Category name for output subdirectory (default: pto basename)",
    )
    parser.add_argument(
        "--output-base-dir",
        default="examples",
        help="Base output directory (default: examples)",
    )
    parser.add_argument(
        "--module-func",
        default="__all__",
        help='When .pto has multiple functions: "__all__" (default) or a single function name',
    )
    parser.add_argument(
        "--enable-fusion",
        action="store_true",
        help="Enable loop fusion where applicable",
    )

    args = parser.parse_args()

    pto_path = os.path.abspath(args.pto)
    if not os.path.isfile(pto_path):
        print(f"Error: .pto file not found: {pto_path}")
        return 1

    module, mod = _load_module_from_pto(pto_path)
    builder_name = f"create_{module.name}_module"
    builder = getattr(mod, builder_name, None)
    if builder is None or not callable(builder):
        print(f"Error: builder function not found: {builder_name}")
        return 1

    obj = builder()

    output_prefix = args.output_prefix
    if not output_prefix:
        output_prefix = os.path.splitext(os.path.basename(pto_path))[0]

    print(f"==> PTO: {pto_path}")
    print(f"==> Module: {module.name}")
    print(f"==> Output prefix: {output_prefix}")
    _emit_outputs(
        obj,
        backend=args.backend,
        output_prefix=output_prefix,
        output_base_dir=args.output_base_dir,
        module_func=args.module_func,
        enable_fusion=args.enable_fusion,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
