#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pto_compile import PTOCompiler, PTOFunctionBuilder, MultiBackendCodeGenerator
from pto_isa_definition import ElementType, MemorySpace


ROWS = int(os.environ.get("PTO_SMOKE_ROWS", "128"))
COLS = int(os.environ.get("PTO_SMOKE_COLS", "128"))
DTYPE = ElementType.F32


def _emit_all(funcs, output_prefix="smoke_ops"):
    output_base = os.path.join(ROOT, "examples")
    arm64_dir = os.path.join(output_base, "output_arm64", output_prefix)
    ascend_dir = os.path.join(output_base, "output_ascend_a2a3", output_prefix)
    pto_dir = os.path.join(output_base, "output_pto", output_prefix)
    os.makedirs(arm64_dir, exist_ok=True)
    os.makedirs(ascend_dir, exist_ok=True)
    os.makedirs(pto_dir, exist_ok=True)

    gen = MultiBackendCodeGenerator(enable_fusion=True, analyze_buffers=True)
    compiler = PTOCompiler(optimize=True)

    for func in funcs:
        arm64_code = gen.generate_arm64(func)
        with open(os.path.join(arm64_dir, f"{func.name}.c"), "w", encoding="utf-8") as f:
            f.write(arm64_code)

        ascend_code = gen.generate_ascend_a2a3(func)
        with open(os.path.join(ascend_dir, f"{func.name}.cpp"), "w", encoding="utf-8") as f:
            f.write(ascend_code)

        pto_code = compiler.compile(func)
        with open(os.path.join(pto_dir, f"{func.name}.pto"), "w", encoding="utf-8") as f:
            f.write(pto_code)


def _make_copy():
    return (PTOFunctionBuilder("op_copy")
        .in_core()
        .tile("x", ROWS, COLS, DTYPE)
        .memref("input", MemorySpace.GM, DTYPE)
        .memref("output", MemorySpace.GM, DTYPE)
        .load("x", "input", 0, 0)
        .store("x", "output", 0, 0)
        .build())


def _make_unary(name, op):
    return (PTOFunctionBuilder(name)
        .in_core()
        .tile("x", ROWS, COLS, DTYPE)
        .tile("y", ROWS, COLS, DTYPE)
        .memref("input", MemorySpace.GM, DTYPE)
        .memref("output", MemorySpace.GM, DTYPE)
        .load("x", "input", 0, 0)
        .__getattribute__(op)("y", "x")
        .store("y", "output", 0, 0)
        .build())


def _make_binary(name, op):
    return (PTOFunctionBuilder(name)
        .in_core()
        .tile("a", ROWS, COLS, DTYPE)
        .tile("b", ROWS, COLS, DTYPE)
        .tile("y", ROWS, COLS, DTYPE)
        .memref("input0", MemorySpace.GM, DTYPE)
        .memref("input1", MemorySpace.GM, DTYPE)
        .memref("output", MemorySpace.GM, DTYPE)
        .load("a", "input0", 0, 0)
        .load("b", "input1", 0, 0)
        .__getattribute__(op)("y", "a", "b")
        .store("y", "output", 0, 0)
        .build())


def _make_scalar(name, op, scalar=1.25):
    return (PTOFunctionBuilder(name)
        .in_core()
        .tile("x", ROWS, COLS, DTYPE)
        .tile("y", ROWS, COLS, DTYPE)
        .memref("input", MemorySpace.GM, DTYPE)
        .memref("output", MemorySpace.GM, DTYPE)
        .load("x", "input", 0, 0)
        .__getattribute__(op)("y", "x", scalar)
        .store("y", "output", 0, 0)
        .build())


def _make_rowsum(name, op):
    return (PTOFunctionBuilder(name)
        .in_core()
        .tile("x", ROWS, COLS, DTYPE)
        .tile("y", ROWS, 1, DTYPE)
        .memref("input", MemorySpace.GM, DTYPE)
        .memref("output", MemorySpace.GM, DTYPE)
        .load("x", "input", 0, 0)
        .__getattribute__(op)("y", "x")
        .store("y", "output", 0, 0)
        .build())


def _make_colsum(name, op):
    return (PTOFunctionBuilder(name)
        .in_core()
        .tile("x", ROWS, COLS, DTYPE)
        .tile("y", 1, COLS, DTYPE)
        .memref("input", MemorySpace.GM, DTYPE)
        .memref("output", MemorySpace.GM, DTYPE)
        .load("x", "input", 0, 0)
        .__getattribute__(op)("y", "x")
        .store("y", "output", 0, 0)
        .build())


def _make_rowexpand(name, op):
    return (PTOFunctionBuilder(name)
        .in_core()
        .tile("x", ROWS, COLS, DTYPE)
        .tile("row", ROWS, 1, DTYPE)
        .tile("y", ROWS, COLS, DTYPE)
        .memref("input", MemorySpace.GM, DTYPE)
        .memref("row_vals", MemorySpace.GM, DTYPE)
        .memref("output", MemorySpace.GM, DTYPE)
        .load("x", "input", 0, 0)
        .load("row", "row_vals", 0, 0)
        .__getattribute__(op)("y", "x", "row")
        .store("y", "output", 0, 0)
        .build())


def _make_expands():
    return (PTOFunctionBuilder("op_expands")
        .in_core()
        .tile("y", ROWS, COLS, DTYPE)
        .memref("output", MemorySpace.GM, DTYPE)
        .expands("y", 1.0)
        .store("y", "output", 0, 0)
        .build())


def main():
    funcs = [
        _make_copy(),
        _make_binary("op_add", "add"),
        _make_binary("op_sub", "sub"),
        _make_binary("op_mul", "mul"),
        _make_binary("op_div", "div"),
        _make_binary("op_max", "max"),
        _make_binary("op_min", "min"),
        _make_unary("op_abs", "abs"),
        _make_unary("op_neg", "neg"),
        _make_unary("op_relu", "relu"),
        _make_unary("op_sqrt", "sqrt"),
        _make_unary("op_rsqrt", "rsqrt"),
        _make_unary("op_recip", "recip"),
        _make_scalar("op_adds", "adds", 1.25),
        _make_scalar("op_subs", "subs", 1.25),
        _make_scalar("op_muls", "muls", 1.25),
        _make_scalar("op_divs", "divs", 1.25),
        _make_scalar("op_maxs", "maxs", 1.25),
        _make_scalar("op_mins", "mins", 1.25),
        _make_rowsum("op_rowsum", "rowsum"),
        _make_rowsum("op_rowmax", "rowmax"),
        _make_rowsum("op_rowmin", "rowmin"),
        _make_colsum("op_colsum", "colsum"),
        _make_rowexpand("op_rowexpandsub", "rowexpandsub"),
        _make_rowexpand("op_rowexpanddiv", "rowexpanddiv"),
        _make_rowexpand("op_rowexpandmul", "rowexpandmul"),
        _make_expands(),
    ]

    _emit_all(funcs)
    print("Generated smoke ops:")
    for f in funcs:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
