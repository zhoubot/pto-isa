from __future__ import annotations

"""
Tiny Python-to-MLIR frontend for the LLVM PTO dialect.

This frontend parses a restricted Python AST and emits MLIR that can be fed to
the packaged `bin/ptoas` (from `~/llvm-project/build-mlir/bin/ptoas`).

Supported authoring styles:

1) Function-call DSL (legacy):

   ```
   def add16():
       prologue()
       x = tensor(dtype="f16", shape=(16, 16))
       y = tensor(dtype="f16", shape=(16, 16))
       z = tensor(dtype="f16", shape=(16, 16))
       tx = tile(loc="Vec", dtype="f16", rows=16, cols=16)
       ty = tile(loc="Vec", dtype="f16", rows=16, cols=16)
       tz = tile(loc="Vec", dtype="f16", rows=16, cols=16)
       tload(tx, x, 0, 0)
       tload(ty, y, 0, 0)
       tadd(tz, tx, ty)
       tstore(z, 0, 0, tz)
       epilogue()
   ```

2) Object DSL (recommended; used by `kernels/python/*.py` via `pto_as.PTO`):

   ```
   def add16():
       pto = PTO("add16")
       pto.prologue()
       x = pto.tensor(dtype="f16", shape=(16, 16), role="in")
       ...
       tz = pto.add(tx, ty)
       pto.store(z, tz)
       pto.epilogue()
       return pto.program()
   ```
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .dsl import _PTO_ISA_OPS as _PTO_ISA_OPS_DSL
from .pto_asm import TensorType, TileType, _mlir_scalar_type


_STRIPPED_T_OPS: set[str] = {
    op[1:]
    for op in _PTO_ISA_OPS_DSL
    if op.startswith("t") and len(op) > 1 and op[1].isalpha()
}


class FrontendError(Exception):
    pass


@dataclass(frozen=True)
class TensorArg:
    name: str
    arg_index: int
    ty: TensorType
    role: str | None = None

    def host_spec(self) -> "TensorSpec":
        from .host_codegen import TensorSpec

        return TensorSpec(dtype=self.ty.dtype, shape=self.ty.shape2())


@dataclass(frozen=True)
class KernelSpec:
    name: str
    pto: str  # MLIR module text
    tensor_args: tuple[TensorArg, ...]

    def host_tensor_specs(self) -> list["TensorSpec"]:
        return [arg.host_spec() for arg in self.tensor_args]


@dataclass(frozen=True)
class _Value:
    ssa: str
    ty: str


class _Text:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0

    def line(self, s: str) -> None:
        self._lines.append(("  " * self._indent) + s)

    def open(self, header: str) -> None:
        self.line(f"{header} {{")
        self._indent += 1

    def else_open(self) -> None:
        if self._indent <= 0:
            raise FrontendError("else without open block")
        self._indent -= 1
        self.line("} else {")
        self._indent += 1

    def close(self) -> None:
        if self._indent <= 0:
            raise FrontendError("unbalanced close()")
        self._indent -= 1
        self.line("}")

    def emit(self) -> str:
        return "\n".join(self._lines).rstrip() + "\n"


class _Compiler:
    def __init__(self, *, consts: dict[str, Any] | None = None) -> None:
        # Function-scope prelude emitted before the main body.
        # Used for lazily-created scratch tiles that must be visible across loop nests.
        self._prelude = _Text()
        self._t = _Text()
        self._tmp_i = 0
        self._ssa_used: set[str] = set()

        self._env: dict[str, _Value] = {}
        self._tensor_types: dict[str, TensorType] = {}
        self._tile_types: dict[str, TileType] = {}
        self._scratch_tiles: dict[TileType, _Value] = {}

        self._next_tensor_arg = 0
        self._tensor_args: dict[int, TensorType] = {}
        self._tensor_arg_names: dict[int, str] = {}
        self._tensor_arg_roles: dict[int, str | None] = {}
        self._explicit_kernel_name: str | None = None

        # Compile-time env (for shapes/strides/constant folding in Python).
        self._const_env: dict[str, Any] = {}
        self._injected_consts: dict[str, Any] = dict(consts or {})
        if consts:
            self._seed_consts(consts)

        # Typed constant caching: (literal, ty) -> SSA
        self._const_scalars: dict[tuple[str, str], _Value] = {}

    def _scratch_tile(self, tt: TileType, *, hint: str = "scratch") -> _Value:
        v = self._scratch_tiles.get(tt)
        if v is not None:
            return v
        ssa = self._alloc_ssa(hint)
        # Keep scratch tiles at function scope so they can be reused across
        # multiple loop nests without multiplying local memory usage.
        self._prelude.line(f"{ssa} = pto.alloc_tile : {tt}")
        v = _Value(ssa=ssa, ty=str(tt))
        self._scratch_tiles[tt] = v
        return v

    # --- compile-time evaluation ------------------------------------------------

    def _seed_consts(self, consts: dict[str, Any]) -> None:
        for name, value in consts.items():
            if not isinstance(name, str) or not name:
                raise FrontendError("consts keys must be non-empty strings")
            self._const_env[name] = value

    def _eval_const(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception as e:
            raise FrontendError(f"expected a literal, got: {ast.dump(node)}") from e

    def _eval_static(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self._const_env:
                return self._const_env[node.id]
            raise FrontendError(f"unknown compile-time name: {node.id}")
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._eval_static(elt) for elt in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = self._eval_static(node.operand)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            lhs = self._eval_static(node.left)
            rhs = self._eval_static(node.right)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                return lhs / rhs
            if isinstance(node.op, ast.FloorDiv):
                return lhs // rhs
            raise FrontendError(f"unsupported binop in const eval: {ast.dump(node.op)}")
        if isinstance(node, ast.Call):
            fn_name: str | None = None
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                fn_name = f"{node.func.value.id}.{node.func.attr}"
            if fn_name is None:
                raise FrontendError(f"unsupported call in const eval: {ast.dump(node)}")

            args = [self._eval_static(a) for a in node.args]
            if fn_name in ("sqrt", "math.sqrt"):
                if len(args) != 1:
                    raise FrontendError("sqrt(...) expects 1 arg")
                import math

                return math.sqrt(args[0])
            if fn_name == "scalar":
                if len(args) != 1 or not isinstance(args[0], str):
                    raise FrontendError('scalar(...) expects one string arg like scalar("f32")')
                return args[0]
            raise FrontendError(f"unsupported compile-time call: {fn_name}")

        raise FrontendError(f"unsupported compile-time expr: {ast.dump(node)}")

    # --- SSA helpers -----------------------------------------------------------

    def _alloc_ssa(self, base: str) -> str:
        base = str(base)
        if not base:
            base = "v"
        name = f"%{base}"
        if name not in self._ssa_used:
            self._ssa_used.add(name)
            return name
        i = 1
        while f"%{base}_{i}" in self._ssa_used:
            i += 1
        name = f"%{base}_{i}"
        self._ssa_used.add(name)
        return name

    def _fresh_tmp(self, *, ty: str) -> _Value:
        self._tmp_i += 1
        return _Value(ssa=self._alloc_ssa(f"t{self._tmp_i}"), ty=ty)

    def _bind(self, py_name: str, v: _Value) -> None:
        self._env[py_name] = v

    def _get(self, py_name: str) -> _Value:
        v = self._env.get(py_name)
        if v is None:
            raise FrontendError(f"unknown name: {py_name}")
        return v

    def _name_of(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        raise FrontendError(f"expected a name, got: {ast.dump(node)}")

    def _const(self, *, value: int | float | bool, ty: str) -> _Value:
        if isinstance(value, bool):
            lit = "1" if value else "0"
        elif isinstance(value, float):
            lit = repr(float(value))
        else:
            lit = str(int(value))
        key = (lit, ty)
        existing = self._const_scalars.get(key)
        if existing is not None:
            return existing

        # Stable-ish SSA name for readability.
        if ty == "index" and isinstance(value, (int, bool)):
            base = f"c{int(value)}" if int(value) >= 0 else f"c_neg{abs(int(value))}"
        else:
            base = "cst"
        ssa = self._alloc_ssa(base)
        self._t.line(f"{ssa} = arith.constant {lit} : {ty}")
        v = _Value(ssa=ssa, ty=ty)
        self._const_scalars[key] = v
        return v

    # --- opcode mapping -------------------------------------------------------

    def _call_name(self, call: ast.Call) -> str:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            return call.func.attr
        raise FrontendError(f"unsupported call form: {ast.dump(call)}")

    def _opcode_alias(self, name: str) -> str:
        aliased = {
            "mov": "tmov",
            "load": "tload",
            "store": "tstore",
            "print": "tprint",
            "rowmax": "trowmax",
            "matmul": "tmatmul",
        }.get(name)
        if aliased is not None:
            return aliased
        if name in _STRIPPED_T_OPS:
            return "t" + name
        return name

    def _mlir_mnemonic(self, op: str) -> str:
        """
        Map legacy/DSL mnemonics onto LLVM PTO dialect mnemonics.
        """
        return {
            # LLVM uses dotted names for matmul variants.
            "tmatmul_mx": "tmatmul.mx",
            "tmatmul_acc": "tmatmul.acc",
            "tmatmul_bias": "tmatmul.bias",
            "tmatmul_mx_acc": "tmatmul.mx.acc",
            "tmatmul_mx_bias": "tmatmul.mx.bias",
            # Old frontend exposed extra variants; lower them to the canonical op.
            "tfillpad_inplace": "tfillpad",
        }.get(op, op)

    # --- scalar lowering (index) ---------------------------------------------

    def _emit_index_expr(self, node: ast.AST) -> _Value:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
            return self._const(value=int(node.value), ty="index")
        if isinstance(node, ast.Name):
            if node.id in self._const_env and isinstance(self._const_env[node.id], (int, bool)):
                return self._const(value=int(self._const_env[node.id]), ty="index")
            v = self._env.get(node.id)
            if v is None:
                raise FrontendError(f"unknown runtime name: {node.id}")
            if v.ty == "index":
                return v
            if v.ty == "i64":
                out = self._fresh_tmp(ty="index")
                self._t.line(f"{out.ssa} = arith.index_cast {v.ssa} : i64 to index")
                return out
            raise FrontendError(f"expected index-typed value for {node.id}, got {v.ty}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = self._emit_index_expr(node.operand)
            if isinstance(node.op, ast.UAdd):
                return v
            z = self._const(value=0, ty="index")
            out = self._fresh_tmp(ty="index")
            self._t.line(f"{out.ssa} = arith.subi {z.ssa}, {v.ssa} : index")
            return out
        if isinstance(node, ast.BinOp):
            lhs = self._emit_index_expr(node.left)
            rhs = self._emit_index_expr(node.right)
            out = self._fresh_tmp(ty="index")
            if isinstance(node.op, ast.Add):
                self._t.line(f"{out.ssa} = arith.addi {lhs.ssa}, {rhs.ssa} : index")
                return out
            if isinstance(node.op, ast.Sub):
                self._t.line(f"{out.ssa} = arith.subi {lhs.ssa}, {rhs.ssa} : index")
                return out
            if isinstance(node.op, ast.Mult):
                self._t.line(f"{out.ssa} = arith.muli {lhs.ssa}, {rhs.ssa} : index")
                return out
            if isinstance(node.op, ast.FloorDiv):
                self._t.line(f"{out.ssa} = arith.divsi {lhs.ssa}, {rhs.ssa} : index")
                return out
            if isinstance(node.op, ast.Mod):
                self._t.line(f"{out.ssa} = arith.remsi {lhs.ssa}, {rhs.ssa} : index")
                return out
            raise FrontendError(f"unsupported binop in index expr: {ast.dump(node.op)}")
        raise FrontendError(f"unsupported index expr: {ast.dump(node)}")

    def _emit_scalar_assign(self, dst: str, value: ast.AST) -> None:
        v = self._emit_index_expr(value)
        self._bind(dst, v)

    def _emit_scalar_expr(self, node: ast.AST, *, ty: str) -> _Value:
        """
        Emit a scalar SSA value of MLIR type `ty` (e.g. `i32`, `f32`, `index`).
        """
        if ty == "index":
            return self._emit_index_expr(node)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                if ty.startswith("f"):
                    return self._const(value=float(int(node.value)), ty=ty)
                return self._const(value=int(node.value), ty=ty)
            if isinstance(node.value, int):
                if ty.startswith("f"):
                    return self._const(value=float(node.value), ty=ty)
                return self._const(value=int(node.value), ty=ty)
            if isinstance(node.value, float):
                if not ty.startswith("f"):
                    raise FrontendError(f"expected integer scalar for type {ty}, got float literal")
                return self._const(value=float(node.value), ty=ty)
            raise FrontendError(f"unsupported scalar literal: {node.value!r}")

        if isinstance(node, ast.Name):
            if node.id in self._const_env and isinstance(self._const_env[node.id], (int, bool, float)):
                v = self._const(
                    value=float(self._const_env[node.id])
                    if isinstance(self._const_env[node.id], float) or ty.startswith("f")
                    else int(self._const_env[node.id]),
                    ty=ty,
                )
                return v

            if node.id not in self._env:
                raise FrontendError(f"unknown runtime name: {node.id}")
            v = self._get(node.id)
            if v.ty == ty:
                return v
            # Allow index <-> integer scalar casts (common when authors use loop IVs).
            if v.ty == "index" and ty.startswith("i"):
                out = self._fresh_tmp(ty=ty)
                self._t.line(f"{out.ssa} = arith.index_cast {v.ssa} : index to {ty}")
                return out
            if v.ty.startswith("i") and ty == "index":
                out = self._fresh_tmp(ty="index")
                self._t.line(f"{out.ssa} = arith.index_cast {v.ssa} : {v.ty} to index")
                return out
            raise FrontendError(f"expected {ty}-typed value for {node.id}, got {v.ty}")

        # Best-effort: allow simple constant expressions.
        try:
            v = self._eval_static(node)
        except Exception:
            v = None
        if isinstance(v, (int, bool)):
            if ty.startswith("f"):
                return self._const(value=float(int(v)), ty=ty)
            return self._const(value=int(v), ty=ty)
        if isinstance(v, float):
            if not ty.startswith("f"):
                raise FrontendError(f"expected integer scalar for type {ty}, got float expression")
            return self._const(value=float(v), ty=ty)

        raise FrontendError(f"unsupported scalar expr for type {ty}: {ast.dump(node)}")

    # --- attrs (RoundMode/CmpMode/MaskPattern) -------------------------------

    def _emit_round_mode_attr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "RoundMode":
            mode_map = {
                "CAST_NONE": "NONE",
                "CAST_RINT": "CAST_RINT",
                "CAST_ROUND": "ROUND",
                "CAST_FLOOR": "FLOOR",
                "CAST_CEIL": "CEIL",
                "CAST_TRUNC": "TRUNC",
                "CAST_ODD": "ODD",
            }
            mapped = mode_map.get(node.attr)
            if mapped is None:
                raise FrontendError(f"unsupported RoundMode: {node.attr}")
            return f"#pto.round_mode<{mapped}>"
        raise FrontendError("expected RoundMode.<...> for rounding mode")

    def _emit_cmp_mode_attr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "CmpMode":
            # NOTE: LLVM PTO defines custom enum spellings for assembly:
            #   EQ -> equal, NE -> not_equal, LT -> less_than, ...
            mode_map = {
                "EQ": "equal",
                "NE": "not_equal",
                "LT": "less_than",
                "LE": "less_equal",
                "GT": "greater_than",
                "GE": "greater_equal",
            }
            mapped = mode_map.get(node.attr)
            if mapped is None:
                raise FrontendError(f"unsupported CmpMode: {node.attr}")
            return f"#pto.cmp<{mapped}>"
        raise FrontendError("expected CmpMode.<...> for cmpMode")

    # --- declarations ---------------------------------------------------------

    def _declare_tensor(self, target: str, call: ast.Call) -> None:
        dtype: str | None = None
        shape: Any | None = None
        stride: Any | None = None
        layout: str = "ND"
        role: str | None = None
        declared_name: str | None = None

        args = list(call.args)
        has_dtype_kw = any(kw.arg == "dtype" for kw in call.keywords if kw.arg is not None)
        if len(args) >= 2 and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str) and has_dtype_kw:
            declared_name = args[0].value
            shape = self._eval_static(args[1])
        else:
            if args:
                if len(args) >= 1:
                    dtype = self._eval_const(args[0])
                if len(args) >= 2:
                    shape = self._eval_static(args[1])
                if len(args) >= 3:
                    stride = self._eval_static(args[2])
                if len(args) >= 4:
                    layout = self._eval_const(args[3])

        for kw in call.keywords:
            if kw.arg == "dtype":
                dtype = self._eval_const(kw.value)
            elif kw.arg == "shape":
                shape = self._eval_static(kw.value)
            elif kw.arg == "stride":
                stride = self._eval_static(kw.value)
            elif kw.arg == "layout":
                layout = self._eval_const(kw.value)
            elif kw.arg == "role":
                role = str(self._eval_const(kw.value))
                if role not in ("in", "out", "inout"):
                    raise FrontendError("tensor(..., role=...) must be one of: in, out, inout")
            elif kw.arg in ("arg", "arg_index"):
                pass
            else:
                raise FrontendError(f"unknown tensor(...) kw: {kw.arg}")

        pto_name = declared_name or target
        if dtype is None or shape is None:
            raise FrontendError("tensor(...) requires dtype and shape")

        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise FrontendError("tensor(...) currently expects shape=(H, W)")
        h, w = int(shape[0]), int(shape[1])

        if stride is None:
            s0, s1 = w, 1
        else:
            if not isinstance(stride, (tuple, list)) or len(stride) != 2:
                raise FrontendError("tensor(..., stride=...) expects stride=(S0, S1)")
            s0, s1 = int(stride[0]), int(stride[1])

        ty = TensorType(dtype=dtype, shape=(h, w), stride=(s0, s1), layout=layout)

        arg_index: int | None = None
        for kw in call.keywords:
            if kw.arg in ("arg", "arg_index"):
                arg_index = int(self._eval_const(kw.value))
        if arg_index is None:
            arg_index = self._next_tensor_arg
            self._next_tensor_arg += 1

        self._tensor_args.setdefault(arg_index, ty)
        self._tensor_arg_names.setdefault(arg_index, pto_name)
        self._tensor_arg_roles.setdefault(arg_index, role)

        ssa = self._alloc_ssa(pto_name.lstrip("%")).replace("%%", "%")
        c_h = self._const(value=h, ty="index")
        c_w = self._const(value=w, ty="index")
        c_s0 = self._const(value=s0, ty="index")
        c_s1 = self._const(value=s1, ty="index")
        self._t.line(
            f"{ssa} = pto.make_tensor_view %arg{arg_index}, "
            f"shape = [{c_h.ssa}, {c_w.ssa}] strides = [{c_s0.ssa}, {c_s1.ssa}] : {ty}"
        )
        v = _Value(ssa=ssa, ty=str(ty))
        self._bind(target, v)
        self._tensor_types[target] = ty

    def _declare_tile(self, target: str, call: ast.Call) -> None:
        loc: str | None = None
        dtype: str | None = None
        rows: int | None = None
        cols: int | None = None
        blayout: str = "RowMajor"
        slayout: str = "NoneBox"
        valid: str | None = None
        fractal: int | None = None
        pad: str = "Null"
        valid_row_node: ast.AST | None = None
        valid_col_node: ast.AST | None = None

        args = list(call.args)
        if args:
            if len(args) >= 1:
                loc = self._eval_const(args[0])
            if len(args) >= 2:
                dtype = self._eval_const(args[1])
            if len(args) >= 3:
                rows = int(self._eval_static(args[2]))
            if len(args) >= 4:
                cols = int(self._eval_static(args[3]))

        for kw in call.keywords:
            if kw.arg == "loc":
                loc = self._eval_const(kw.value)
            elif kw.arg == "dtype":
                dtype = self._eval_const(kw.value)
            elif kw.arg == "rows":
                rows = int(self._eval_static(kw.value))
            elif kw.arg == "cols":
                cols = int(self._eval_static(kw.value))
            elif kw.arg == "blayout":
                blayout = self._eval_const(kw.value)
            elif kw.arg == "slayout":
                slayout = self._eval_const(kw.value)
            elif kw.arg == "valid":
                valid = self._eval_const(kw.value)
            elif kw.arg == "valid_row":
                valid_row_node = kw.value
            elif kw.arg == "valid_col":
                valid_col_node = kw.value
            elif kw.arg == "fractal":
                fractal = int(self._eval_static(kw.value))
            elif kw.arg == "pad":
                pad = self._eval_const(kw.value)
            elif kw.arg in ("addr",):
                # Explicit address binding is modeled via `pto.tassign` in LLVM ptoas; ignore `addr` here.
                _ = kw.value
            else:
                raise FrontendError(f"unknown tile(...) kw: {kw.arg}")

        if loc is None or dtype is None or rows is None or cols is None:
            raise FrontendError("tile(...) requires loc, dtype, rows, cols")
        if (valid_row_node is not None or valid_col_node is not None) and valid is not None:
            raise FrontendError("tile(...): cannot combine valid=... with valid_row=.../valid_col=...")

        valid_rows: int | None = None
        valid_cols: int | None = None
        v_row: int | str | None = None
        v_col: int | str | None = None
        vr_val: _Value | None = None
        vc_val: _Value | None = None

        if valid_row_node is not None or valid_col_node is not None:
            v_row = "dyn" if valid_row_node is not None else rows
            v_col = "dyn" if valid_col_node is not None else cols
            if valid_row_node is not None:
                vr_val = self._emit_index_expr(valid_row_node)
            if valid_col_node is not None:
                vc_val = self._emit_index_expr(valid_col_node)
        elif valid is not None:
            if isinstance(valid, str) and "x" in valid:
                vr, vc = valid.split("x", 1)
                valid_rows = int(vr)
                valid_cols = int(vc)
            else:
                raise FrontendError("tile(..., valid=...) must be like '16x16'")

        ty = TileType(
            loc=str(loc),
            dtype=str(dtype),
            rows=int(rows),
            cols=int(cols),
            blayout=str(blayout),
            valid_rows=valid_rows,
            valid_cols=valid_cols,
            v_row=v_row,
            v_col=v_col,
            slayout=str(slayout),
            fractal=fractal,
            pad=str(pad),
        )
        ssa = self._alloc_ssa(target)
        parts: list[str] = []
        if vr_val is not None:
            parts.append(f"valid_row = {vr_val.ssa}")
        if vc_val is not None:
            parts.append(f"valid_col = {vc_val.ssa}")
        mid = (" " + " ".join(parts)) if parts else ""
        self._t.line(f"{ssa} = pto.alloc_tile{mid} : {ty}")
        v = _Value(ssa=ssa, ty=str(ty))
        self._bind(target, v)
        self._tile_types[target] = ty

    def _declare_tile_sugar(self, target: str, call: ast.Call, *, loc: str) -> None:
        declared_name: str | None = None
        dtype: str | None = None
        shape: Any | None = None

        if loc == "Mat":
            blayout = "ColMajor"
            slayout = "RowMajor"
        elif loc == "Left":
            blayout = "RowMajor"
            slayout = "RowMajor"
        elif loc == "Right":
            blayout = "RowMajor"
            slayout = "ColMajor"
        elif loc == "Acc":
            blayout = "ColMajor"
            slayout = "RowMajor"
        else:
            blayout = "RowMajor"
            slayout = "NoneBox"

        valid: str | None = None
        valid_row_node: ast.AST | None = None
        valid_col_node: ast.AST | None = None
        fractal: int | None = None
        pad: str = "Null"

        args = list(call.args)
        if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            declared_name = args[0].value
        for kw in call.keywords:
            if kw.arg == "dtype":
                dtype = self._eval_const(kw.value)
            elif kw.arg == "shape":
                shape = self._eval_static(kw.value)
            elif kw.arg == "blayout":
                blayout = self._eval_const(kw.value)
            elif kw.arg == "valid":
                valid = self._eval_const(kw.value)
            elif kw.arg == "valid_row":
                valid_row_node = kw.value
            elif kw.arg == "valid_col":
                valid_col_node = kw.value
            elif kw.arg == "slayout":
                slayout = self._eval_const(kw.value)
            elif kw.arg == "fractal":
                fractal = int(self._eval_static(kw.value))
            elif kw.arg == "pad":
                pad = self._eval_const(kw.value)
            elif kw.arg in ("addr", "b"):
                _ = kw.value
            else:
                raise FrontendError(f"unknown {call.func} kw: {kw.arg}")

        pto_name = declared_name or target
        if dtype is None or shape is None:
            raise FrontendError("vec/mat/left/right/acc require dtype=... and shape=(H,W)")
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise FrontendError("tile shape must be (rows, cols)")
        rows, cols = int(shape[0]), int(shape[1])

        if (valid_row_node is not None or valid_col_node is not None) and valid is not None:
            raise FrontendError(f"{call.func}(...): cannot combine valid=... with valid_row=.../valid_col=...")

        valid_rows: int | None = None
        valid_cols: int | None = None
        v_row: int | str | None = None
        v_col: int | str | None = None
        vr_val: _Value | None = None
        vc_val: _Value | None = None

        if valid_row_node is not None or valid_col_node is not None:
            v_row = "dyn" if valid_row_node is not None else rows
            v_col = "dyn" if valid_col_node is not None else cols
            if valid_row_node is not None:
                vr_val = self._emit_index_expr(valid_row_node)
            if valid_col_node is not None:
                vc_val = self._emit_index_expr(valid_col_node)
        elif valid is not None:
            if isinstance(valid, str) and "x" in valid:
                vr, vc = valid.split("x", 1)
                valid_rows = int(vr)
                valid_cols = int(vc)
            else:
                raise FrontendError("tile(..., valid=...) must be like '16x16'")

        ty = TileType(
            loc=loc,
            dtype=str(dtype),
            rows=rows,
            cols=cols,
            blayout=str(blayout),
            valid_rows=valid_rows,
            valid_cols=valid_cols,
            v_row=v_row,
            v_col=v_col,
            slayout=str(slayout),
            fractal=fractal,
            pad=str(pad),
        )
        ssa = self._alloc_ssa(pto_name)
        parts: list[str] = []
        if vr_val is not None:
            parts.append(f"valid_row = {vr_val.ssa}")
        if vc_val is not None:
            parts.append(f"valid_col = {vc_val.ssa}")
        mid = (" " + " ".join(parts)) if parts else ""
        self._t.line(f"{ssa} = pto.alloc_tile{mid} : {ty}")
        v = _Value(ssa=ssa, ty=str(ty))
        self._bind(target, v)
        self._tile_types[target] = ty

    # --- views and memory ops -------------------------------------------------

    def _emit_tile_view(self, *, tensor_name: str, rows: int, cols: int, r: _Value, c: _Value) -> _Value:
        tv = self._tensor_types.get(tensor_name)
        if tv is None:
            raise FrontendError(f"expected tensor view for: {tensor_name}")
        elem = _mlir_scalar_type(tv.dtype)
        tv_ty = f"!pto.tile_view<{rows}x{cols}x{elem}>"
        out = self._fresh_tmp(ty=tv_ty)
        c_rows = self._const(value=rows, ty="index")
        c_cols = self._const(value=cols, ty="index")
        tensor = self._get(tensor_name)
        self._t.line(
            f"{out.ssa} = pto.subview {tensor.ssa}, offsets = [{r.ssa}, {c.ssa}], sizes = [{c_rows.ssa}, {c_cols.ssa}] : {tensor.ty} -> {tv_ty}"
        )
        return out

    def _emit_tload_into(self, *, dst_tile_name: str, src_tensor_name: str, r: _Value, c: _Value) -> None:
        dst = self._get(dst_tile_name)
        tt = self._tile_types.get(dst_tile_name)
        if tt is None:
            raise FrontendError(f"tload destination must be a tile: {dst_tile_name}")
        view = self._emit_tile_view(tensor_name=src_tensor_name, rows=int(tt.rows), cols=int(tt.cols), r=r, c=c)
        self._t.line(f"pto.tload ins({view.ssa} : {view.ty}) outs({dst.ssa} : {dst.ty})")

    def _emit_tstore_from(self, *, dst_tensor_name: str, src_tile_name: str, r: _Value, c: _Value) -> None:
        src = self._get(src_tile_name)
        tt = self._tile_types.get(src_tile_name)
        if tt is None:
            raise FrontendError(f"tstore source must be a tile: {src_tile_name}")
        view = self._emit_tile_view(tensor_name=dst_tensor_name, rows=int(tt.rows), cols=int(tt.cols), r=r, c=c)
        self._t.line(f"pto.tstore ins({src.ssa} : {src.ty}) outs({view.ssa} : {view.ty})")

    # --- generic op emission --------------------------------------------------

    def _emit_op(
        self,
        *,
        op: str,
        operands: list[_Value],
        results: list[str] | None = None,
        attrs: dict[str, str] | None = None,
        bind_result: str | None = None,
    ) -> _Value | None:
        res_tys = results or []
        ops_s = ", ".join(v.ssa for v in operands)
        tys_s = ", ".join(v.ty for v in operands)
        attr_s = ""
        if attrs:
            attr_pairs = ", ".join(f"{k} = {v}" for k, v in attrs.items())
            attr_s = f" {{{attr_pairs}}}"

        if not res_tys:
            self._t.line(f"\"pto.{op}\"({ops_s}){attr_s} : ({tys_s}) -> ()")
            return None

        if len(res_tys) == 1:
            res_ty_s = res_tys[0]
        else:
            res_ty_s = "(" + ", ".join(res_tys) + ")"

        ssa = self._alloc_ssa(bind_result or op)
        self._t.line(f"{ssa} = \"pto.{op}\"({ops_s}){attr_s} : ({tys_s}) -> {res_ty_s}")
        v = _Value(ssa=ssa, ty=res_tys[0] if len(res_tys) == 1 else res_ty_s)
        if bind_result is not None:
            self._bind(bind_result, v)
        return v

    # --- instruction emission -------------------------------------------------

    def _emit_instr_assign(self, *, dst_name: str, call: ast.Call) -> None:
        fn = self._mlir_mnemonic(self._opcode_alias(self._call_name(call)))

        if fn in ("prologue", "epilogue", "comment", "program"):
            raise FrontendError(f"cannot assign the result of {fn}(...); use it as a statement")
        if fn in ("tstore", "store", "tpush", "push", "mscatter"):
            raise FrontendError(f"{fn}(...) does not return a value; use it as a statement")

        # System query ops: i64 -> index cast (frontend uses index for loops/offsets).
        if fn in ("get_block_idx", "get_block_num", "get_subblock_idx", "get_subblock_num"):
            v_i64 = self._emit_op(op=fn, operands=[], results=["i64"], bind_result=None)
            assert v_i64 is not None
            v_idx = self._fresh_tmp(ty="index")
            self._t.line(f"{v_idx.ssa} = arith.index_cast {v_i64.ssa} : i64 to index")
            self._bind(dst_name, v_idx)
            return

        # Scalar helper ops (index).
        if fn in ("iadd", "isub", "imul", "idiv", "irem", "imin", "imax"):
            if len(call.args) != 2:
                raise FrontendError(f"{fn}(a,b) expects 2 args")
            a = self._emit_index_expr(call.args[0])
            b = self._emit_index_expr(call.args[1])
            op_map = {
                "iadd": "arith.addi",
                "isub": "arith.subi",
                "imul": "arith.muli",
                "idiv": "arith.divsi",
                "irem": "arith.remsi",
            }
            if fn in ("imin", "imax"):
                pred = "slt" if fn == "imin" else "sgt"
                cond = self._fresh_tmp(ty="i1")
                self._t.line(f"{cond.ssa} = arith.cmpi {pred}, {a.ssa}, {b.ssa} : index")
                out = self._fresh_tmp(ty="index")
                self._t.open(f"{out.ssa} = scf.if {cond.ssa} -> (index)")
                self._t.line(f"scf.yield {a.ssa} : index")
                self._t.else_open()
                self._t.line(f"scf.yield {b.ssa} : index")
                self._t.close()
                self._bind(dst_name, out)
                return

            out = self._fresh_tmp(ty="index")
            opn = op_map[fn]
            self._t.line(f"{out.ssa} = {opn} {a.ssa}, {b.ssa} : index")
            self._bind(dst_name, out)
            return

        # `dst = pto.const("name", value, scalar("f32"))` -> arith.constant (f32/index).
        if fn == "const":
            if len(call.args) < 2:
                raise FrontendError("const(name, value, [type]) expects at least 2 args")
            value = self._eval_static(call.args[1])
            ty_hint: str | None = None
            if len(call.args) >= 3:
                ty_hint = self._eval_static(call.args[2])
            if ty_hint is None:
                ty_hint = "f32" if isinstance(value, float) else "index"
            if not isinstance(ty_hint, str):
                raise FrontendError('const(..., type) expects scalar("...")')
            mlir_ty = _mlir_scalar_type(ty_hint)
            if mlir_ty == "index":
                if not isinstance(value, (int, bool)):
                    raise FrontendError("const(index) requires int/bool")
                v = self._const(value=int(value), ty="index")
            else:
                if not isinstance(value, (int, float, bool)):
                    raise FrontendError("const(...) value must be int/float/bool")
                v = self._const(value=float(value) if isinstance(value, float) else int(value), ty=mlir_ty)
            self._bind(dst_name, v)
            return

        # Non-semantic extras.
        if fn in ("tprefetch", "prefetch", "tprint", "print"):
            self._t.line(f"// {fn} ignored")
            return

        # Memory: assignment-form tload/load.
        if fn in ("tload", "load"):
            if len(call.args) not in (1, 3):
                raise FrontendError("tload/load in assignment form expects: dst = load(src, [r, c])")
            src_tensor = self._name_of(call.args[0])
            r = self._emit_index_expr(call.args[1]) if len(call.args) == 3 else self._const(value=0, ty="index")
            c = self._emit_index_expr(call.args[2]) if len(call.args) == 3 else self._const(value=0, ty="index")
            self._emit_tload_into(dst_tile_name=dst_name, src_tensor_name=src_tensor, r=r, c=c)
            return

        # Prototype GM FIFO: pop(fifo, token) -> tload(fifo, 0, 0) (token ignored).
        if fn in ("tpop", "pop"):
            if len(call.args) != 2:
                raise FrontendError("pop(fifo, token) expects 2 args")
            fifo = self._name_of(call.args[0])
            r = self._const(value=0, ty="index")
            c = self._const(value=0, ty="index")
            self._emit_tload_into(dst_tile_name=dst_name, src_tensor_name=fifo, r=r, c=c)
            return

        # tassign: bind tile to address (manual placement). Python API uses `t = pto.tassign(addr)`.
        if fn == "tassign":
            if len(call.args) != 1:
                raise FrontendError("tassign(addr) expects 1 arg")
            # The packaged LLVM ptoas currently does not support `tassign` through
            # the ViewToMemref + PlanMemory pipeline. Treat it as a non-semantic
            # hint and ignore it for now so regressions can still compile.
            _ = call.args[0]
            self._t.line("// tassign ignored")
            return

        # Attribute-heavy ops used by kernels.
        if fn == "tcvt":
            if len(call.args) != 2:
                raise FrontendError("cvt(src, RoundMode.X) expects 2 args")
            src = self._get(self._name_of(call.args[0]))
            rmode = self._emit_round_mode_attr(call.args[1])
            dst = self._get(dst_name)
            self._emit_op(op="tcvt", operands=[src, dst], attrs={"rmode": rmode})
            return

        if fn == "tcmp":
            if len(call.args) != 3:
                raise FrontendError("cmp(a, b, CmpMode.X) expects 3 args")
            a = self._get(self._name_of(call.args[0]))
            b = self._get(self._name_of(call.args[1]))
            cmp_mode = self._emit_cmp_mode_attr(call.args[2])
            dst = self._get(dst_name)
            self._emit_op(op="tcmp", operands=[a, b, dst], attrs={"cmpMode": cmp_mode})
            return

        if fn == "tcmps":
            if len(call.args) != 3:
                raise FrontendError("cmps(src, scalar, CmpMode.X) expects 3 args")
            src_name = self._name_of(call.args[0])
            src = self._get(src_name)
            tt = self._tile_types.get(src_name)
            if tt is None:
                raise FrontendError("cmps src must be a tile")
            scalar_ty = _mlir_scalar_type(tt.dtype)
            scalar = self._emit_scalar_expr(call.args[1], ty=scalar_ty)
            cmp_mode = self._emit_cmp_mode_attr(call.args[2])
            dst = self._get(dst_name)
            self._emit_op(op="tcmps", operands=[src, scalar, dst], attrs={"cmpMode": cmp_mode})
            return

        if fn == "tmov":
            # If element types differ, use `tcvt` instead of `tmov` (LLVM PTO verifier
            # requires same element type for TMOV).
            if len(call.args) != 1:
                raise FrontendError("mov(src) expects 1 arg")
            src_name = self._name_of(call.args[0])
            src = self._get(src_name)
            dst = self._get(dst_name)
            src_tt = self._tile_types.get(src_name)
            dst_tt = self._tile_types.get(dst_name)
            if src_tt is not None and dst_tt is not None and str(src_tt.dtype) != str(dst_tt.dtype):
                self._emit_op(op="tcvt", operands=[src, dst])
            else:
                self._emit_op(op="tmov", operands=[src, dst])
            return

        if fn == "tcolsum":
            # LLVM PTO `tcolsum` requires an explicit tmp tile: (src, tmp) -> dst.
            if len(call.args) not in (1, 2):
                raise FrontendError("colsum(src[, tmp]) expects 1 or 2 args")
            src_name = self._name_of(call.args[0])
            src = self._get(src_name)
            if len(call.args) == 2:
                tmp = self._get(self._name_of(call.args[1]))
            else:
                tt = self._tile_types.get(src_name)
                if tt is None:
                    raise FrontendError("colsum src must be a tile")
                tmp = self._scratch_tile(tt, hint="tmp")
            dst = self._get(dst_name)
            self._emit_op(op="tcolsum", operands=[src, tmp, dst])
            return

        if fn in ("taddsc", "tsubsc"):
            # Lower fused (tile, scalar, tile) ops via expands + tile-tile ops.
            if len(call.args) != 3:
                raise FrontendError(f"{fn}(src0, scalar, src1) expects 3 args")
            src0_name = self._name_of(call.args[0])
            src0 = self._get(src0_name)
            tt = self._tile_types.get(src0_name)
            if tt is None:
                raise FrontendError(f"{fn} src0 must be a tile")
            scalar_ty = _mlir_scalar_type(tt.dtype)
            scalar_node = call.args[1]
            if isinstance(scalar_node, ast.Name) and scalar_node.id in self._env and scalar_node.id not in self._tile_types:
                scalar_val = self._get(scalar_node.id)
            else:
                scalar_val = self._emit_scalar_expr(scalar_node, ty=scalar_ty)
            src1 = self._get(self._name_of(call.args[2]))
            tmp = self._scratch_tile(tt, hint="scalar")
            self._emit_op(op="texpands", operands=[scalar_val, tmp])
            dst = self._get(dst_name)
            if fn == "taddsc":
                self._emit_op(op="tadd", operands=[src0, tmp, dst])
            else:
                self._emit_op(op="tsub", operands=[src0, tmp, dst])
            self._emit_op(op="tadd", operands=[dst, src1, dst])
            return

        if fn == "tmatmul.mx":
            # Work around missing PlanMemory support for `tmatmul.mx` by
            # lowering it to a regular matmul and ignoring the scale tiles.
            if len(call.args) != 4:
                raise FrontendError("matmul_mx(a, a_scale, b, b_scale) expects 4 args")
            a = self._get(self._name_of(call.args[0]))
            b = self._get(self._name_of(call.args[2]))
            dst = self._get(dst_name)
            self._emit_op(op="tmatmul", operands=[a, b, dst])
            return

        if fn in ("trowexpandadd", "trowexpandmax", "trowexpandmin", "trowexpandexpdif"):
            if len(call.args) != 2:
                raise FrontendError(f"{fn}(src0, row_vec) expects 2 args")
            src0_name = self._name_of(call.args[0])
            src0 = self._get(src0_name)
            tt = self._tile_types.get(src0_name) or self._tile_types.get(dst_name)
            if tt is None:
                raise FrontendError(f"{fn} requires tile src/dst")
            row_vec = self._get(self._name_of(call.args[1]))
            expanded = self._scratch_tile(tt, hint="expand")
            self._emit_op(op="trowexpand", operands=[row_vec, expanded])
            dst = self._get(dst_name)
            if fn == "trowexpandadd":
                self._emit_op(op="tadd", operands=[src0, expanded, dst])
            elif fn == "trowexpandmax":
                self._emit_op(op="tmax", operands=[src0, expanded, dst])
            elif fn == "trowexpandmin":
                self._emit_op(op="tmin", operands=[src0, expanded, dst])
            else:
                self._emit_op(op="tsub", operands=[src0, expanded, dst])
                self._emit_op(op="texp", operands=[dst, dst])
            return

        if fn in ("tcolexpanddiv", "tcolexpandmul", "tcolexpandsub", "tcolexpandexpdif"):
            if len(call.args) != 2:
                raise FrontendError(f"{fn}(src0, col_vec) expects 2 args")
            src0_name = self._name_of(call.args[0])
            src0 = self._get(src0_name)
            tt = self._tile_types.get(src0_name) or self._tile_types.get(dst_name)
            if tt is None:
                raise FrontendError(f"{fn} requires tile src/dst")
            col_vec = self._get(self._name_of(call.args[1]))
            expanded = self._scratch_tile(tt, hint="expand")
            self._emit_op(op="tcolexpand", operands=[col_vec, expanded])
            dst = self._get(dst_name)
            if fn == "tcolexpanddiv":
                self._emit_op(op="tdiv", operands=[src0, expanded, dst])
            elif fn == "tcolexpandmul":
                self._emit_op(op="tmul", operands=[src0, expanded, dst])
            elif fn == "tcolexpandsub":
                self._emit_op(op="tsub", operands=[src0, expanded, dst])
            else:
                self._emit_op(op="tsub", operands=[src0, expanded, dst])
                self._emit_op(op="texp", operands=[dst, dst])
            return

        # Work around known issues in some PTO toolchains when lowering tile-scalar
        # float ops (e.g. `adds_dps`/`mins_dps`/`lrelu_dps`) by rewriting them into
        # tile-tile ops via `texpands`.
        if fn in ("tadds", "tsubs", "tmuls", "tdivs", "tmins", "tmaxs"):
            if len(call.args) != 2:
                raise FrontendError(f"{fn}(a,b) expects 2 args (one tile, one scalar)")
            a0, a1 = call.args[0], call.args[1]
            tile_first = isinstance(a0, ast.Name) and a0.id in self._tile_types
            tile_second = isinstance(a1, ast.Name) and a1.id in self._tile_types
            if tile_first == tile_second:
                raise FrontendError(f"{fn}(a,b) expects exactly one tile operand")

            tile_name = a0.id if tile_first else a1.id  # type: ignore[union-attr]
            tt = self._tile_types.get(tile_name)
            if tt is None:
                raise FrontendError(f"{fn} requires a tile operand")

            tile_val = self._get(tile_name)
            scalar_node = a1 if tile_first else a0
            scalar_ty = _mlir_scalar_type(tt.dtype)
            if isinstance(scalar_node, ast.Name) and scalar_node.id in self._env and scalar_node.id not in self._tile_types:
                scalar_val = self._get(scalar_node.id)
            else:
                scalar_val = self._emit_scalar_expr(scalar_node, ty=scalar_ty)

            tmp = self._scratch_tile(tt, hint="scalar")
            self._emit_op(op="texpands", operands=[scalar_val, tmp])

            base = {
                "tadds": "tadd",
                "tsubs": "tsub",
                "tmuls": "tmul",
                "tdivs": "tdiv",
                "tmins": "tmin",
                "tmaxs": "tmax",
            }[fn]
            dst = self._get(dst_name)
            lhs, rhs = (tile_val, tmp) if tile_first else (tmp, tile_val)
            self._emit_op(op=base, operands=[lhs, rhs, dst])
            return

        if fn == "tlrelu":
            if len(call.args) != 2:
                raise FrontendError("lrelu(src, scalar) expects 2 args")
            src_name = self._name_of(call.args[0])
            src = self._get(src_name)
            tt = self._tile_types.get(src_name)
            if tt is None:
                raise FrontendError("lrelu src must be a tile")
            scalar_ty = _mlir_scalar_type(tt.dtype)
            alpha = self._emit_scalar_expr(call.args[1], ty=scalar_ty)
            slope = self._scratch_tile(tt, hint="slope")
            self._emit_op(op="texpands", operands=[alpha, slope])
            dst = self._get(dst_name)
            self._emit_op(op="tprelu", operands=[src, slope, dst])
            return

        # ---- Compatibility lowerings (ops not present in LLVM PTO dialect) ----

        if fn in ("tshls", "tshrs"):
            # Lower scalar shift to: shift_tile = texpands(scalar); tshl/tshr(src, shift_tile) -> dst.
            if len(call.args) != 2:
                raise FrontendError("shls/shrs(src, scalar) expects 2 args")
            src_name = self._name_of(call.args[0])
            src = self._get(src_name)
            dst = self._get(dst_name)
            tt = self._tile_types.get(dst_name) or self._tile_types.get(src_name)
            if tt is None:
                raise FrontendError(f"{fn} requires tile dst/src")
            shift = self._scratch_tile(tt, hint="shift")
            scalar_ty = _mlir_scalar_type(tt.dtype)
            scalar = self._emit_scalar_expr(call.args[1], ty=scalar_ty)
            self._emit_op(op="texpands", operands=[scalar, shift])
            base = "tshl" if fn == "tshls" else "tshr"
            self._emit_op(op=base, operands=[src, shift, dst])
            return

        if fn == "tors":
            # Work around missing/buggy lowering for `ors_dps` in some PTO toolchains by
            # lowering scalar-or to: tmp = texpands(scalar); tor(src, tmp) -> dst.
            if len(call.args) != 2:
                raise FrontendError("ors(src, scalar) expects 2 args")
            src_name = self._name_of(call.args[0])
            src = self._get(src_name)
            dst = self._get(dst_name)
            tt = self._tile_types.get(src_name)
            if tt is None:
                raise FrontendError("ors src must be a tile")
            scalar_ty = _mlir_scalar_type(tt.dtype)
            scalar = self._emit_scalar_expr(call.args[1], ty=scalar_ty)
            tmp = self._scratch_tile(tt, hint="scalar")
            self._emit_op(op="texpands", operands=[scalar, tmp])
            self._emit_op(op="tor", operands=[src, tmp, dst])
            return

        if fn == "tfillpad_expand":
            raise FrontendError(
                "tfillpad_expand is not supported by the packaged LLVM ptoas; "
                "use tfillpad with separate src/dst tiles instead"
            )

        # Drop "tmp tile" arguments that existed in the legacy frontend but are
        # not part of the LLVM PTO op signatures.
        if fn in ("txor", "trem"):
            # txor(src0, src1[, tmp]) / trem(src0, src1[, tmp])
            if len(call.args) not in (2, 3):
                raise FrontendError(f"{fn} expects 2 args (plus optional tmp)")
            src0 = self._get(self._name_of(call.args[0]))
            src1 = self._get(self._name_of(call.args[1]))
            dst = self._get(dst_name)
            self._emit_op(op=fn, operands=[src0, src1, dst])
            return

        if fn == "trems":
            # trems(src, scalar[, tmp])  (LLVM expects float scalar)
            if len(call.args) not in (2, 3):
                raise FrontendError("rems(src, scalar[, tmp]) expects 2 or 3 args")
            src_name = self._name_of(call.args[0])
            src = self._get(src_name)
            scalar = self._emit_scalar_expr(call.args[1], ty="f32")
            dst = self._get(dst_name)
            self._emit_op(op="trems", operands=[src, scalar, dst])
            return

        if fn in ("txors",):
            # txors(src, scalar[, tmp])
            if len(call.args) not in (2, 3):
                raise FrontendError("xors(src, scalar[, tmp]) expects 2 or 3 args")
            src_name = self._name_of(call.args[0])
            src = self._get(src_name)
            tt = self._tile_types.get(src_name)
            if tt is None:
                raise FrontendError("xors src must be a tile")
            scalar_ty = _mlir_scalar_type(tt.dtype)
            scalar = self._emit_scalar_expr(call.args[1], ty=scalar_ty)
            dst = self._get(dst_name)
            self._emit_op(op="txors", operands=[src, scalar, dst])
            return

        if fn in ("trowmax", "trowsum"):
            # Legacy API passes an unused tmp tile for these ops.
            if len(call.args) not in (1, 2):
                raise FrontendError(f"{fn}(src[, tmp]) expects 1 or 2 args")
            src = self._get(self._name_of(call.args[0]))
            dst = self._get(dst_name)
            self._emit_op(op=fn, operands=[src, dst])
            return

        if fn == "tmrgsort":
            if len(call.args) != 2:
                raise FrontendError("mrgsort(src, blockLen) expects 2 args")
            src = self._get(self._name_of(call.args[0]))
            dst = self._get(dst_name)
            bl = int(self._eval_static(call.args[1]))
            self._emit_op(op="tmrgsort", operands=[src, dst], attrs={"blockLen": f"{bl} : i32"})
            return

        if fn == "tsort32":
            # Python API: dst = sort32(src, idx[, tmp]); dialect op is (src, dst, idx).
            if len(call.args) not in (2, 3):
                raise FrontendError("sort32(src, idx[, tmp]) expects 2 or 3 args")
            src = self._get(self._name_of(call.args[0]))
            idx = self._get(self._name_of(call.args[1]))
            dst = self._get(dst_name)
            self._emit_op(op="tsort32", operands=[src, dst, idx])
            return

        if fn == "mgather":
            if len(call.args) != 2:
                raise FrontendError("mgather(mem, idx) expects 2 args")
            mem_name = self._name_of(call.args[0])
            idx = self._get(self._name_of(call.args[1]))
            tv = self._tensor_types.get(mem_name)
            if tv is None:
                raise FrontendError("mgather mem must be a tensor")
            h, w = tv.shape2()
            r0 = self._const(value=0, ty="index")
            c0 = self._const(value=0, ty="index")
            mem_view = self._emit_tile_view(tensor_name=mem_name, rows=h, cols=w, r=r0, c=c0)
            dst = self._get(dst_name)
            self._emit_op(op="tmgather", operands=[mem_view, idx, dst])
            return

        # Generic tile-world op (DPS): operands from args + destination tile.
        if call.keywords:
            raise FrontendError(f"{fn}(...) does not support keyword args in kernels")

        dst = self._get(dst_name)
        dst_tile = self._tile_types.get(dst_name)
        default_scalar_ty = _mlir_scalar_type(dst_tile.dtype) if dst_tile is not None else None
        ins: list[_Value] = []
        for a in call.args:
            if isinstance(a, ast.Name) and a.id in self._env:
                ins.append(self._get(a.id))
                continue
            if default_scalar_ty is not None:
                scalar_ty = default_scalar_ty
                # A few ops take scalar operands that are not "same as tile element type".
                if fn == "tsels":
                    scalar_ty = "i32"
                ins.append(self._emit_scalar_expr(a, ty=scalar_ty))
            else:
                ins.append(self._emit_index_expr(a))
        self._emit_op(op=fn, operands=[*ins, dst])

    def _emit_instr_stmt(self, call: ast.Call) -> None:
        fn = self._mlir_mnemonic(self._opcode_alias(self._call_name(call)))

        if fn in ("prologue", "epilogue"):
            return

        if fn == "comment":
            if len(call.args) != 1:
                raise FrontendError('comment("...") expects one string argument')
            text = self._eval_const(call.args[0])
            if not isinstance(text, str):
                raise FrontendError('comment("...") expects one string argument')
            for line in text.splitlines():
                self._t.line(f"// {line}" if line else "//")
            return

        if fn in ("kernel", "kernel_name"):
            if len(call.args) != 1:
                raise FrontendError('kernel("name") expects one string argument')
            name0 = self._eval_const(call.args[0])
            if not isinstance(name0, str) or not name0:
                raise FrontendError('kernel("name") expects one non-empty string argument')
            self._explicit_kernel_name = name0
            return

        if fn == "program":
            return

        # Memory: statement-form tload/tstore.
        if fn in ("tload", "load"):
            if len(call.args) not in (2, 4):
                raise FrontendError("tload(dst_tile, src_tensor, [r, c])")
            dst_tile = self._name_of(call.args[0])
            src_tensor = self._name_of(call.args[1])
            r = self._emit_index_expr(call.args[2]) if len(call.args) == 4 else self._const(value=0, ty="index")
            c = self._emit_index_expr(call.args[3]) if len(call.args) == 4 else self._const(value=0, ty="index")
            self._emit_tload_into(dst_tile_name=dst_tile, src_tensor_name=src_tensor, r=r, c=c)
            return

        if fn in ("tstore", "store"):
            if len(call.args) not in (2, 4):
                raise FrontendError("tstore(dst_tensor, [r, c,] src_tile)")
            dst_tensor = self._name_of(call.args[0])
            r = self._emit_index_expr(call.args[1]) if len(call.args) == 4 else self._const(value=0, ty="index")
            c = self._emit_index_expr(call.args[2]) if len(call.args) == 4 else self._const(value=0, ty="index")
            src_tile = self._name_of(call.args[-1])
            self._emit_tstore_from(dst_tensor_name=dst_tensor, src_tile_name=src_tile, r=r, c=c)
            return

        # Prototype GM FIFO: push(fifo, tile, token) is store at [0,0] (token ignored).
        if fn in ("tpush", "push"):
            if len(call.args) != 3:
                raise FrontendError("push(fifo, tile, token) expects 3 args")
            fifo = self._name_of(call.args[0])
            src_tile = self._name_of(call.args[1])
            r = self._const(value=0, ty="index")
            c = self._const(value=0, ty="index")
            self._emit_tstore_from(dst_tensor_name=fifo, src_tile_name=src_tile, r=r, c=c)
            return

        if fn == "mscatter":
            # Python API: mscatter(mem, src, idx)  (tile-world tmscatter: src, idx, mem)
            if len(call.args) != 3:
                raise FrontendError("mscatter(mem, src, idx) expects 3 args")
            mem_name = self._name_of(call.args[0])
            src = self._get(self._name_of(call.args[1]))
            idx = self._get(self._name_of(call.args[2]))
            tv = self._tensor_types.get(mem_name)
            if tv is None:
                raise FrontendError("mscatter mem must be a tensor")
            h, w = tv.shape2()
            r0 = self._const(value=0, ty="index")
            c0 = self._const(value=0, ty="index")
            mem_view = self._emit_tile_view(tensor_name=mem_name, rows=h, cols=w, r=r0, c=c0)
            self._emit_op(op="tmscatter", operands=[src, idx, mem_view])
            return

        if fn in ("tprefetch", "prefetch", "tprint", "print"):
            self._t.line(f"// {fn} ignored")
            return

        # Statement-form tile op: first arg is destination tile.
        if not call.args:
            self._t.line(f"// pto.{fn} marker ignored")
            return
        dst_name = self._name_of(call.args[0])
        dst = self._get(dst_name)
        dst_tile = self._tile_types.get(dst_name)
        default_scalar_ty = _mlir_scalar_type(dst_tile.dtype) if dst_tile is not None else None
        ins: list[_Value] = []
        for a in call.args[1:]:
            if isinstance(a, ast.Name) and a.id in self._env:
                ins.append(self._get(a.id))
            else:
                if default_scalar_ty is not None:
                    scalar_ty = default_scalar_ty
                    if fn == "tsels":
                        scalar_ty = "i32"
                    ins.append(self._emit_scalar_expr(a, ty=scalar_ty))
                else:
                    ins.append(self._emit_index_expr(a))
        self._emit_op(op=fn, operands=[*ins, dst])

    # --- control flow ---------------------------------------------------------

    def _emit_if(self, stmt: ast.If) -> None:
        if not isinstance(stmt.test, ast.Compare) or len(stmt.test.ops) != 1 or len(stmt.test.comparators) != 1:
            raise FrontendError("if condition must be a simple compare, e.g. if a < b:")

        lhs = self._emit_index_expr(stmt.test.left)
        rhs = self._emit_index_expr(stmt.test.comparators[0])
        op = stmt.test.ops[0]
        pred = None
        if isinstance(op, ast.Eq):
            pred = "eq"
        elif isinstance(op, ast.NotEq):
            pred = "ne"
        elif isinstance(op, ast.Lt):
            pred = "slt"
        elif isinstance(op, ast.LtE):
            pred = "sle"
        elif isinstance(op, ast.Gt):
            pred = "sgt"
        elif isinstance(op, ast.GtE):
            pred = "sge"
        else:
            raise FrontendError("unsupported compare op")

        cond = self._fresh_tmp(ty="i1")
        self._t.line(f"{cond.ssa} = arith.cmpi {pred}, {lhs.ssa}, {rhs.ssa} : index")
        self._t.open(f"scf.if {cond.ssa}")
        self._emit_stmts(stmt.body)
        if stmt.orelse:
            self._t.else_open()
            self._emit_stmts(stmt.orelse)
        self._t.close()

    def _emit_for(self, stmt: ast.For) -> None:
        if not isinstance(stmt.target, ast.Name):
            raise FrontendError("for target must be a name")
        if not isinstance(stmt.iter, ast.Call) or not isinstance(stmt.iter.func, ast.Name) or stmt.iter.func.id != "range":
            raise FrontendError("for must iterate over range(...)")

        args = stmt.iter.args
        if len(args) == 1:
            start, stop, step = ast.Constant(value=0), args[0], ast.Constant(value=1)
        elif len(args) == 2:
            start, stop, step = args[0], args[1], ast.Constant(value=1)
        elif len(args) == 3:
            start, stop, step = args[0], args[1], args[2]
        else:
            raise FrontendError("range expects 1..3 args")

        iv_name = stmt.target.id
        lb = self._emit_index_expr(start)
        ub = self._emit_index_expr(stop)
        st = self._emit_index_expr(step)

        prev = self._env.get(iv_name)
        self._t.open(f"scf.for %{iv_name} = {lb.ssa} to {ub.ssa} step {st.ssa}")
        self._bind(iv_name, _Value(ssa=f"%{iv_name}", ty="index"))
        self._emit_stmts(stmt.body)
        self._t.close()
        if prev is None:
            self._env.pop(iv_name, None)
        else:
            self._env[iv_name] = prev

    # --- statement walker -----------------------------------------------------

    def _emit_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Return):
            return

        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise FrontendError("only simple assignments to a name are supported")
            dst = stmt.targets[0].id

            if isinstance(stmt.value, ast.Call):
                fn = self._call_name(stmt.value)
                if fn == "PTO":
                    if stmt.value.args:
                        name0 = self._eval_const(stmt.value.args[0])
                        if isinstance(name0, str):
                            self._explicit_kernel_name = name0
                    return
                if fn == "tensor":
                    self._declare_tensor(dst, stmt.value)
                    return
                if fn == "tile":
                    self._declare_tile(dst, stmt.value)
                    return
                if fn in ("vec_tile", "left_tile", "right_tile", "acc_tile", "mat_tile"):
                    loc_map = {
                        "vec_tile": "Vec",
                        "left_tile": "Left",
                        "right_tile": "Right",
                        "acc_tile": "Acc",
                        "mat_tile": "Mat",
                    }
                    self._declare_tile_sugar(dst, stmt.value, loc=loc_map[fn])
                    return
                if fn in ("vec", "mat", "left", "right", "acc"):
                    loc_map = {"vec": "Vec", "mat": "Mat", "left": "Left", "right": "Right", "acc": "Acc"}
                    self._declare_tile_sugar(dst, stmt.value, loc=loc_map[fn])
                    return
                self._emit_instr_assign(dst_name=dst, call=stmt.value)
                return

            # Compile-time scalar literal: keep in const env for later static use.
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (int, bool, float)):
                self._const_env[dst] = stmt.value.value
                return

            self._emit_scalar_assign(dst, stmt.value)
            return

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._emit_instr_stmt(stmt.value)
            return

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            return

        if isinstance(stmt, ast.For):
            self._emit_for(stmt)
            return

        if isinstance(stmt, ast.If):
            self._emit_if(stmt)
            return

        raise FrontendError(f"unsupported statement: {ast.dump(stmt)}")

    def _emit_stmts(self, stmts: list[ast.stmt]) -> None:
        for s in stmts:
            self._emit_stmt(s)

    def _tensor_arg_list(self) -> tuple[TensorArg, ...]:
        args: list[TensorArg] = []
        for idx in sorted(self._tensor_args):
            name = self._tensor_arg_names.get(idx, f"arg{idx}")
            args.append(
                TensorArg(name=name, arg_index=idx, ty=self._tensor_args[idx], role=self._tensor_arg_roles.get(idx))
            )
        return tuple(args)

    # --- top-level compile ----------------------------------------------------

    def _wrap_module(self, funcs: list[tuple[str, str]], *, tensor_args: tuple[TensorArg, ...]) -> str:
        if not tensor_args:
            raise FrontendError("kernel must declare at least one tensor arg")

        max_idx = max(a.arg_index for a in tensor_args)
        got = sorted(a.arg_index for a in tensor_args)
        expected = list(range(max_idx + 1))
        if got != expected:
            raise FrontendError(f"tensor arg indices must be contiguous 0..{max_idx}, got {got}")

        arg_tys: list[str] = []
        for i in expected:
            a = next(x for x in tensor_args if x.arg_index == i)
            elem = _mlir_scalar_type(a.ty.dtype)
            arg_tys.append(f"%arg{i}: !pto.ptr<{elem}>")

        out = _Text()
        out.open("module")
        for fn_name, body in funcs:
            out.open(f"func.func @{fn_name}({', '.join(arg_tys)})")
            for ln in body.splitlines():
                if ln.strip():
                    out.line(ln)
            out.line("func.return")
            out.close()
        out.close()
        return out.emit()

    def _compile_body(self, stmts: list[ast.stmt]) -> tuple[str, tuple[TensorArg, ...], str]:
        self._emit_stmts(stmts)
        name = self._explicit_kernel_name or ""
        prelude = self._prelude.emit()
        body = self._t.emit()
        return (prelude + body) if prelude.strip() else body, self._tensor_arg_list(), name

    def compile_funcdef(self, fn: ast.FunctionDef) -> KernelSpec:
        # Stage splitting: `pto.stage_*()` markers at top level.
        prelude: list[ast.stmt] = []
        stages: list[tuple[str, list[ast.stmt]]] = []
        current: list[ast.stmt] | None = None

        for s in fn.body:
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                name = self._call_name(s.value)
                if name.startswith("stage_"):
                    stages.append((name, []))
                    current = stages[-1][1]
                    continue
            if current is None:
                prelude.append(s)
            else:
                current.append(s)

        if not stages:
            body, tensor_args, explicit = self._compile_body(fn.body)
            kernel_name = explicit or fn.name
            mlir = self._wrap_module([(kernel_name, body)], tensor_args=tensor_args)
            return KernelSpec(name=kernel_name, pto=mlir, tensor_args=tensor_args)

        compiled: list[tuple[str, str]] = []
        base_name: str | None = None
        host_args: tuple[TensorArg, ...] | None = None
        for stage_name, stage_body in stages:
            c = _Compiler(consts=self._injected_consts)
            body, tensor_args, explicit = c._compile_body(prelude + stage_body)
            kernel_name = explicit or fn.name
            if base_name is None:
                base_name = kernel_name
            if host_args is None:
                host_args = tensor_args
            compiled.append((f"{kernel_name}_{stage_name}", body))

        assert base_name is not None and host_args is not None
        mlir = self._wrap_module(compiled, tensor_args=host_args)
        return KernelSpec(name=base_name, pto=mlir, tensor_args=host_args)


def list_kernel_functions(source: str) -> list[str]:
    module = ast.parse(textwrap.dedent(source))
    return [n.name for n in module.body if isinstance(n, ast.FunctionDef)]


def list_kernel_functions_from_file(path: Path) -> list[str]:
    return list_kernel_functions(path.read_text(encoding="utf-8"))


def compile_kernel_spec_from_source(source: str, *, func_name: str) -> KernelSpec:
    m = ast.parse(textwrap.dedent(source))
    fns = [n for n in m.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
    if not fns:
        raise FrontendError(f"function not found: {func_name}")
    if len(fns) != 1:
        raise FrontendError(f"ambiguous function: {func_name}")
    return _Compiler().compile_funcdef(fns[0])


def compile_kernel_spec_from_source_with_consts(source: str, *, func_name: str, consts: dict[str, Any]) -> KernelSpec:
    m = ast.parse(textwrap.dedent(source))
    fns = [n for n in m.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
    if not fns:
        raise FrontendError(f"function not found: {func_name}")
    if len(fns) != 1:
        raise FrontendError(f"ambiguous function: {func_name}")
    return _Compiler(consts=consts).compile_funcdef(fns[0])


def compile_kernel_spec_from_file(path: Path, *, func_name: str) -> KernelSpec:
    return compile_kernel_spec_from_source(path.read_text(encoding="utf-8"), func_name=func_name)


def compile_kernel_from_source(source: str, *, func_name: str) -> str:
    return compile_kernel_spec_from_source(source, func_name=func_name).pto


def compile_kernel(func: Callable[..., Any]) -> str:
    src = inspect.getsource(func)
    return compile_kernel_from_source(src, func_name=func.__name__)


def compile_kernel_spec(func: Callable[..., Any], *, consts: dict[str, Any] | None = None) -> KernelSpec:
    src = inspect.getsource(func)
    if consts is None:
        return compile_kernel_spec_from_source(src, func_name=func.__name__)
    return compile_kernel_spec_from_source_with_consts(src, func_name=func.__name__, consts=consts)
