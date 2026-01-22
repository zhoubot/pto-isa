from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

from .pto_asm import TensorType, TileType


class FrontendError(Exception):
    pass


@dataclass
class _Sym:
    name: str  # without leading %

    @property
    def pto(self) -> str:
        return f"%{self.name}"


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
        return "\n".join(self._lines).strip() + "\n"


class _Compiler:
    def __init__(self) -> None:
        self._t = _Text()
        self._sym: dict[str, _Sym] = {}
        self._tmp_i = 0
        self._next_tensor_arg = 0
        # Python-name -> literal string (emits as an immediate, not an SSA value).
        self._literal: dict[str, str] = {}

    def _tmp(self) -> _Sym:
        self._tmp_i += 1
        name = f"t{self._tmp_i}"
        self._sym[name] = _Sym(name)
        return self._sym[name]

    def _sym_for(self, py_name: str) -> _Sym:
        if py_name not in self._sym:
            self._sym[py_name] = _Sym(py_name)
        return self._sym[py_name]

    def _eval_const(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception as e:
            raise FrontendError(f"expected a literal, got: {ast.dump(node)}") from e

    def _opnd(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            lit = self._literal.get(node.id)
            if lit is not None:
                return lit
            return self._sym_for(node.id).pto
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "1" if node.value else "0"
            return str(node.value)
        raise FrontendError(f"unsupported operand node: {ast.dump(node)}")

    def _declare_tensor(self, target: str, call: ast.Call) -> None:
        dtype: str | None = None
        shape: Any | None = None
        stride: Any | None = None
        layout: str = "ND"

        args = list(call.args)
        if args:
            if len(args) >= 1:
                dtype = self._eval_const(args[0])
            if len(args) >= 2:
                shape = self._eval_const(args[1])
            if len(args) >= 3:
                stride = self._eval_const(args[2])
            if len(args) >= 4:
                layout = self._eval_const(args[3])

        for kw in call.keywords:
            if kw.arg == "dtype":
                dtype = self._eval_const(kw.value)
            elif kw.arg == "shape":
                shape = self._eval_const(kw.value)
            elif kw.arg == "stride":
                stride = self._eval_const(kw.value)
            elif kw.arg == "layout":
                layout = self._eval_const(kw.value)
            elif kw.arg in ("arg", "arg_index"):
                # Parsed below (controls %argN binding).
                pass
            else:
                raise FrontendError(f"unknown tensor(...) kw: {kw.arg}")

        if dtype is None or shape is None:
            raise FrontendError("tensor(...) requires dtype and shape")

        # `tensor(...)` in the Python frontend declares a kernel tensor argument, mapped
        # to `%argN` in declaration order, and introduces a view via `pto.make_tensor_view`.
        arg_index: int | None = None
        for kw in call.keywords:
            if kw.arg in ("arg", "arg_index"):
                arg_index = int(self._eval_const(kw.value))

        if arg_index is None:
            arg_index = self._next_tensor_arg
            self._next_tensor_arg += 1

        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise FrontendError("tensor(...) currently expects shape=(H, W)")
        h, w = int(shape[0]), int(shape[1])

        if stride is None:
            s0, s1 = w, 1
        else:
            if not isinstance(stride, (tuple, list)) or len(stride) != 2:
                raise FrontendError("tensor(..., stride=...) expects stride=(S0, S1)")
            s0, s1 = int(stride[0]), int(stride[1])

        sym = self._sym_for(target)
        self._t.line(
            f"{sym.pto} = pto.make_tensor_view %arg{arg_index}, dtype={dtype}, "
            f"shape=[{h},{w}] strides=[{s0},{s1}], layout={layout}"
        )

    def _declare_tile(self, target: str, call: ast.Call) -> None:
        loc: str | None = None
        dtype: str | None = None
        rows: int | None = None
        cols: int | None = None

        blayout: str = "RowMajor"
        valid: str | None = None
        slayout: str = "NoneBox"
        fractal: int | None = None
        pad: str = "Null"
        addr: int | None = None

        args = list(call.args)
        if args:
            if len(args) >= 1:
                loc = self._eval_const(args[0])
            if len(args) >= 2:
                dtype = self._eval_const(args[1])
            if len(args) >= 3:
                rows = self._eval_const(args[2])
            if len(args) >= 4:
                cols = self._eval_const(args[3])

        for kw in call.keywords:
            if kw.arg == "loc":
                loc = self._eval_const(kw.value)
            elif kw.arg == "dtype":
                dtype = self._eval_const(kw.value)
            elif kw.arg == "rows":
                rows = self._eval_const(kw.value)
            elif kw.arg == "cols":
                cols = self._eval_const(kw.value)
            elif kw.arg == "blayout":
                blayout = self._eval_const(kw.value)
            elif kw.arg == "valid":
                valid = self._eval_const(kw.value)
            elif kw.arg == "slayout":
                slayout = self._eval_const(kw.value)
            elif kw.arg == "fractal":
                fractal = self._eval_const(kw.value)
            elif kw.arg == "pad":
                pad = self._eval_const(kw.value)
            elif kw.arg == "addr":
                addr = int(self._eval_const(kw.value))
            else:
                raise FrontendError(f"unknown tile(...) kw: {kw.arg}")

        if loc is None or dtype is None or rows is None or cols is None:
            raise FrontendError("tile(...) requires loc, dtype, rows, cols")

        sym = self._sym_for(target)
        if valid is not None:
            if isinstance(valid, str) and "x" in valid:
                vr, vc = valid.split("x", 1)
                valid_rows = int(vr)
                valid_cols = int(vc)
            else:
                raise FrontendError("tile(..., valid=...) must be like '16x16'")
        else:
            valid_rows = None
            valid_cols = None

        ty = TileType(
            loc=loc,
            dtype=dtype,
            rows=rows,
            cols=cols,
            blayout=blayout,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
            slayout=slayout,
            fractal=fractal,
            pad=pad,
        )
        if addr is None:
            self._t.line(f"{sym.pto} = pto.alloc_tile : {ty}")
        else:
            self._t.line(f"{sym.pto} = pto.alloc_tile {addr} : {ty}")

    def _emit_scalar_assign(self, dst: str, value: ast.AST) -> None:
        dst_sym = self._sym_for(dst)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            fn = value.func.id
            if fn == "get_block_idx":
                self._t.line(f"{dst_sym.pto} = pto.get_block_idx : index")
                return
            if fn == "get_block_num":
                self._t.line(f"{dst_sym.pto} = pto.get_block_num : index")
                return
        if isinstance(value, ast.BinOp):
            lhs = self._opnd(value.left)
            rhs = self._opnd(value.right)
            if isinstance(value.op, ast.Add):
                self._t.line(f"{dst_sym.pto} = pto.iadd {lhs}, {rhs} : index")
                return
            if isinstance(value.op, ast.Mult):
                self._t.line(f"{dst_sym.pto} = pto.imul {lhs}, {rhs} : index")
                return
        raise FrontendError(f"unsupported scalar assignment: {dst} = {ast.dump(value)}")

    def _emit_instr_stmt(self, call: ast.Call) -> None:
        if not isinstance(call.func, ast.Name):
            raise FrontendError(f"unsupported call form: {ast.dump(call)}")
        fn = call.func.id

        if fn in ("prologue", "epilogue"):
            self._t.line(fn)
            return

        def opnds() -> list[str]:
            return [self._opnd(a) for a in call.args]

        if fn == "tassign":
            raise FrontendError(
                "tassign(...) is not supported in the new PTO-AS syntax; "
                "use tile(..., addr=0x...) or omit `addr` and run ptoas with --assign-tile-addrs"
            )
        if fn == "tmov":
            a = opnds()
            if len(a) != 2:
                raise FrontendError("tmov(dst, src)")
            self._t.line(f"{a[0]} = pto.tmov {a[1]}")
            return
        if fn == "tadd":
            a = opnds()
            if len(a) != 3:
                raise FrontendError("tadd(dst, a, b)")
            self._t.line(f"{a[0]} = pto.tadd {a[1]}, {a[2]}")
            return
        if fn == "tmatmul":
            a = opnds()
            if len(a) != 3:
                raise FrontendError("tmatmul(dst, a, b)")
            self._t.line(f"{a[0]} = pto.tmatmul {a[1]}, {a[2]}")
            return

        if fn == "tload":
            if len(call.args) != 4:
                raise FrontendError("tload(dst_tile, src_tensor, r, c)")
            dst = self._opnd(call.args[0])
            src = self._opnd(call.args[1])
            r = self._opnd(call.args[2])
            c = self._opnd(call.args[3])
            self._t.line(f"{dst} = pto.tload {src}[{r}, {c}]")
            return
        if fn == "tstore":
            if len(call.args) != 4:
                raise FrontendError("tstore(dst_tensor, r, c, src_tile)")
            dst = self._opnd(call.args[0])
            r = self._opnd(call.args[1])
            c = self._opnd(call.args[2])
            src = self._opnd(call.args[3])
            self._t.line(f"pto.tstore {dst}[{r}, {c}], {src}")
            return

        raise FrontendError(f"unknown instruction call: {fn}")

    def _emit_if(self, stmt: ast.If) -> None:
        # Only support simple compare -> icmp_* -> scf.if.
        if not isinstance(stmt.test, ast.Compare) or len(stmt.test.ops) != 1 or len(stmt.test.comparators) != 1:
            raise FrontendError("if condition must be a simple compare, e.g. if a < b:")

        lhs = self._opnd(stmt.test.left)
        rhs = self._opnd(stmt.test.comparators[0])
        op = stmt.test.ops[0]
        mode = None
        if isinstance(op, ast.Eq):
            mode = "eq"
        elif isinstance(op, ast.NotEq):
            mode = "ne"
        elif isinstance(op, ast.Lt):
            mode = "lt"
        elif isinstance(op, ast.LtE):
            mode = "le"
        elif isinstance(op, ast.Gt):
            mode = "gt"
        elif isinstance(op, ast.GtE):
            mode = "ge"
        else:
            raise FrontendError("unsupported compare op")

        cond = self._tmp()
        self._t.line(f"{cond.pto} = pto.icmp_{mode} {lhs}, {rhs} : i1")
        self._t.open(f"scf.if {cond.pto}")
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

        iv = self._sym_for(stmt.target.id)
        lb = self._opnd(start)
        ub = self._opnd(stop)
        st = self._opnd(step)
        self._t.open(f"scf.for {iv.pto} = {lb} to {ub} step {st}")
        self._emit_stmts(stmt.body)
        self._t.close()

    def _emit_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise FrontendError("only simple assignments to a name are supported")
            dst = stmt.targets[0].id
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "tensor":
                self._declare_tensor(dst, stmt.value)
                return
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "tile":
                self._declare_tile(dst, stmt.value)
                return
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (int, bool, float)):
                # Inline numeric literals directly (new PTO-AS removes `.const`).
                if dst in self._sym:
                    raise FrontendError(f"cannot rebind existing SSA symbol as a literal: {dst}")
                if isinstance(stmt.value.value, bool):
                    self._literal[dst] = "1" if stmt.value.value else "0"
                else:
                    self._literal[dst] = str(stmt.value.value)
                return
            self._emit_scalar_assign(dst, stmt.value)
            return

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._emit_instr_stmt(stmt.value)
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

    def compile_funcdef(self, fn: ast.FunctionDef) -> str:
        self._emit_stmts(fn.body)
        return self._t.emit()


def compile_kernel_from_source(source: str, *, func_name: str) -> str:
    m = ast.parse(textwrap.dedent(source))
    fns = [n for n in m.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
    if not fns:
        raise FrontendError(f"function not found: {func_name}")
    if len(fns) != 1:
        raise FrontendError(f"ambiguous function: {func_name}")
    return _Compiler().compile_funcdef(fns[0])


def compile_kernel(func: Callable[..., Any]) -> str:
    src = inspect.getsource(func)
    return compile_kernel_from_source(src, func_name=func.__name__)


def make_add16_program() -> str:
    return compile_kernel_from_source(
        '''
def add16():
    prologue()
    bn = get_block_num()
    bid = get_block_idx()
    r0 = bid * 16

    x = tensor(dtype="f16", shape=(16, 16))
    y = tensor(dtype="f16", shape=(16, 16))
    z = tensor(dtype="f16", shape=(16, 16))
    tx = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    ty = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    tz = tile(loc="Vec", dtype="f16", rows=16, cols=16)

    tload(tx, x, r0, 0)
    tload(ty, y, r0, 0)
    tadd(tz, tx, ty)
    tstore(z, r0, 0, tz)
    epilogue()
''',
        func_name="add16",
    )


def make_gemm16_program() -> str:
    return compile_kernel_from_source(
        '''
def gemm16():
    prologue()
    bn = get_block_num()
    bid = get_block_idx()

    a = tensor(dtype="f16", shape=(16, 16))
    b = tensor(dtype="f16", shape=(16, 16))
    c = tensor(dtype="f32", shape=(16, 16))

    a_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    a_left = tile(loc="Left", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="RowMajor")
    b_right = tile(loc="Right", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="ColMajor")
    c_acc = tile(loc="Acc", dtype="f32", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    tload(a_mat, a, 0, 0)
    tload(b_mat, b, 0, 0)
    tmov(a_left, a_mat)
    tmov(b_right, b_mat)
    tmatmul(c_acc, a_left, b_right)
    tstore(c, 0, 0, c_acc)
    epilogue()
''',
        func_name="gemm16",
    )


def make_gemm16_cpu_program() -> str:
    # CPU simulator uses different matrix fractal constraints for TMATMUL.
    return compile_kernel_from_source(
        '''
def gemm16_cpu():
    prologue()
    bn = get_block_num()
    bid = get_block_idx()

    a = tensor(dtype="f16", shape=(16, 16))
    b = tensor(dtype="f16", shape=(16, 16))
    c = tensor(dtype="f32", shape=(16, 16))

    a_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    a_left = tile(loc="Left", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_right = tile(loc="Right", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="ColMajor")
    c_acc = tile(loc="Acc", dtype="f32", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    tload(a_mat, a, 0, 0)
    tload(b_mat, b, 0, 0)
    tmov(a_left, a_mat)
    tmov(b_right, b_mat)
    tmatmul(c_acc, a_left, b_right)
    tstore(c, 0, 0, c_acc)
    epilogue()
''',
        func_name="gemm16_cpu",
    )
