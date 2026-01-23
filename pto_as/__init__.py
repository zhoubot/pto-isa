from __future__ import annotations

"""
`pto_as` — small, readable Python frontend for writing PTO-AS kernels.

This module is intentionally lightweight:

- The *primary* compilation path in this repo parses Python source via `ast`:
  `ptoas/python/ast_frontend.py` → emits PTO-AS text.
- This runtime builder exists for ergonomics and IDE friendliness: you can
  execute `build()` to produce PTO-AS text without string templates.

The textual PTO-AS emitted here is compatible with the existing toolchain:
`ptoas/tools/python_to_pto.py` / `ptoas/tools/python_kernel_flow.py`.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ptoas.python.pto_asm import PTOProgram, TensorType, TileType


@dataclass(frozen=True)
class ScalarType:
    dtype: str


def scalar(dtype: str) -> ScalarType:
    return ScalarType(dtype=dtype)


@dataclass(frozen=True)
class Value:
    ref: str

    def __str__(self) -> str:
        return self.ref


Tensor = Value
Tile = Value


class PTO:
    def __init__(self, name: str) -> None:
        self.name = name
        self._p = PTOProgram()
        self._next_arg = 0

    # --- High-level structure ---

    def comment(self, text: str) -> None:
        self._p.comment(text)

    def prologue(self) -> None:
        self._p.prologue()

    def epilogue(self) -> None:
        self._p.epilogue()

    def program(self) -> PTOProgram:
        return self._p

    def emit(self) -> str:
        return self._p.emit()

    # --- Declarations ---

    def tensor(
        self,
        name: str,
        shape: tuple[int, int],
        *,
        dtype: str,
        stride: tuple[int, int] | None = None,
        layout: str = "ND",
        arg: int | None = None,
        role: str | None = None,
    ) -> Tensor:
        # `role` is host-metadata only; stored externally by the AST frontend. Kept here for API parity.
        _ = role
        view = f"%{name}"
        ty = TensorType(dtype=dtype, shape=shape, stride=stride, layout=layout)
        if arg is None:
            arg = self._next_arg
            self._next_arg += 1
        self._p.make_tensor_view(view=view, arg_index=arg, ty=ty)
        return Tensor(ref=view)

    def _tile(
        self,
        name: str,
        *,
        loc: str,
        dtype: str,
        shape: tuple[int, int],
        blayout: str,
        slayout: str,
        valid: str | None = None,
        fractal: int | None = None,
        pad: str = "Null",
        addr: int | None = None,
        b: str | None = None,
    ) -> Tile:
        # `b` is a convenience annotation in some examples (broadcast axis); not encoded in PTO-AS tile types.
        _ = b
        rows, cols = int(shape[0]), int(shape[1])
        valid_rows = None
        valid_cols = None
        if valid is not None:
            vr, vc = valid.split("x", 1)
            valid_rows = int(vr)
            valid_cols = int(vc)
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
        ref = f"%{name}"
        self._p.alloc_tile(ref, ty, addr=str(addr) if addr is not None else None)
        return Tile(ref=ref)

    def vec_tile(self, name: str, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Vec", dtype=dtype, shape=shape, blayout="RowMajor", slayout="NoneBox", **kw)

    def mat_tile(self, name: str, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Mat", dtype=dtype, shape=shape, blayout="ColMajor", slayout="RowMajor", **kw)

    def left_tile(self, name: str, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Left", dtype=dtype, shape=shape, blayout="RowMajor", slayout="RowMajor", **kw)

    def right_tile(self, name: str, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Right", dtype=dtype, shape=shape, blayout="RowMajor", slayout="ColMajor", **kw)

    def acc_tile(self, name: str, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Acc", dtype=dtype, shape=shape, blayout="ColMajor", slayout="RowMajor", **kw)

    # --- Constants ---

    def const(self, name: str, value: Any, ty: ScalarType) -> Any:
        # PTO-AS uses immediates for scalars today. Keep the API for readability.
        _ = name, ty
        return value

    # --- Memory ops ---

    def tload(self, dst: Tile, src: Tensor, r: int = 0, c: int = 0) -> Tile:
        self._p.assign(dst.ref, "tload", [f"{src.ref}[{r}, {c}]"])
        return dst

    def tstore(self, dst: Tensor, src: Tile, r: int = 0, c: int = 0) -> None:
        self._p.op("tstore", [f"{dst.ref}[{r}, {c}]", src.ref])

    # --- Generic instruction helpers ---

    def _fmt(self, x: Any) -> str:
        if isinstance(x, Value):
            return x.ref
        if isinstance(x, bool):
            return "1" if x else "0"
        if isinstance(x, float):
            return repr(x)
        if isinstance(x, int):
            return str(x)
        raise TypeError(f"unsupported operand: {type(x).__name__}")

    def _emit_dst_first(self, opcode: str, operands: Iterable[Any]) -> Any:
        ops = list(operands)
        if not ops:
            self._p.op(opcode, [])
            return None
        dst = ops[0]
        rest = ops[1:]
        if not isinstance(dst, Value):
            raise TypeError(f"{opcode} expects first arg as a Tile/Value (dst)")
        self._p.assign(dst.ref, opcode, [self._fmt(o) for o in rest])
        return dst

    def __getattr__(self, opcode: str) -> Callable[..., Any]:
        # Allow `pto.trowmax(dst, a, b)` without having to define every method.
        def _op(*operands: Any) -> Any:
            return self._emit_dst_first(opcode, operands)

        return _op

