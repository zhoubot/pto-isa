from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: Sequence[int]
    stride: Sequence[int] | None = None
    layout: str = "ND"

    def shape2(self) -> tuple[int, int]:
        shape = list(self.shape)
        if len(shape) != 2:
            raise ValueError("TensorType currently expects shape as 2D (H, W)")
        return int(shape[0]), int(shape[1])

    def stride2(self) -> tuple[int, int]:
        h, w = self.shape2()
        if self.stride is None:
            return w, 1
        stride = list(self.stride)
        if len(stride) != 2:
            raise ValueError("TensorType currently expects stride as 2D (S0, S1)")
        return int(stride[0]), int(stride[1])

    def __str__(self) -> str:
        # Preserve the canonical spelling used by the MLIR `ptoas` emitter.
        h, w = self.shape2()
        s0, s1 = self.stride2()
        return f"!pto.tensor<dtype={self.dtype}, shape=[{h},{w}], stride=[{s0},{s1}], layout={self.layout}>"


@dataclass(frozen=True)
class TileType:
    loc: str
    dtype: str
    rows: int
    cols: int
    blayout: str = "RowMajor"
    valid_rows: int | None = None
    valid_cols: int | None = None
    # Dynamic valid shape (recommended): emitted as `!pto.tile_buf<..., v_row=..., v_col=...>`.
    # Use the string "dyn" to represent `pto::DYNAMIC`.
    v_row: int | str | None = None
    v_col: int | str | None = None
    slayout: str = "NoneBox"
    fractal: int | None = None
    pad: str = "Null"

    def __str__(self) -> str:
        fractal = self.fractal
        if fractal is None:
            fractal = 1024 if self.loc == "Acc" else 512
        if self.v_row is not None or self.v_col is not None:
            v_row = self.rows if self.v_row is None else self.v_row
            v_col = self.cols if self.v_col is None else self.v_col
            return (
                f"!pto.tile_buf<loc={self.loc}, dtype={self.dtype}, rows={self.rows}, cols={self.cols}, "
                f"v_row={v_row}, v_col={v_col}, blayout={self.blayout}, slayout={self.slayout}, "
                f"fractal={fractal}, pad={self.pad}>"
            )

        valid_rows = self.rows if self.valid_rows is None else self.valid_rows
        valid_cols = self.cols if self.valid_cols is None else self.valid_cols
        return (
            f"!pto.tile<loc={self.loc}, dtype={self.dtype}, rows={self.rows}, cols={self.cols}, "
            f"blayout={self.blayout}, valid={valid_rows}x{valid_cols}, slayout={self.slayout}, "
            f"fractal={fractal}, pad={self.pad}>"
        )


class PTOProgram:
    """
    Low-level PTO-AS builder that emits the new surface syntax:
      - `%x = pto.make_tensor_view %arg0, ...`
      - `%t = pto.alloc_tile : !pto.tile<...>`
      - `%dst = pto.op %src0, %src1`

    Notes:
      - `.arg` / `.const` are intentionally not supported anymore (the MLIR `ptoas` frontend rejects them).
      - Numeric literals should be used directly (e.g. `16`, `0x10000`).
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._next_tensor_arg = 0

    def comment(self, text: str) -> "PTOProgram":
        for line in text.splitlines():
            self._lines.append(f"; {line}" if line else ";")
        return self

    def line(self, text: str) -> "PTOProgram":
        self._lines.append(text)
        return self

    def prologue(self) -> "PTOProgram":
        return self.line("prologue")

    def epilogue(self) -> "PTOProgram":
        return self.line("epilogue")

    def make_tensor_view(self, *, view: str, arg_index: int, ty: TensorType) -> "PTOProgram":
        if not view.startswith("%"):
            raise ValueError("view must start with %")
        h, w = ty.shape2()
        s0, s1 = ty.stride2()
        self._lines.append(
            f"{view} = pto.make_tensor_view %arg{arg_index}, dtype={ty.dtype}, "
            f"shape=[{h},{w}] strides=[{s0},{s1}], layout={ty.layout}"
        )
        return self

    def tensor_arg(self, name: str, ty: TensorType) -> "PTOProgram":
        # Convenience: sequentially bind tensors to %arg0, %arg1, ...
        idx = self._next_tensor_arg
        self._next_tensor_arg += 1
        return self.make_tensor_view(view=name, arg_index=idx, ty=ty)

    def alloc_tile(self, name: str, ty: TileType, addr: str | None = None) -> "PTOProgram":
        if not name.startswith("%"):
            raise ValueError("tile name must start with %")
        if addr is None:
            self._lines.append(f"{name} = pto.alloc_tile : {ty}")
        else:
            self._lines.append(f"{name} = pto.alloc_tile {addr} : {ty}")
        return self

    def assign(self, dst: str, opcode: str, operands: Iterable[str], typesig: str | None = None) -> "PTOProgram":
        if not dst.startswith("%"):
            raise ValueError("dst must start with %")
        op_text = f"{dst} = pto.{opcode} {', '.join(operands)}".rstrip()
        if typesig:
            op_text += f" : {typesig}"
        self._lines.append(op_text)
        return self

    def op(self, opcode: str, operands: Iterable[str], typesig: str | None = None) -> "PTOProgram":
        op_text = f"pto.{opcode} {', '.join(operands)}".rstrip()
        if typesig:
            op_text += f" : {typesig}"
        self._lines.append(op_text)
        return self

    def emit(self) -> str:
        return "\n".join(self._lines).strip() + "\n"


def make_add16_program() -> str:
    prog = PTOProgram()
    prog.comment("Generated by ptoas.python.pto_asm (new-format PTO-AS)\nVec add: z = x + y for one 16x16 tile.")
    prog.prologue()

    prog.assign("%bn", "get_block_num", [], "index")
    prog.assign("%bid", "get_block_idx", [], "index")

    t = TensorType(dtype="f16", shape=[16, 16])
    prog.tensor_arg("%x", t).tensor_arg("%y", t).tensor_arg("%z", t)

    tile = TileType(loc="Vec", dtype="f16", rows=16, cols=16)
    prog.alloc_tile("%tx", tile).alloc_tile("%ty", tile).alloc_tile("%tz", tile)

    prog.assign("%tx", "tload", ["%x[0, 0]"])
    prog.assign("%ty", "tload", ["%y[0, 0]"])
    prog.assign("%tz", "tadd", ["%tx", "%ty"])
    prog.op("tstore", ["%z[0, 0]", "%tz"])

    prog.epilogue()
    return prog.emit()


def make_gemm16_program(*, target: str = "npu") -> str:
    prog = PTOProgram()
    prog.comment("Generated by ptoas.python.pto_asm (new-format PTO-AS)\nCube GEMM: C = A @ B (16x16).")
    prog.prologue()

    a = TensorType(dtype="f16", shape=[16, 16])
    b = TensorType(dtype="f16", shape=[16, 16])
    c = TensorType(dtype="f32", shape=[16, 16])
    prog.tensor_arg("%a", a).tensor_arg("%b", b).tensor_arg("%c", c)

    a_mat = TileType("Mat", "f16", 16, 16, blayout="ColMajor", slayout="RowMajor")
    b_mat = TileType("Mat", "f16", 16, 16, blayout="ColMajor", slayout="RowMajor")
    a_left = TileType("Left", "f16", 16, 16, blayout="ColMajor", slayout="RowMajor")
    b_right = TileType("Right", "f16", 16, 16, blayout="RowMajor", slayout="ColMajor")
    c_acc = TileType("Acc", "f32", 16, 16, blayout="ColMajor", slayout="RowMajor")

    prog.alloc_tile("%a_mat", a_mat)
    prog.alloc_tile("%b_mat", b_mat)
    prog.alloc_tile("%a_left", a_left)
    prog.alloc_tile("%b_right", b_right)
    prog.alloc_tile("%c_acc", c_acc)

    prog.assign("%a_mat", "tload", ["%a[0, 0]"])
    prog.assign("%b_mat", "tload", ["%b[0, 0]"])
    prog.assign("%a_left", "tmov", ["%a_mat"])
    prog.assign("%b_right", "tmov", ["%b_mat"])
    prog.assign("%c_acc", "tmatmul", ["%a_left", "%b_right"])
    prog.op("tstore", ["%c[0, 0]", "%c_acc"])

    prog.epilogue()
    return prog.emit()
