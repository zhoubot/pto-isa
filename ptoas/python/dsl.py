from __future__ import annotations

"""
Python DSL stubs for writing PTO kernels with normal Python syntax.

These are intentionally NOT executable. The AST frontend parses your source code
and translates calls like `tensor(...)`, `tile(...)`, `tload(...)`, etc into PTO-AS.

Recommended usage in kernel files:

  from ptoas.python.dsl import *

  def add16():
      prologue()
      x = tensor(dtype="f16", shape=(16, 16))
      ...
      epilogue()
"""

from dataclasses import dataclass
from typing import Any, Literal, overload


class _NotExecutableError(RuntimeError):
    pass


def _noexec() -> None:
    raise _NotExecutableError("ptoas python DSL is not executable; compile it with ptoas/python/binding.py")


@dataclass(frozen=True)
class Tensor:
    pass


@dataclass(frozen=True)
class Tile:
    pass


def prologue() -> None:
    _noexec()


def epilogue() -> None:
    _noexec()


def get_block_idx() -> int:
    _noexec()


def get_block_num() -> int:
    _noexec()


@overload
def tensor(
    dtype: str,
    shape: tuple[int, int],
    stride: tuple[int, int] | None = None,
    layout: str = "ND",
    *,
    arg: int | None = None,
    arg_index: int | None = None,
    role: Literal["in", "out", "inout"] | None = None,
) -> Tensor: ...


def tensor(*args: Any, **kwargs: Any) -> Tensor:
    _noexec()


def tile(
    loc: Literal["Vec", "Mat", "Left", "Right", "Acc"],
    dtype: str,
    rows: int,
    cols: int,
    *,
    blayout: str = "RowMajor",
    valid: str | None = None,
    slayout: str = "NoneBox",
    fractal: int | None = None,
    pad: str = "Null",
    addr: int | None = None,
) -> Tile:
    _noexec()


def tmov(dst: Tile, src: Tile) -> None:
    _noexec()


def tadd(dst: Tile, a: Tile, b: Tile) -> None:
    _noexec()


def tmatmul(dst: Tile, a: Tile, b: Tile) -> None:
    _noexec()


def tmatmul_acc(dst: Tile, acc: Tile, a: Tile, b: Tile) -> None:
    _noexec()


def tload(dst: Tile, src: Tensor, r: int, c: int) -> None:
    _noexec()


def tstore(dst: Tensor, r: int, c: int, src: Tile) -> None:
    _noexec()


def record_event(*, src_op: str, dst_op: str, token: int) -> None:
    _noexec()


def wait_event(*, src_op: str, dst_op: str, token: int) -> None:
    _noexec()
