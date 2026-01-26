from __future__ import annotations

from dataclasses import dataclass

from compile.pto_compile_common import PTOProgram
from isa_definition.pto_isa_definition import (
    ElementType,
    ImmediateOperand,
    IndexOperand,
    TLOAD,
    TMATMUL,
    TSTORE,
)


def _dtype_str(dt: ElementType) -> str:
    return str(dt.value)


def _idx_str(x: ImmediateOperand | IndexOperand) -> str:
    if isinstance(x, ImmediateOperand):
        return str(x.value)
    if isinstance(x, IndexOperand):
        return f"%{x.name}"
    raise TypeError(f"unsupported index operand: {type(x).__name__}")


@dataclass(frozen=True)
class _TileBindings:
    # Map a logical PTO tile name (without leading %) to concrete PTO-AS tile symbols.
    #
    # For GEMM we lower:
    #   a (logical) -> %a_mat, %a_left
    #   b (logical) -> %b_mat, %b_right
    #   c (logical) -> %c_acc
    mat: dict[str, str]
    left: dict[str, str]
    right: dict[str, str]
    acc: dict[str, str]

    def tile_for_load(self, name: str) -> str:
        # Prefer loading matmul operands into Mat tiles (then TMOV -> Left/Right).
        if name in self.mat:
            return self.mat[name]
        return f"%{name}"

    def tile_for_matmul_a(self, name: str) -> tuple[str, str]:
        # Returns (src_mat, dst_left)
        return (self.mat[name], self.left[name])

    def tile_for_matmul_b(self, name: str) -> tuple[str, str]:
        return (self.mat[name], self.right[name])

    def tile_for_store(self, name: str) -> str:
        if name in self.acc:
            return self.acc[name]
        return f"%{name}"

    def tile_for_matmul_dst(self, name: str) -> str:
        return self.acc[name]


def export_program_to_ptoas_gemm16(*, program: PTOProgram, block_dim: int = 1, kernel_name: str = "pto_kernel") -> str:
    """
    Export a small GEMM-like PTOProgram (built via the *old* PTOFunctionBuilder) into the
    *new* PTO-AS text format accepted by `ptoas`.

    Scope (fast-path):
    - Supports straight-line programs containing only: TLOAD, TMATMUL, TSTORE.
    - Lowers matmul operands via Mat->Left/Right (TMOV) implicitly, matching `ptoas/examples/gemm16_e2e.pto`.
    """
    # Import locally to keep `compile/` usable without the ptoas toolchain for other paths.
    from ptoas.python.host_spec import HostSpec, HostTensorArg, prepend_host_spec_to_pto
    from ptoas.python.pto_asm import PTOProgram as PTOASProgram
    from ptoas.python.pto_asm import TensorType, TileType

    # --- Infer arg order + shapes (required for host spec and make_tensor_view). ---
    memrefs = list(program.memref_declarations.items())
    if not memrefs:
        raise ValueError("program has no memref declarations")

    # Mark memrefs written by TSTORE as outputs.
    out_memrefs: set[str] = set()
    for ins in program.instructions:
        if isinstance(ins, TSTORE):
            out_memrefs.add(ins.dst_mem.name)

    host_args: list[HostTensorArg] = []
    for name, mty in memrefs:
        if mty.shape is None:
            raise ValueError(f"memref {name} missing shape; pass shape=(H,W) in PTOFunctionBuilder.memref(...)")
        host_args.append(
            HostTensorArg(
                dtype=_dtype_str(mty.element_type),
                shape=(int(mty.shape.rows), int(mty.shape.cols)),
                role=("out" if name in out_memrefs else "in"),
            )
        )

    spec = HostSpec(args=tuple(host_args), seed=0, block_dim=int(block_dim), kernel_name=str(kernel_name))

    # --- Identify matmul tiles (to pick TileType::Mat/Left/Right/Acc). ---
    matmul_a: set[str] = set()
    matmul_b: set[str] = set()
    matmul_dst: set[str] = set()
    for ins in program.instructions:
        if isinstance(ins, TMATMUL):
            matmul_a.add(ins.a.name)
            matmul_b.add(ins.b.name)
            matmul_dst.add(ins.dst.name)

    # Bind logical tiles to concrete PTO-AS tiles.
    mat: dict[str, str] = {}
    left: dict[str, str] = {}
    right: dict[str, str] = {}
    acc: dict[str, str] = {}
    for tname, tty in program.tile_declarations.items():
        if tname in matmul_a:
            mat[tname] = f"%{tname}_mat"
            left[tname] = f"%{tname}_left"
        if tname in matmul_b:
            mat[tname] = f"%{tname}_mat"
            right[tname] = f"%{tname}_right"
        if tname in matmul_dst:
            acc[tname] = f"%{tname}_acc"

    tb = _TileBindings(mat=mat, left=left, right=right, acc=acc)

    # --- Emit PTO-AS ---
    out = PTOASProgram()
    out.comment(f"Generated from old PTOFunctionBuilder program: {program.name}")
    out.prologue()

    # Tensor views in arg order.
    for arg_i, (name, mty) in enumerate(memrefs):
        assert mty.shape is not None
        out.make_tensor_view(
            view=f"%{name}",
            arg_index=int(arg_i),
            ty=TensorType(dtype=_dtype_str(mty.element_type), shape=(int(mty.shape.rows), int(mty.shape.cols))),
        )

    # Tile allocations.
    for tname, tty in program.tile_declarations.items():
        rows = int(tty.shape.rows)
        cols = int(tty.shape.cols)
        dt = _dtype_str(tty.element_type)

        if tname in matmul_a or tname in matmul_b:
            # Load into Mat, then convert to Left/Right.
            out.alloc_tile(tb.mat[tname], TileType(loc="Mat", dtype=dt, rows=rows, cols=cols, blayout="ColMajor", slayout="RowMajor"))
            if tname in matmul_a:
                out.alloc_tile(tb.left[tname], TileType(loc="Left", dtype=dt, rows=rows, cols=cols, blayout="ColMajor", slayout="RowMajor"))
            if tname in matmul_b:
                out.alloc_tile(tb.right[tname], TileType(loc="Right", dtype=dt, rows=rows, cols=cols, blayout="RowMajor", slayout="ColMajor"))
            continue

        if tname in matmul_dst:
            out.alloc_tile(tb.acc[tname], TileType(loc="Acc", dtype=dt, rows=rows, cols=cols, blayout="ColMajor", slayout="RowMajor"))
            continue

        out.alloc_tile(f"%{tname}", TileType(loc="Vec", dtype=dt, rows=rows, cols=cols, blayout="RowMajor", slayout="NoneBox"))

    # Instructions.
    for ins in program.instructions:
        if isinstance(ins, TLOAD):
            dst = tb.tile_for_load(ins.dst.name)
            src = f"%{ins.src_mem.name}"
            r0 = _idx_str(ins.row_offset)
            c0 = _idx_str(ins.col_offset)
            out.assign(dst, "tload", [f"{src}[{r0}, {c0}]"])
            continue

        if isinstance(ins, TMATMUL):
            # Lower: Mat -> Left/Right via TMOV, then TMATMUL into Acc.
            a_mat, a_left = tb.tile_for_matmul_a(ins.a.name)
            b_mat, b_right = tb.tile_for_matmul_b(ins.b.name)
            out.assign(a_left, "tmov", [a_mat])
            out.assign(b_right, "tmov", [b_mat])
            out.assign(tb.tile_for_matmul_dst(ins.dst.name), "tmatmul", [a_left, b_right])
            continue

        if isinstance(ins, TSTORE):
            src = tb.tile_for_store(ins.src.name)
            dst = f"%{ins.dst_mem.name}"
            r0 = _idx_str(ins.row_offset)
            c0 = _idx_str(ins.col_offset)
            out.op("tstore", [f"{dst}[{r0}, {c0}]", src])
            continue

        raise NotImplementedError(f"export_program_to_ptoas_gemm16: unsupported instruction: {type(ins).__name__}")

    out.epilogue()
    return prepend_host_spec_to_pto(pto=out.emit(), spec=spec)
