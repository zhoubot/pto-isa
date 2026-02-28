<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](../../LICENSE)

</div>

# PTO ISA Reference

This directory contains the **per-instruction reference** for the PTO Tile Library ISA. Each instruction is documented with its operation semantics, assembly syntax, C++ intrinsic interface, constraints, and usage examples.

## Quick Reference

| Resource | Description |
|----------|-------------|
| [Source of Truth (C++)](../../include/pto/common/pto_instr.hpp) | C++ intrinsic declarations |
| [Conventions](conventions.md) | Operand, event, and modifier notation |
| [ISA Overview](../PTOISA.md) | Quick reference table |

---

## Instruction Categories

### 1. Synchronization

Controls execution ordering and event management.

| Instruction | Description |
|-------------|-------------|
| [TSYNC](TSYNC.md) | Synchronize PTO execution (wait on events or insert a per-op pipeline barrier). |

---

### 2. Manual / Resource Binding

Explicit resource allocation and mode configuration for Manual mode.

| Instruction | Description |
|-------------|-------------|
| [TASSIGN](TASSIGN.md) | Bind a Tile object to an implementation-defined on-chip address (manual placement). |
| [TSETHF32MODE](TSETHF32MODE.md) | Configure HF32 transform mode (implementation-defined). |
| [TSETTF32MODE](TSETTF32MODE.md) | Configure TF32 transform mode (implementation-defined). |
| [TSETFMATRIX](TSETFMATRIX.md) | Set FMATRIX register(s) for IMG2COL-like ops. |
| [TSET_IMG2COL_RPT](TSET_IMG2COL_RPT.md) | Set IMG2COL repeat metadata from an IMG2COL configuration tile. |
| [TSET_IMG2COL_PADDING](TSET_IMG2COL_PADDING.md) | Set IMG2COL padding metadata from an IMG2COL configuration tile. |

---

### 3. Elementwise (Tile-Tile)

Binary and unary operations on tile operands.

| Instruction | Description |
|-------------|-------------|
| [TADD](TADD.md) | Elementwise add of two tiles. |
| [TSUB](TSUB.md) | Elementwise subtract of two tiles. |
| [TMUL](TMUL.md) | Elementwise multiply of two tiles. |
| [TDIV](TDIV.md) | Elementwise division of two tiles. |
| [TABS](TABS.md) | Elementwise absolute value of a tile. |
| [TNEG](TNEG.md) | Elementwise negation of a tile. |
| [TNOT](TNOT.md) | Elementwise bitwise NOT of a tile. |
| [TAND](TAND.md) | Elementwise bitwise AND of two tiles. |
| [TOR](TOR.md) | Elementwise bitwise OR of two tiles. |
| [TXOR](TXOR.md) | Elementwise bitwise XOR of two tiles. |
| [TSHL](TSHL.md) | Elementwise shift-left of two tiles. |
| [TSHR](TSHR.md) | Elementwise shift-right of two tiles. |
| [TMIN](TMIN.md) | Elementwise minimum of two tiles. |
| [TMAX](TMAX.md) | Elementwise maximum of two tiles. |
| [TCMP](TCMP.md) | Compare two tiles and write a packed predicate mask. |
| [TLOG](TLOG.md) | Elementwise natural logarithm of a tile. |
| [TEXP](TEXP.md) | Elementwise exponential. |
| [TSQRT](TSQRT.md) | Elementwise square root. |
| [TRSqrt](TRSqrt.md) | Elementwise reciprocal square root. |
| [TRECIP](TRECIP.md) | Elementwise reciprocal of a tile. |
| [TRELU](TRELU.md) | Elementwise ReLU of a tile. |
| [TLRELU](TLRELU.md) | Leaky ReLU with per-element slope. |
| [TPRELU](TPRELU.md) | Elementwise PReLU (parametric ReLU) with a per-element slope tile. |
| [TCVT](TCVT.md) | Elementwise type conversion with a specified rounding mode. |
| [TSEL](TSEL.md) | Select between two tiles using a mask tile (per-element selection). |
| [TREM](TREM.md) | Elementwise remainder of two tiles. |
| [TFMOD](TFMOD.md) | Elementwise fmod of two tiles. |
| [TADDC](TADDC.md) | Elementwise ternary add: `src0 + src1 + src2`. |
| [TSUBC](TSUBC.md) | Elementwise ternary op: `src0 - src1 + src2`. |

---

### 4. Tile-Scalar / Tile-Immediate

Operations between tile and scalar/immediate values.

| Instruction | Description |
|-------------|-------------|
| [TADDS](TADDS.md) | Elementwise add a scalar to a tile. |
| [TSUBS](TSUBS.md) | Elementwise subtract a scalar from a tile. |
| [TMULS](TMULS.md) | Elementwise multiply a tile by a scalar. |
| [TDIVS](TDIVS.md) | Elementwise division with a scalar (tile/scalar or scalar/tile). |
| [TMAXS](TMAXS.md) | Elementwise max of a tile and a scalar. |
| [TMINS](TMINS.md) | Elementwise min of a tile and a scalar. |
| [TANDS](TANDS.md) | Elementwise bitwise AND of a tile and a scalar. |
| [TORS](TORS.md) | Elementwise bitwise OR of a tile and a scalar. |
| [TXORS](TXORS.md) | Elementwise bitwise XOR of a tile and a scalar. |
| [TSHLS](TSHLS.md) | Elementwise shift-left a tile by a scalar. |
| [TSHRS](TSHRS.md) | Elementwise shift-right a tile by a scalar. |
| [TEXPANDS](TEXPANDS.md) | Broadcast a scalar into a destination tile. |
| [TCMPS](TCMPS.md) | Compare a tile against a scalar and write per-element comparison results. |
| [TSELS](TSELS.md) | Select one of two source tiles using a scalar `selectMode` (global select). |
| [TLRELU](TLRELU.md) | Leaky ReLU with a scalar slope. |
| [TADDSC](TADDSC.md) | Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`. |
| [TSUBSC](TSUBSC.md) | Elementwise fused op: `src0 - scalar + src1`. |
| [TREMS](TREMS.md) | Elementwise remainder with a scalar. |
| [TFMODS](TFMODS.md) | Elementwise fmod with a scalar. |

---

### 5. Axis Reduce

Reduction operations along row or column axes.

#### Row Reduction

| Instruction | Description |
|-------------|-------------|
| [TROWSUM](TROWSUM.md) | Reduce each row by summing across columns. |
| [TROWPROD](TROWPROD.md) | Reduce each row by multiplying across columns. |
| [TROWMAX](TROWMAX.md) | Reduce each row by taking the maximum across columns. |
| [TROWMIN](TROWMIN.md) | Reduce each row by taking the minimum across columns. |

#### Column Reduction

| Instruction | Description |
|-------------|-------------|
| [TCOLSUM](TCOLSUM.md) | Reduce each column by summing across rows. |
| [TCOLPROD](TCOLPROD.md) | Reduce each column by multiplying across rows. |
| [TCOLMAX](TCOLMAX.md) | Reduce each column by taking the maximum across rows. |
| [TCOLMIN](TCOLMIN.md) | Reduce each column by taking the minimum across rows. |

---

### 6. Axis Expand / Broadcast

Broadcast operations from scalar vector to full tile.

#### Row Expansion

| Instruction | Description |
|-------------|-------------|
| [TROWEXPAND](TROWEXPAND.md) | Broadcast the first element of each source row across the destination row. |
| [TROWEXPANDADD](TROWEXPANDADD.md) | Row-wise broadcast add: add a per-row scalar vector. |
| [TROWEXPANDSUB](TROWEXPANDSUB.md) | Row-wise broadcast subtract: subtract a per-row scalar vector from each row. |
| [TROWEXPANDMUL](TROWEXPANDMUL.md) | Row-wise broadcast multiply: multiply each row by a per-row scalar vector. |
| [TROWEXPANDDIV](TROWEXPANDDIV.md) | Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector. |
| [TROWEXPANDMAX](TROWEXPANDMAX.md) | Row-wise broadcast max with a per-row scalar vector. |
| [TROWEXPANDMIN](TROWEXPANDMIN.md) | Row-wise broadcast min with a per-row scalar vector. |
| [TROWEXPANDEXPDIF](TROWEXPANDEXPDIF.md) | Row-wise exp-diff: compute exp(src0 - src1) with per-row scalars. |

#### Column Expansion

| Instruction | Description |
|-------------|-------------|
| [TCOLEXPAND](TCOLEXPAND.md) | Broadcast the first element of each source column across the destination column. |
| [TCOLEXPANDADD](TCOLEXPANDADD.md) | Column-wise broadcast add with per-column scalar vector. |
| [TCOLEXPANDSUB](TCOLEXPANDSUB.md) | Column-wise broadcast subtract. |
| [TCOLEXPANDMUL](TCOLEXPANDMUL.md) | Column-wise broadcast multiply. |
| [TCOLEXPANDDIV](TCOLEXPANDDIV.md) | Column-wise broadcast divide. |
| [TCOLEXPANDMAX](TCOLEXPANDMAX.md) | Column-wise broadcast max. |
| [TCOLEXPANDMIN](TCOLEXPANDMIN.md) | Column-wise broadcast min. |
| [TCOLEXPANDEXPDIF](TCOLEXPANDEXPDIF.md) | Column-wise exp-diff. |

---

### 7. Memory Operations

Data movement between global memory (GM) and tile storage.

| Instruction | Description |
|-------------|-------------|
| [TLOAD](TLOAD.md) | Load data from a GlobalTensor (GM) into a Tile. |
| [TSTORE](TSTORE.md) | Store data from a Tile into a GlobalTensor (GM). |
| [TSTORE_FP](TSTORE_FP.md) | Store accumulator tile with vector quantization parameters. |
| [TPREFETCH](TPREFETCH.md) | Prefetch data from global memory into a tile-local cache/buffer. |

### Indexed Memory Operations

| Instruction | Description |
|-------------|-------------|
| [MGATHER](MGATHER.md) | Gather-load elements from GM into a tile using per-element indices. |
| [MSCATTER](MSCATTER.md) | Scatter-store elements from a tile into GM using per-element indices. |

---

### 8. Matrix Multiply

General matrix multiply and specialized variants.

| Instruction | Description |
|-------------|-------------|
| [TMATMUL](TMATMUL.md) | Matrix multiply (GEMM) producing an accumulator/output tile. |
| [TMATMUL_ACC](TMATMUL_ACC.md) | Matrix multiply with accumulator input (fused accumulate). |
| [TMATMUL_BIAS](TMATMUL_BIAS.md) | Matrix multiply with bias add. |
| [TMATMUL_MX](TMATMUL_MX.md) | Matrix multiply with scaling tiles for mixed-precision/quantized matmul. |
| [TGEMV](TGEMV.md) | General Matrix-Vector multiplication. |
| [TGEMV_ACC](TGEMV_ACC.md) | GEMV with explicit accumulator input/output. |
| [TGEMV_BIAS](TGEMV_BIAS.md) | GEMV with bias add. |
| [TGEMV_MX](TGEMV_MX.md) | GEMV with additional scaling tiles for mixed-precision. |

---

### 9. Data Movement / Layout

Tile transformation and layout operations.

| Instruction | Description |
|-------------|-------------|
| [TEXTRACT](TEXTRACT.md) | Extract a sub-tile from a source tile. |
| [TEXTRACT_FP](TEXTRACT_FP.md) | Extract with fp/scaling tile (vector quantization). |
| [TINSERT](TINSERT.md) | Insert a sub-tile into a destination tile at an offset. |
| [TINSERT_FP](TINSERT_FP.md) | Insert with fp/scaling tile. |
| [TFILLPAD](TFILLPAD.md) | Copy+pad a tile outside the valid region with a compile-time pad value. |
| [TFILLPAD_INPLACE](TFILLPAD_INPLACE.md) | In-place fill/pad variant. |
| [TFILLPAD_EXPAND](TFILLPAD_EXPAND.md) | Fill/pad while allowing dst to be larger than src. |
| [TMOV](TMOV.md) | Move/copy between tiles with optional conversion. |
| [TMOV_FP](TMOV_FP.md) | Move/convert from accumulator to destination with quantization. |
| [TRESHAPE](TRESHAPE.md) | Reinterpret tile as another type/shape. |
| [TTRANS](TTRANS.md) | Transpose with implementation-defined temporary tile. |
| [TIMG2COL](TIMG2COL.md) | Image-to-column transform for convolution. |

---

### 10. Complex / Specialized

Specialized operations and debugging tools.

| Instruction | Description |
|-------------|-------------|
| [TPRINT](TPRINT.md) | Debug/print elements from a tile. |
| [TMRGSORT](TMRGSORT.md) | Merge sort for multiple sorted lists. |
| [TSORT32](TSORT32.md) | Sort a fixed-size 32-element block with index mapping. |
| [TGATHER](TGATHER.md) | Gather/select elements using index tile or compile-time mask. |
| [TGATHERB](TGATHERB.md) | Gather elements using byte offsets. |
| [TSCATTER](TSCATTER.md) | Scatter rows of a source tile using per-element indices. |
| [TCI](TCI.md) | Generate contiguous integer sequence into a destination tile. |
| [TTRI](TTRI.md) | Generate triangular (lower/upper) mask tile. |
| [TPARTADD](TPARTADD.md) | Partial elementwise add with mismatched valid region handling. |
| [TPARTMUL](TPARTMUL.md) | Partial elementwise multiply with mismatched valid region handling. |
| [TPARTMAX](TPARTMAX.md) | Partial elementwise max with mismatched valid region handling. |
| [TPARTMIN](TPARTMIN.md) | Partial elementwise min with mismatched valid region handling. |
| [TQUANT](TQUANT.md) | Quantize a tile (e.g., FP32 to FP8) with exponent/scaling outputs. |

---

## Instruction Format

Each instruction document follows a standardized structure:

```text
# INSTRUCTION_NAME

## Tile Operation Diagram
[Visual representation of the operation]

## Math Interpretation
[Formal mathematical specification]

## Assembly Syntax
PTO-AS form: ...
MLIR SSA form: ...
MLIR DPS form: ...

## C++ Intrinsic
[Template signature]

## Constraints
- Implementation requirements
- Valid region behavior
- Type restrictions

## Examples
### Auto Mode
[Code example]

### Manual Mode
[Code example]
```

---

## Related Documentation

- [Programming Model](../coding/ProgrammingModel.md)
- [Tile API](../coding/Tile.md)
- [GlobalTensor API](../coding/GlobalTensor.md)
- [PTO Assembly Syntax](../grammar/PTO-AS.md)
- [Virtual ISA Manual](../mkdocs/src/manual/09-virtual-isa-and-ir.md)
