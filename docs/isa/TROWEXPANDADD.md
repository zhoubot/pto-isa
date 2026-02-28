# TROWEXPANDADD

## Introduction

Row-wise broadcast add.

Adds a per-row scalar/broadcast vector (`src1`) to each row of `src0`.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`.

For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,0} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## Backend support

- CPU: **TODO**
- A2/A3: **Yes** (`include/pto/npu/a2a3/TRowExpandAdd.hpp`)
- A5: **Yes** (`include/pto/npu/a5/TRowExpandAdd.hpp`)

## Constraints (from implementation)

### A2/A3

- DType: `half` or `float`.
- Compile-time: `TileDataDst::isRowMajor` and at least one of `src0/src1` has the same Tile type as `dst`.
- Runtime (shape):
  - `src1.GetValidRow() == dst.GetValidRow()`
  - `src1` must be either:
    - row-major with `src1.GetValidCol() == 32 / sizeof(T)` (one block per row), or
    - non-row-major with `src1.GetValidCol() == 1`.

## Intrinsics (A2/A3)

- `vadd` (via row-expand helper)
