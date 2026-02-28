# TROWEXPANDMIN

## Introduction

Row-wise broadcast min.

Computes `min(src0[i,j], src1[i,0])` for each element.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`.

For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,0}) $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## Backend support

- CPU: **TODO**
- A2/A3: **Yes** (`include/pto/npu/a2a3/TRowExpandMin.hpp`)
- A5: **Yes** (`include/pto/npu/a5/TRowExpandMin.hpp`)

## Constraints (from implementation)

### A2/A3

- DType: `half` or `float`.
- Compile-time: `TileDataDst::isRowMajor` and at least one of `src0/src1` has the same Tile type as `dst`.
- Runtime (shape): `src1` constraints are the same as `TROWEXPANDADD`.

## Intrinsics (A2/A3)

- `vmin`
