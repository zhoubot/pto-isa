# TCOLEXPANDMUL

## Introduction

Column-wise broadcast multiply.

This op multiplies `src0` by a per-column factor taken from the **first row** of `src1`.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`.

For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \times \mathrm{src1}_{0,j} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMUL(TileDataDst& dst, TileDataDst& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## Backend support

- CPU: **TODO**
- A2/A3: **TODO**
- A5: **Yes** (`include/pto/npu/a5/TColExpandMul.hpp`)

## Constraints (from implementation)

### A5

- DType: `half` or `float`.
- Layout: `TileDataDst::isRowMajor`.

### Shape / boundary conditions (semantic requirements)

For correctness:
- `src0` valid shape equals `dst` valid shape.
- `src1`’s first row has at least `dst.GetValidCol()` valid elements.

## Intrinsics (A5)

- Vector math: `vmul`
- Loads/stores: `vlds`, `vsts`
