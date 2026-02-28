# TFILLPAD_INPLACE

## Introduction

Pad `dst` in-place according to `src`’s valid region and `dst`’s pad value.

Compared with `TFILLPAD`, the in-place form **does not** perform the “copy src into dst” step.

## Math Interpretation

Let:

- `Rdst = dst.GetValidRow()`, `Cdst = dst.GetValidCol()`
- `Rsrc = src.GetValidRow()`, `Csrc = src.GetValidCol()`

For `0 <= i < Rdst` and `0 <= j < Cdst`:

- if `i < Rsrc` and `j < Csrc`: keep `dst[i,j]` unchanged
- else: set `dst[i,j] = PadValue(TileDataDst::PadVal)`

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData& dst, SrcTileData& src, WaitEvents&... events);
```

## Backend support

- CPU: **Yes** (CPU helper exists; strict naming/impl wiring will be verified)
- A2/A3: **Yes** (`include/pto/npu/a2a3/TFillPad.hpp`)
- A5: **Yes** (`include/pto/npu/a5/TFillPad.hpp`)

## Constraints (from implementation)

- `TileDataDst::PadVal != PadValue::Null`.
- `sizeof(src.DType) == sizeof(dst.DType)`.
- element size must be 1/2/4 bytes.

### A2/A3 and A5

- Tile compile-time shape (in-place): `TileDataDst::Cols == TileDataSrc::Cols` and `TileDataDst::Rows == TileDataSrc::Rows`.
- If `dst.GetValidRow()==0` or `dst.GetValidCol()==0`, the implementation returns early.

## Intrinsics (NPU)

- Uses `vector_dup` / `vcopy` and barriers (`dsb`) in the NPU backend.
