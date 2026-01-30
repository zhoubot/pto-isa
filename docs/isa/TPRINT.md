# TPRINT

## Introduction

Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code.

The `TPRINT` instruction outputs the logical view of data stored in a Tile or GlobalTensor. It supports common data types (e.g., `float`, `half`, `int8`, `uint32`) and multiple memory layouts (`ND`, `DN`, `NZ` for GlobalTensor; vector tiles for on-chip buffers).

> **Important**:
> - This instruction is **for development and debugging ONLY**.
> - It incurs **significant runtime overhead** and **must not be used in production kernels**.
> - Output may be **truncated** if it exceeds the internal print buffer.
> - **Requires CCE compilation option `-D_DEBUG --cce-enable-print`** (see [Behavior](#behavior)).

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

```text
tprint %src : !pto.tile<...> | !pto.global<...>
```

## IR Syntax

### IR-level1 (SSA)

```mlir
// Not supported in IR-level1 (frontend lowers to a comment)
```

### IR-level2 (DPS)

```mlir
// Not supported in IR-level2 (frontend lowers to a comment)
```

## C++ Intrinsic
Declared in `include/pto/common/pto_instr.hpp`:
```cpp
template <typename T, typename... WaitEvents>
PTO_INST RecordEvent TPRINT(T &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TPRINT, src);
  return {};
}
```

### Supported Types for T
- **Tile**: Must be a vector tile (`TileType::Vec`) with supported element type.
- **GlobalTensor**: Must use layout `ND`, `DN`, or `NZ`, and have a supported element type.

## Constraints

- **Supported element type**:
  - Floating-point: `float`, `half`
  - Signed integers: `int8_t`, `int16_t`, `int32_t`
  - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`
- **For Tiles**: `TileData::Loc == TileType::Vec` (only vector tiles are printable).
- **For GlobalTensor**: Layout must be one of `Layout::ND`, `Layout::DN`, or `Layout::NZ`.

## Behavior
- **Mandatory Compilation Flag**:

  On A2/A3/A5 devices, `TPRINT` uses `cce::printf` to emit output via the device-to-host debug channel. **You must enable the CCE option `-D_DEBUG --cce-enable-print`**.

- **Buffer Limitation:**

  The internal print buffer of `cce::printf` is limited in size. If the output exceeds this buffer, a warning message such as `"Warning: out of bound! try best to print"` may appear, and **only partial data will be printed**.

- **Synchronization**:

  Automatically inserts a `pipe_barrier(PIPE_ALL)` before printing to ensure all prior operations complete and data is consistent.

- **Formatting**:

  - Floating-point values: printed as `%6.2f`
  - Integer values: printed as `%6d`
  - For `GlobalTensor`, due to data size and buffer limitations, only elements within its logical shape (defined by `Shape`) are printed.
  - For `Tile`, invalid regions (beyond `validRows`/`validCols`) are still printed but marked with a `|` separator when partial validity is specified.

## Examples

### Print a Tile

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugTile(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  using srcTileData = Tile<TileType::Vec, float, 16, 16>;
  srcTileData srcTile;
  TASSIGN(srcTile, 0x0);

  TLOAD(srcTile, srcGlobal);
  TPRINT(srcTile);
}
```

### Print a GlobalTensor

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugGlobalTensor(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  TPRINT(srcGlobal);
}
```