# TSYNC

## Introduction

Synchronize PTO execution:

- `TSYNC(events...)` waits on a set of explicit event tokens.
- `TSYNC<Op>()` inserts a pipeline barrier for a single vector op class.

Many intrinsics in `include/pto/common/pto_instr.hpp` call `TSYNC(events...)` internally before issuing the instruction.

## Math Interpretation

Not applicable.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Event operand form:

```text
tsync %e0, %e1 : !pto.event<...>, !pto.event<...>
```

Single-op barrier form:

```text
tsync.op #pto.op<TADD>
```

## IR Syntax

### IR-level1 (SSA)

```mlir
// IR-level1: synchronization is inserted by the compiler (no user-facing op)
```

### IR-level2 (DPS)

```mlir
pto.barrier <PIPE_ALL>
// or event-based
pto.set_flag[<PIPE_SRC>, <PIPE_DST>, <EVENT_IDn>]
pto.wait_flag[<PIPE_SRC>, <PIPE_DST>, <EVENT_IDn>]
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <Op OpCode>
PTO_INST void TSYNC();

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents&... events);
```

## Constraints

- **Implementation checks (`TSYNC<Op>()`)**:
  - `TSYNC_IMPL<Op>()` only supports vector-pipeline ops (`static_assert(pipe == PIPE_V)` in `include/pto/common/event.hpp`).
- **`TSYNC(events...)` semantics**:
  - `TSYNC(events...)` calls `WaitAllEvents(events...)`, which invokes `events.Wait()` on each event token.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto(__gm__ float* in) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin(in);
  TileT t;
  Event<Op::TLOAD, Op::TADD> e;
  e = TLOAD(t, gin);
  TSYNC(e);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  Event<Op::TADD, Op::TSTORE_VEC> e;
  e = TADD(c, a, b);
  TSYNC<Op::TADD>();
  TSYNC(e);
}
```
