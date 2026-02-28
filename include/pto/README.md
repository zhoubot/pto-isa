<p align="center">
  <img src="../docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](../LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-yellow.svg)](../version.cmake)

</div>

# PTO C++ API Reference

This directory contains the **public C++ header files** for PTO Tile Library. It provides the Tile type system, PTO instruction API declarations, CPU simulation support, and NPU instruction implementations.

---

## Quick Start

### Recommended Include

```cpp
#include <pto/pto-inst.hpp>
```

This unified entry header automatically selects the appropriate backend based on build configuration:
- **CPU simulation**: Includes stubs when `__CPU_SIM` is defined
- **NPU**: Includes target-specific implementations

---

## Directory Layout

```
include/pto/
├── pto-inst.hpp           # Unified entry point (recommended)
├── pto.hpp                # Legacy entry point
├── common/                # Platform-independent infrastructure
│   ├── pto_tile.hpp       # Core Tile types and layout
│   ├── pto_instr.hpp      # Instruction declarations
│   ├── pto_instr_impl.hpp # Shared instruction implementations
│   ├── memory.hpp         # Memory utilities
│   ├── constants.hpp      # Constants and enumerations
│   ├── utils.hpp          # Utility functions
│   └── type.hpp           # Type definitions
├── cpu/                   # CPU simulation/stub support
│   └── cpu_stub.hpp       # CPU implementation stubs
└── npu/                  # NPU implementations (split by SoC)
    ├── a2a3/             # Ascend A2/A3 (910B/910C)
    └── a5/               # Ascend A5 (950)
```

---

## Core Components

### Tile Type System

The Tile is the fundamental compute unit in PTO:

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Define a 16x16 float tile
using TileT = Tile<TileType::Vec, float, 16, 16>;
```

See [Tile API](../docs/coding/Tile.md) for detailed documentation.

### Instruction API

Each PTO instruction is available as a C++ template function:

```cpp
// Elementwise addition
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events);

// Matrix multiply
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileData& dst, TileData& lhs, TileData& rhs, WaitEvents&... events);

// Load from global memory
template <typename TileData, typename GlobalTensor, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& tile, const GlobalTensor& gmem, WaitEvents&... events);
```

### Programming Modes

#### Auto Mode

In Auto mode, the compiler/runtime manages resource allocation and synchronization:

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void kernel_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  
  // Runtime handles placement and synchronization
  TADD(dst, src0, src1);
  TSTORE(gmem_out, dst);
}
```

#### Manual Mode

In Manual mode, you control placement and synchronization explicitly:

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void kernel_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  
  // Explicit resource binding
  TASSIGN(src0, 0x1000);  // Bind to address 0x1000
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  
  // Explicit synchronization
  Event e;
  TLOAD(src0, gmem_in0, e);
  TLOAD(src1, gmem_in1, e);
  TSYNC(e);
  
  // Compute
  TADD(dst, src0, src1);
  
  // Store
  TSTORE(gmem_out, dst);
}
```

---

## API Reference by Category

### Core Types

| Header | Description |
|--------|-------------|
| `pto_tile.hpp` | Tile, TileType, TileData definitions |
| `type.hpp` | Data type definitions (DType) |
| `constants.hpp` | Constants and enumerations |

### Instructions

| Header | Description |
|--------|-------------|
| `pto_instr.hpp` | All instruction declarations |
| `pto_instr_impl.hpp` | Shared implementations |

### Memory

| Header | Description |
|--------|-------------|
| `memory.hpp` | GlobalTensor, memory operations |
| `constants.hpp` | Memory layout constants |

### Utilities

| Header | Description |
|--------|-------------|
| `utils.hpp` | Helper utilities |
| `event.hpp` | Event management |

---

## Build Configuration

### CPU Simulation

Define `__CPU_SIM` to use CPU stubs:

```bash
cmake -DCMAKE_CXX_FLAGS="-D__CPU_SIM" ...
```

### NPU Backend

Target-specific builds:

```bash
# Ascend A2/A3
cmake -DPO_ISA_BACKEND=a2a3 ...

# Ascend A5
cmake -DPO_ISA_BACKEND=a5 ...
```

---

## Source of Truth

| Resource | Description |
|----------|-------------|
| [Instruction Reference](../docs/isa/README.md) | Per-instruction documentation |
| [Programming Guide](../docs/coding/ProgrammingModel.md) | Programming concepts |
| [C++ Intrinsics](common/pto_instr.hpp) | Implementation |

---

## Platform Support

| Platform | Header Path | Status |
|----------|-------------|--------|
| CPU (simulation) | `pto/cpu/` | ✅ Stable |
| Ascend A2 (910B) | `pto/npu/a2a3/` | ✅ Stable |
| Ascend A3 (910C) | `pto/npu/a2a3/` | ✅ Stable |
| Ascend A5 (950) | `pto/npu/a5/` | ✅ Stable |

---

## Related Documentation

- [Getting Started](../docs/getting-started.md) - Setup guide
- [Programming Model](../docs/coding/ProgrammingModel.md) - Core concepts
- [Tile Tutorial](../docs/coding/tutorial.md) - Writing your first kernel
- [ISA Manual](../docs/PTO-Virtual-ISA-Manual.md) - Architecture specification
