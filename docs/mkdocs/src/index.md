<!-- Project Logo -->
<p align="center">
  <img src="../../figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](../../../LICENSE)
[![Version](https://img.shields.io/badge/Version-9.0.0-green.svg)](../../../version.cmake)
[![Platform](https://img.shields.io/badge/Platform-Ascend%20A2%2FA3%2FA5-orange.svg)](../../../docs/getting-started.md)

</div>

---

# PTO Virtual ISA Manual

**Parallel Tile Operation (PTO)** is a virtual instruction set architecture designed by **Ascend CANN**, focusing on tile-level operations. This manual provides comprehensive documentation for the PTO ISA, programming model, and toolchain.

> **Note:** This is the authoritative reference for the PTO Virtual ISA. For quick navigation, see the table of contents below.

---

## Table of Contents

### 🚀 Getting Started

| Topic | Description |
|-------|-------------|
| [Home](index.md) | This page |
| [Overview](manual/01-overview.md) | Design goals, architectural identity |
| [Quick Start](manual/08-programming.md) | Write your first PTO program |
| [Setup Guide](../../../docs/getting-started.md) | Environment setup |

### 📖 Virtual ISA Manual

The complete architecture specification in 12 chapters:

1. [Preface](manual/index.md) - How to read this manual
2. [Execution Model](manual/02-machine-model.md) - Abstract machine
3. [State and Types](manual/03-state-and-types.md) - Data types
4. [Tiles and GlobalTensor](manual/04-tiles-and-globaltensor.md) - Core abstractions
5. [Synchronization](manual/05-synchronization.md) - Event ordering
6. [PTO Assembly](manual/06-assembly.md) - Assembly syntax
7. [Instructions](manual/07-instructions.md) - ISA overview
8. [Programming](manual/08-programming.md) - Development patterns
9. [Virtual ISA and IR](manual/09-virtual-isa-and-ir.md) - IR contract
10. [Bytecode and Toolchain](manual/10-bytecode-and-toolchain.md) - Compilation
11. [Memory Ordering](manual/11-memory-ordering-and-consistency.md) - Consistency
12. [Backend Profiles](manual/12-backend-profiles-and-conformance.md) - Platform conformance

### 💻 ISA Reference

Complete instruction set reference:

- [ISA Overview](../docs/isa/README.md) - All instructions by category
- [Conventions](../docs/isa/conventions.md) - Notation and syntax
- [Instruction Table](../docs/PTOISA.md) - Quick reference table

**Instruction Categories:**
- Synchronization: `TSYNC`
- Resource Binding: `TASSIGN`, `TSETHF32MODE`, `TSETTF32MODE`
- Elementwise: `TADD`, `TSUB`, `TMUL`, `TDIV`, `TCMP`, etc.
- Tile-Scalar: `TADDS`, `TCMPS`, `TEXPANDS`, etc.
- Reduction: `TROWSUM`, `TCOLSUM`, `TROWMAX`, `TCOLMAX`, etc.
- Memory: `TLOAD`, `TSTORE`, `MGATHER`, `MSCATTER`
- Matrix Multiply: `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS`, `TMATMUL_MX`
- Data Movement: `TEXTRACT`, `TINSERT`, `TMOV`, `TTRANS`

### 🔧 IR Reference

PTO intermediate representation:

- [IR Overview](../docs/ir/README.md) - All IR operations
- [Tile Operations](../docs/ir/PTO-IR-ops.md) - L1/L2 IR ops
- [Scalar Operations](../docs/ir/PTO-IR-scalar-arith-ops.md) - Arithmetic ops
- [Control Flow](../docs/ir/PTO-IR-control-flow-ops.md) - Flow control

### 📚 Programming Guide

Developer documentation:

- [Programming Model](../docs/coding/ProgrammingModel.md) - Core concepts
- [Tile API](../docs/coding/Tile.md) - Tile type system
- [GlobalTensor API](../docs/coding/GlobalTensor.md) - Memory views
- [Scalar Types](../docs/coding/Scalar.md) - Immediate values
- [Event API](../docs/coding/Event.md) - Synchronization
- [Optimization](../docs/coding/opt.md) - Performance tips
- [Debugging](../docs/coding/debug.md) - Troubleshooting

### 📝 Tutorials

Step-by-step guides:

- [Tutorial Index](../docs/coding/tutorials/README.md)
- [Vector Addition](../docs/coding/tutorials/vec-add.md) - Basic elementwise
- [Row Softmax](../docs/coding/tutorials/row-softmax.md) - Reduction + broadcast
- [GEMM](../docs/coding/tutorials/gemm.md) - Matrix multiplication

### ⚙️ Machine Model

Hardware abstraction:

- [Abstract Machine](../docs/machine/abstract-machine.md) - Core/device/host
- [Architecture](../docs/machine/README.md) - Hardware model

### 🔨 Assembly Reference

PTO-AS assembly language:

- [PTO-AS Specification](../docs/grammar/PTO-AS.md) - Syntax reference
- [BNF Grammar](../docs/grammar/PTO-AS.bnf) - Formal grammar
- [Conventions](../docs/grammar/conventions.md) - Notation

### 📎 Appendices

Additional reference material:

- [Glossary](manual/appendix-a-glossary.md) - Terminology
- [Instruction Template](manual/appendix-b-instruction-contract-template.md) - Adding new ops
- [Diagnostics](manual/appendix-c-diagnostics-taxonomy.md) - Error codes
- [Family Matrix](manual/appendix-d-instruction-family-matrix.md) - Instruction matrix

---

## Quick Examples

### Elementwise Addition

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void kernel_add() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src0, src1, dst;
    
    TADD(dst, src0, src1);
}
```

### Matrix Multiply

```cpp
void kernel_gemm() {
    using TileT = Tile<TileType::Mat, half, 16, 16>;
    TileT lhs, rhs, acc;
    
    // Load matrices
    TLOAD(lhs, gmem_lhs);
    TLOAD(rhs, gmem_rhs);
    
    // Matrix multiply
    TMATMUL(acc, lhs, rhs);
    
    // Store result
    TSTORE(gmem_out, acc);
}
```

---

## External Resources

| Resource | Description |
|----------|-------------|
| [GitHub](https://github.com/ascend/pto-isa) | Source code |
| [GitCode](https://gitcode.com/cann/pto-isa) | Mirror repository |
| [PTO-DSL](../../../PTODSL/README.md) | In-core level Python DSL (included in this repo) |
| [PyPTO](https://gitcode.com/cann/pypto/) | Formal Pythonic programming interface |
| [TileLang Ascend](https://github.com/tile-ai/tilelang-ascend/) | High-level DSL for Ascend NPUs |

---

## Version History

- **v9.0.0** (2025-12-27): Initial public release

---

*For the latest updates, see the [Release Notes](../../../ReleaseNote.md).*
