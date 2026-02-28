<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](../LICENSE)
[![Version](https://img.shields.io/badge/Version-9.0.0-green.svg)](../version.cmake)

</div>

# PTO ISA Guide

This directory contains the authoritative documentation for the **PTO (Parallel Tile Operation) Virtual Instruction Set Architecture** used by PTO Tile Library. This guide explains instruction naming conventions, common notation, and how to navigate the per-instruction reference pages.

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [🏠 Main README](../README.md) | Project overview and quick start |
| [📖 Virtual ISA Manual](PTO-Virtual-ISA-Manual.md) | Complete manual with 12 chapters |
| [🚀 Getting Started](getting-started.md) | Setup guide for CPU simulation and NPU |
| [💻 PTODSL](../PTODSL/README.md) | Pythonic programming interface |

---

## Naming Conventions

### Fundamental Types

| Type | Description | Example |
|------|-------------|---------|
| **Tile** | Fundamental data type for small 2D tensors | `MatTile`, `LeftTile`, `RightTile`, `VecTile` |
| **GlobalTensor** | A tensor stored in global memory (GM) | Used with `TLOAD`/`TSTORE` |
| **Scalar** | Immediate values and enumerations | `%R` register, comparison modes |
| **Event** | Explicit dependency tokens | Used for pipeline synchronization |

### Instruction Prefixes

| Prefix | Meaning | Example |
|--------|---------|---------|
| `T` | Tile operation (fundamental) | `TADD`, `TMATMUL`, `TLOAD` |
| `M` | Memory operation (complex) | `MGATHER`, `MSCATTER` |

### Notation

- **`%R`**: A scalar immediate register
- **Shape and alignment**: Enforced by compile-time constraints and runtime assertions
- **Valid region**: Defined by `Rv` (valid rows) and `Cv` (valid columns)

---

## Documentation Structure

### 1. Virtual ISA Manual (Chapters)

The complete manual is organized into 12 chapters plus appendices:

| Chapter | Title | Description |
|---------|-------|-------------|
| 1 | [Overview](mkdocs/src/manual/01-overview.md) | Design goals, architectural identity |
| 2 | [Machine Model](mkdocs/src/manual/02-machine-model.md) | Abstract execution model |
| 3 | [State and Types](mkdocs/src/manual/03-state-and-types.md) | Architectural state, data types |
| 4 | [Tiles and GlobalTensor](mkdocs/src/manual/04-tiles-and-globaltensor.md) | Core data structures |
| 5 | [Synchronization](mkdocs/src/manual/05-synchronization.md) | Event-based ordering |
| 6 | [PTO Assembly](mkdocs/src/manual/06-assembly.md) | Assembly syntax (PTO-AS) |
| 7 | [Instructions](mkdocs/src/manual/07-instructions.md) | ISA overview |
| 8 | [Programming Guide](mkdocs/src/manual/08-programming.md) | Development patterns |
| 9 | [Virtual ISA and IR](mkdocs/src/manual/09-virtual-isa-and-ir.md) | IR contracts |
| 10 | [Bytecode and Toolchain](mkdocs/src/manual/10-bytecode-and-toolchain.md) | Compilation pipeline |
| 11 | [Memory Ordering](mkdocs/src/manual/11-memory-ordering-and-consistency.md) | Consistency model |
| 12 | [Backend Profiles](mkdocs/src/manual/12-backend-profiles-and-conformance.md) | Platform conformance |

### 2. ISA Reference

| Resource | Description |
|----------|-------------|
| [ISA Index](isa/README.md) | Complete instruction listing by category |
| [PTOISA.md](PTOISA.md) | ISA quick reference table |
| [Conventions](isa/conventions.md) | Operand, event, and modifier notation |

### 3. IR Reference

| Resource | Description |
|----------|-------------|
| [IR Index](ir/README.md) | PTO IR operations |
| [PTO-IR-Ops.md](ir/PTO-IR-ops.md) | L1/L2 IR operation reference |

### 4. Programming Guide

| Resource | Description |
|----------|-------------|
| [Programming Model](coding/ProgrammingModel.md) | Core concepts and architecture |
| [Tile API](coding/Tile.md) | Tile type system |
| [GlobalTensor API](coding/GlobalTensor.md) | Memory view abstraction |
| [Scalar Types](coding/Scalar.md) | Immediate values |
| [Event API](coding/Event.md) | Synchronization |
| [Tutorial](coding/tutorial.md) | Writing your first kernel |

### 5. Assembly and Grammar

| Resource | Description |
|----------|-------------|
| [PTO-AS Spec](grammar/PTO-AS.md) | Assembly syntax reference |
| [BNF Grammar](grammar/PTO-AS.bnf) | Formal grammar definition |
| [Conventions](grammar/conventions.md) | Grammar notation |

### 6. Machine Model

| Resource | Description |
|----------|-------------|
| [Abstract Machine](machine/abstract-machine.md) | Core/device/host model |
| [Machine Index](machine/README.md) | Machine architecture |

---

## Architecture Contracts

### Source of Truth

| Source | Location |
|--------|----------|
| Per-op semantics | `docs/isa/*.md` |
| Public API | `include/pto/common/pto_instr.hpp` |
| Assembly grammar | `docs/grammar/PTO-AS.md`, `docs/grammar/PTO-AS.bnf` |

### Instruction Semantics

Each instruction document follows a standardized format:

1. **Operation diagram** - Visual representation
2. **Math interpretation** - Formal specification
3. **Assembly syntax** - PTO-AS and MLIR forms
4. **C++ intrinsic** - Programmatic interface
5. **Constraints** - Implementation requirements
6. **Examples** - Auto and Manual mode usage

---

## Implementation Notes

### Backend Support

| Backend | Path | Status |
|---------|------|--------|
| CPU (simulation) | `include/pto/cpu/` | ✅ Stable |
| Ascend A2/A3 | `include/pto/npu/a2a3/` | ✅ Stable |
| Ascend A5 | `include/pto/npu/a5/` | ✅ Stable |

### Extending PTO

For developers contributing new instructions, see:

- [Instruction Contract Template](mkdocs/src/manual/appendix-b-instruction-contract-template.md)
- [Coding README](coding/README.md)

---

## Navigation Tips

1. **New to PTO?** Start with [Chapter 1: Overview](mkdocs/src/manual/01-overview.md)
2. **Want to write code?** See [Programming Model](coding/ProgrammingModel.md)
3. **Need instruction details?** Browse [ISA Reference](isa/README.md)
4. **Understanding the toolchain?** See [Chapter 10: Bytecode and Toolchain](mkdocs/src/manual/10-bytecode-and-toolchain.md)

---

*For setup and installation, see [Getting Started](getting-started.md).*

