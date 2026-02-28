<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](../../LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-yellow.svg)](../../version.cmake)

</div>

# PTO Programming Guide

This directory describes the **PTO Tile Library programming model** as seen from C++ — Tiles, GlobalTensor, events, scalar parameters — and provides guidance for extending the library.

If you are looking for the *ISA reference*, start from [`docs/isa/README.md`](../isa/README.md).

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [Programming Model](ProgrammingModel.md) | Core concepts and architecture |
| [Tutorial](tutorial.md) | Write your first kernel |
| [Tutorials](tutorials/README.md) | More example kernels |

---

## Core Concepts

### Data Structures

| Document | Description |
|----------|-------------|
| [Tile](Tile.md) | Tile abstraction, layout, and valid-region rules |
| [GlobalTensor](GlobalTensor.md) | Global memory tensors (shape/stride/layout) |
| [Scalar](Scalar.md) | Scalar values, type mnemonics, and enumerations |
| [Event](Event.md) | Events and synchronization model |

### Programming Guide

| Document | Description |
|----------|-------------|
| [ProgrammingModel](ProgrammingModel.md) | High-level model: PTO-Auto vs PTO-Manual |
| [Tutorial](tutorial.md) | Hands-on tutorial |
| [Debug](debug.md) | Debugging and assertion lookup |
| [Optimization](opt.md) | Performance optimization tips |

### Tutorials

| Document | Description |
|----------|-------------|
| [Tutorials Index](tutorials/README.md) | All tutorials |
| [Vector Addition](tutorials/vec-add.md) | Basic elementwise operation |
| [Row Softmax](tutorials/row-softmax.md) | Reduction and broadcast |
| [GEMM](tutorials/gemm.md) | Matrix multiplication |

---

## Architecture

### Abstract Machine

See the [Abstract Machine Model](../machine/abstract-machine.md) for the core/device/host execution model.

### Extension Guide

To extend PTO with new instructions:

1. See the [Instruction Contract Template](../mkdocs/src/manual/appendix-b-instruction-contract-template.md)
2. Follow the conventions in [ISA Reference](../isa/README.md)
3. Add implementation in `include/pto/npu/`

---

## Related Resources

| Resource | Description |
|----------|-------------|
| [ISA Reference](../isa/README.md) | Complete instruction listing |
| [PTO-AS Spec](../grammar/PTO-AS.md) | Assembly syntax |
| [Virtual ISA Manual](../mkdocs/src/manual/09-virtual-isa-and-ir.md) | ISA contract |
| [Machine Model](../machine/README.md) | Abstract machine |

