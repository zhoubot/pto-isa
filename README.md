<p align="center">
  <img src="docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# PTO Tile Library

> High-performance tile-level operations for Ascend platforms

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Ascend%20A2%20%7C%20A3%20%7C%20A5%20%7C%20CPU-green.svg)](#platform-support)

## Overview

**PTO (Parallel Tile Operation)** is a virtual instruction set architecture (ISA) designed by Ascend CANN, focusing on tile-level operations. This repository provides **high-performance, cross-platform tile operations** across Ascend platforms, enabling easier migration between different Ascend hardware generations.

The PTO ISA is built on Ascend's underlying hardware and software abstractions, providing **90+ standard tile-level operations**.

## News

- **2025-12-27**: PTO Tile Library becomes publicly available.

## Key Components

This repository contains several interconnected components:

| Component | Description | Location |
|-----------|-------------|----------|
| **PTO ISA** | Virtual instruction set for tile operations | [`include/pto/`](include/pto/) |
| **PTO-AS** | Textual assembly format for PTO | [`docs/grammar/PTO-AS.md`](docs/grammar/PTO-AS.md) |
| **PTO-BC** | Binary bytecode encoding for PTO | [`docs/bytecode/pto-bc.md`](docs/bytecode/pto-bc.md) |
| **PTOAS** | Assembler tool (`ptoas`) and MLIR dialect | [`PTOAS/`](PTOAS/) (submodule) |
| **PTO-Lang** | High-level kernel language | (Coming Soon) |
| **PTODSL** | Python DSL for kernel authoring | [`PTODSL/`](PTODSL/) (submodule) |

### Complete PTO Toolchain Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Kernels                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   PTO-Lang     │  │     PTODSL      │  │   TileLang Ascend   │ │
│  │ (Coming Soon)   │  │  (Python DSL)   │  │   (External)        │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘ │
└───────────┼─────────────────────┼──────────────────────┼────────────┘
            │                     │                      │
            ▼                     ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PTO (Text Format - .pto files)                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Textual representation: tadd, tload, tmatmul, tstore, ...   │ │
│  │  See: [`docs/grammar/PTO-AS.md`](docs/grammar/PTO-AS.md)    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼ (ptoas tool)
┌─────────────────────────────────────────────────────────────────────┐
│  PTO-C++ (Generated C++ Code)                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  ptoas compiles .pto files to AscendC C++ kernel code        │ │
│  │  Usage: ptoas input.pto -o output.cpp                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼ (bisheng)
┌─────────────────────────────────────────────────────────────────────┐
│  Binary (.bin)                                                      │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼ (optional: ptobc tool)
┌─────────────────────────────────────────────────────────────────────┐
│  PTO-BC (PTO Bytecode - Binary)                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Binary encoding for PTO programs                               │ │
│  │  Usage: ptobc encode input.pto -o out.ptobc                   │ │
│  │  See: [`docs/bytecode/pto-bc.md`](docs/bytecode/pto-bc.md)  │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼ (via MLIR lowering)
┌─────────────────────────────────────────────────────────────────────┐
│  PTO ISA (PTO Instruction Set Architecture)                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  90+ tile operations: TADD, TMATMUL, TLOAD, TSTORE, ...      │ │
│  │  See: [`include/pto/`](include/pto/)                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Backend Implementations                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Ascend A2   │  │  Ascend A3   │  │  Ascend A5   │   CPU       │
│  │  (910B)     │  │  (910C)     │  │  (950)      │   (x86/AArch)│
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `ptoas` | PTO Assembler - compiles PTO text to C++ | `ptoas input.pto -o output.cpp` |
| `ptobc` | PTO Bytecode encoder/decoder | `ptobc encode input.pto -o out.ptobc` |

## Frontends

PTO supports multiple frontends for kernel authoring:

| Frontend | Language | Description | Location |
|----------|----------|-------------|----------|
| **PTO-Lang** | Various | High-level kernel language (Coming Soon) | TBD |
| **PTODSL** | Python | Pythonic DSL similar to cuTile | [`PTODSL/`](PTODSL/) |
| **TileLang Ascend** | Python | High-level framework integration | [External](https://github.com/tile-ai/tilelang-ascend/) |

## Target Users

PTO Tile Lib is designed for experienced developers:

- **Backend developers** implementing frameworks that directly interface with Ascend hardware
- **Cross-platform application developers** needing hardware abstraction
- **High-performance operator developers** (manual operator/kernel implementations)

> **Note**: This is not a beginner-level library.

## Platform Support

| Platform | Status | Implementation |
|----------|--------|----------------|
| Ascend A2 (910B) | Supported | [`include/pto/npu/a2a3/`](include/pto/npu/a2a3/) |
| Ascend A3 (910C) | Supported | [`include/pto/npu/a2a3/`](include/pto/npu/a2a3/) |
| Ascend A5 (950) | Supported | [`include/pto/npu/a5/`](include/pto/npu/a5/) |
| CPU (x86_64/AArch64) | Supported | Simulation backend |

## Quick Start

### CPU Simulator (Recommended First Step)

```bash
python3 tests/run_cpu.py --clean --verbose
```

### Build & Run Demos

```bash
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

## Repository Structure

| Directory | Description |
|-----------|-------------|
| [`include/pto/`](include/pto/) | PTO C++ header files and public API |
| [`PTOAS/`](PTOAS/) | PTO Assembler - MLIR dialect (submodule) |
| [`PTODSL/`](PTODSL/) | PTO Python DSL (submodule) |
| [`docs/`](docs/) | ISA documentation, guides, and tutorials |
| [`docs/grammar/`](docs/grammar/) | PTO-AS assembly language specification |
| [`docs/bytecode/`](docs/bytecode/) | PTO-BC bytecode specification |
| [`tools/ptobc/`](tools/ptobc/) | PTO-BC encoder/decoder tool |
| [`tests/`](tests/) | ST/CPU test scripts and test cases |

## Documentation

- [PTO-AS Specification](docs/grammar/PTO-AS.md)
- [PTO-BC Bytecode](docs/bytecode/pto-bc.md)
- [Getting Started](docs/getting-started.md)
- [ISA Reference](docs/isa/README.md)

## License

This project is licensed under the CANN Open Software License Agreement Version 2.0. See the [`LICENSE`](LICENSE) file for details.

---

<p align="center">
  <strong>PTO Tile Library</strong> — Enabling high-performance tile operations across Ascend platforms
</p>
