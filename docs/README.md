<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO ISA Documentation

This directory contains comprehensive documentation for the PTO ISA (Instruction Set Architecture) used by PTO Tile Library.

## Documentation Overview

| Section | Description | Path |
|---------|-------------|------|
| **ISA Reference** | Complete instruction reference with specifications | [`docs/isa/`](isa/) |
| **PTO-AS Grammar** | Assembly language specification (textual format) | [`docs/grammar/`](grammar/) |
| **PTO-BC Bytecode** | Binary encoding specification | [`docs/bytecode/`](bytecode/) |
| **PTO-IR** | Non-ISA operations (L1/L2) | [`docs/ir/`](ir/) |
| **Programming Guide** | Developer guides and tutorials | [`docs/coding/`](coding/) |
| **Tutorials** | Step-by-step tutorials for common operations | [`docs/coding/tutorials/`](coding/tutorials/) |
| **Machine Model** | Abstract machine architecture | [`docs/machine/`](machine/) |

## Quick Navigation

### For New Users

- **Getting Started**: [getting-started.md](../getting-started.md)
- **Programming Model**: [coding/ProgrammingModel.md](coding/ProgrammingModel.md)
- **Beginner Tutorial**: [coding/tutorial.md](coding/tutorial.md)
- **Tutorials**: [coding/tutorials/](coding/tutorials/)
  - [Vector Add](coding/tutorials/vec-add.md)
  - [GEMM Operation](coding/tutorials/gemm.md)
  - [Row Softmax](coding/tutorials/row-softmax.md)

### For ISA Reference

- **Instruction Index**: [isa/README.md](isa/README.md)
- **Common Conventions**: [isa/conventions.md](isa/conventions.md)
- **PTO-AS Specification**: [grammar/PTO-AS.md](grammar/PTO-AS.md)
- **PTO-BC Bytecode**: [bytecode/pto-bc.md](bytecode/pto-bc.md)

### For Advanced Topics

- **Abstract Machine Model**: [machine/abstract-machine.md](machine/abstract-machine.md)
- **Optimization Guide**: [coding/opt.md](coding/opt.md)
- **Debugging Guide**: [coding/debug.md](coding/debug.md)

## Key Concepts

### PTO ISA

The PTO (Parallel Tile Operation) ISA defines over 90 standard tile-level operations. Key instructions include:

- **Memory Operations**: `TLOAD`, `TSTORE`, `TGATHER`, `TSCATTER`
- **Arithmetic Operations**: `TADD`, `TSUB`, `TMUL`, `TDIV`
- **Matrix Operations**: `TMATMUL`, `TEXTRACT`
- **Reduction Operations**: `TROWSUM`, `TCOLSUM`, `TROWMAX`, `TCOLMAX`

See: [PTOISA.md](PTOISA.md) for complete instruction list.

### PTO-AS (PTO Assembly)

PTO-AS is the **textual assembly format** for PTO. It provides:

- Readable, instruction-centric syntax
- MLIR-like type system
- Support for both SSA and destination-passing styles

See: [grammar/PTO-AS.md](grammar/PTO-AS.md) for specification.

### PTO-BC (PTO Bytecode)

PTO-BC is the **binary bytecode encoding** for PTO programs. It provides:

- Compact binary representation
- Forward compatibility
- MLIR-independent decoding (optional)

See: [bytecode/pto-bc.md](bytecode/pto-bc.md) for specification.

### Programming Models

PTO supports two programming models:

1. **Auto Mode** (CPU Simulation Only): Automatic buffer allocation and synchronization
2. **Manual Mode**: Explicit buffer management and pipeline control

See: [coding/ProgrammingModel.md](coding/ProgrammingModel.md) for details.

## How the PTO Flow Works

The PTO toolchain transforms user kernels into executable code for Ascend NPUs or CPU simulation:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           1. User Kernel Development                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │      pyPTO      │  │     PTODSL      │  │  TileLang Ascend │  │  PTO-Lang   │ │
│  │  (Python DSL)  │  │   (Python DSL)  │  │  (High-level)   │  │  (Generic)  │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └──────┬─────┘ │
└───────────┼─────────────────────┼─────────────────────┼─────────────────────┼────────┘
            │                     │                     │                     │
            ▼                     ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           2. PTO Text Format (.pto files)                          │
│  Textual representation of tile operations: tadd, tload, tmatmul, tstore, etc.    │
│  Example:                                                                            │
│  ```pto                                                                              │
│  func.func @gemm(%a: !pto.tensor<16x16xf16>, %b: !pto.tensor<16x16xf16>) {       │
│    %tA = pto.tload %a : !pto.tensor<16x16xf16>                                    │
│    %tB = pto.tload %b : !pto.tensor<16x16xf16>                                    │
│    %tC = pto.tmatmul %tA, %tB : (!pto.MatTile, !pto.RightTile) -> !pto.AccTile  │
│    pto.tstore %c, %tC : !pto.tensor<16x16xf16>                                    │
│    return                                                                            │
│  }                                                                                  │
│  ```                                                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼ (ptoas assembler)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           3. ptoas - PTO Assembler                                  │
│  Compiles PTO text to AscendC C++ kernel code                                      │
│                                                                                     │
│  Usage: `ptoas input.pto -o output.cpp`                                           │
│  Tool: [`PTOAS/`](../PTOAS/) (submodule)                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
            │
            ├──────────────────────────────────┬─────────────────────────────────────┐
            │                                  │                                     │
            ▼ (optional)                       ▼ (optional)                          │
┌─────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│    4a. PTO-C++ Output     │   │    4b. ptobc - PTO Bytecode Encoder           │
│    Generated AscendC C++   │   │    Binary encoding for PTO programs            │
│    kernel code             │   │                                                 │
│                             │   │    Usage: `ptobc encode input.pto -o out.ptobc│
│                             │   │    Tool: [`tools/ptobc/`](../tools/ptobc/)  │
└──────────────┬──────────────┘   └─────────────────────┬───────────────────────┘
               │                                        │
               ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           5. PTO ISA (90+ Tile Operations)                          │
│  Core instruction set: TADD, TMATMUL, TLOAD, TSTORE, TGATHER, TSCATTER, etc.      │
│  Headers: [`include/pto/`](../include/pto/)                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           6. Backend Execution                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  Ascend A2   │  │  Ascend A3   │  │  Ascend A5   │  │        CPU          │  │
│  │   (910B)    │  │   (910C)    │  │   (950)     │  │   (x86_64/AArch64)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Quick Start: Compiling a PTO Kernel

```bash
# Step 1: Write your kernel in PTO text format (example.pto)
# Step 2: Compile with ptoas
ptoas example.pto -o example.cpp

# Step 3: Compile the generated C++ with AscendC compiler
aicompile example.cpp -o example.o

# Step 4: Link and run on NPU
```

```bash
# Alternative: Encode to PTO-BC bytecode
ptobc encode example.pto -o example.ptobc
```

## PTO Ecosystem

```
User Kernels (pyPTO / PTODSL / TileLang)
         │
         ▼
   PTO Text (.pto)
         │
    ┌────┴────┐
    │         │
  ptoas    ptobc
    │         │
    ▼         ▼
  C++     PTO-BC
    │         │
    └────┬────┘
         │
         ▼
    PTO ISA
```

## Related Documentation

| Document | Description |
|----------|-------------|
| [include/README.md](../include/README.md) | C++ API and header files |
| [kernels/README.md](../kernels/README.md) | Kernel implementations |
| [tests/README.md](../tests/README.md) | Test suite documentation |
| [PTOAS submodule](../PTOAS/) | Assembler and MLIR dialect |
| [PTODSL submodule](../PTODSL/) | Python DSL for kernel authoring |
| [tools/ptobc/](../tools/ptobc/) | PTO-BC encoder/decoder |

## External Resources

| Project | Description | Link |
|---------|-------------|------|
| **pyPTO** | Python-first frontend for PTO kernels | [GitCode](https://gitcode.com/cann/pypto/) |
| **PTODSL** | Python DSL for kernel authoring | [GitHub](https://github.com/huawei-csl/pto-dsl) |
| **PTOAS** | Assembler and MLIR dialect | [GitHub](https://github.com/zhangstevenunity/PTOAS) |
| **TileLang Ascend** | High-level framework for Ascend | [GitHub](https://github.com/tile-ai/tilelang-ascend/) |

---

For the main project README, see: [README.md](../README.md)
