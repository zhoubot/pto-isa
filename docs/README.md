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
| **Machine Model** | Abstract machine architecture | [`docs/machine/`](machine/) |

## Quick Navigation

### For New Users

- **Getting Started**: [getting-started.md](../getting-started.md)
- **Programming Model**: [coding/ProgrammingModel.md](coding/ProgrammingModel.md)
- **Tutorial**: [coding/tutorial.md](coding/tutorial.md)

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
         │
    ┌────┴────┐
    │         │
  NPU      CPU
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
| **PTODSL** | Python DSL | [GitHub](https://github.com/huawei-csl/pto-dsl) |
| **PTOAS** | Assembler and MLIR dialect | [GitHub](https://github.com/zhangstevenunity/PTOAS) |
| **TileLang Ascend** | High-level framework | [GitHub](https://github.com/tile-ai/tilelang-ascend/) |

---

For the main project README, see: [README.md](../README.md)
