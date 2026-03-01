<p align="center">
  <img src="../../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO-AS (PTO Assembly)

PTO-AS is the **textual assembly format** for PTO (Parallel Tile Operation) programs.

## Overview

PTO-AS provides:

- **Readable, instruction-centric syntax** - Human-friendly representation of PTO operations
- **MLIR-like type system** - Strong typing with tensor and scalar types
- **Support for both SSA and destination-passing styles** - Flexible coding patterns
- **Direct compilation to C++** - Via the `ptoas` tool

## Quick Start

### Writing PTO Assembly

```pto
// Example: Matrix multiplication
module {
  func.func @matmul(%a: !pto.tensor<16x16xf16>, %b: !pto.tensor<16x16xf16>, %c: !pto.tensor<16x16xf16>) {
    %tA = pto.tload %a : !pto.tensor<16x16xf16>
    %tB = pto.tload %b : !pto.tensor<16x16xf16>
    %tC = pto.tload %c : !pto.tensor<16x16xf16>
    
    %tRes = pto.tmatmul %tA, %tB, %tC : 
      (!pto.MatTile, !pto.RightTile, !pto.AccTile) -> !pto.AccTile
    
    pto.tstore %c, %tRes : !pto.tensor<16x16xf16>
    return
  }
}
```

### Compiling with ptoas

```bash
# Compile PTO assembly to C++
ptoas input.pto -o output.cpp

# With auto-sync insertion
ptoas input.pto --enable-insert-sync -o output.cpp
```

## Specification

For the complete BNF grammar specification, see: [PTO-AS.bnf](PTO-AS.bnf)

For the full specification document, see: [PTO-AS.md](PTO-AS.md)

## Related Components

| Component | Description | Location |
|-----------|-------------|----------|
| **PTO-ISA** | Virtual instruction set for tile operations | [`include/pto/`](../../include/pto/) |
| **PTO-BC** | Binary bytecode encoding | [`docs/bytecode/`](../bytecode/) |
| **PTOAS** | MLIR-based assembler (submodule) | [`PTOAS/`](../../PTOAS/) |
| **pyPTO** | Python frontend for PTO | [External](https://gitcode.com/cann/pypto/) |
| **PTODSL** | Python DSL for kernel authoring | [`PTODSL/`](../../PTODSL/) |
| **TileLang Ascend** | High-level framework | [External](https://github.com/tile-ai/tilelang-ascend/) |

## Toolchain Flow

```
User Kernels (pyPTO / PTODSL / TileLang)
         │
         ▼
   PTO Text (.pto files)
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

## See Also

- [PTO-AS Grammar Specification](PTO-AS.md)
- [PTO-AS BNF](PTO-AS.bnf)
- [PTO Conventions](conventions.md)
- [PTO ISA Reference](../isa/README.md)
- [PTO Bytecode](../bytecode/pto-bc.md)

---

For the main documentation index, see: [docs/README.md](../README.md)
