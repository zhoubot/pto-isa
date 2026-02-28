<p align="center">
  <img src="../../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO-BC (PTO Bytecode)

PTO-BC is the **binary bytecode encoding** for PTO (Parallel Tile Operation) programs.

## Overview

PTO-BC provides:

- **Compact binary representation** - Efficient storage and transmission
- **Forward compatibility** - Version-aware encoding/decoding
- **MLIR-independent decoding** - Can be decoded without MLIR dependencies
- **Fast loading** - Optimized for quick program loading

## Quick Start

### Encoding PTO to Bytecode

```bash
# Encode PTO text to bytecode
ptobc encode input.pto -o output.ptobc

# Decode bytecode back to text
ptobc decode input.ptobc -o output.pto
```

### Programmatic Usage

```python
from ptobc import encode, decode

# Encode
with open("input.pto", "r") as f:
    pto_text = f.read()
    
bytecode = encode(pto_text)

with open("output.ptobc", "wb") as f:
    f.write(bytecode)

# Decode
with open("input.ptobc", "rb") as f:
    bytecode = f.read()
    
pto_text = decode(bytecode)
```

## Specification

For the complete bytecode specification, see: [pto-bc.md](pto-bc.md)

## Related Components

| Component | Description | Location |
|-----------|-------------|----------|
| **PTO-ISA** | Virtual instruction set for tile operations | [`include/pto/`](include/pto/) |
| **PTO-AS** | Textual assembly format | [`docs/grammar/`](docs/grammar/) |
| **PTOAS** | MLIR-based assembler (submodule) | [`PTOAS/`](PTOAS/) |
| **pyPTO** | Python frontend for PTO | [External](https://gitcode.com/cann/pypto/) |
| **PTODSL** | Python DSL for kernel authoring | [`PTODSL/`](PTODSL/) |
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

## Tool: ptobc

The `ptobc` tool is located at [`tools/ptobc/`](../../tools/ptobc/) and currently provides:

- `ptobc encode` - Convert PTO text to bytecode
- `ptobc decode` - Convert bytecode back to PTO text

Tool behavior / encoding rules (implementation notes): [`ptobc.md`](ptobc.md)

## See Also

- [PTO Bytecode Specification](pto-bc.md)
- [PTO-AS Specification](../grammar/PTO-AS.md)
- [PTO ISA Reference](../isa/README.md)
- [PTO Toolchain Overview](../mkdocs/src/manual/10-bytecode-and-toolchain.md)

---

For the main documentation index, see: [docs/README.md](../README.md)
