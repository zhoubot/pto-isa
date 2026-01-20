# PTO ISA Compiler

A Domain-Specific Language (DSL) compiler for **Programmable Tensor Operations (PTO)** Instruction Set Architecture.

## Overview

The PTO ISA operates on **Tiles** - 2-dimensional blocks of data representing tensor slices. This compiler provides:

- **Complete ISA Definition**: All PTO instructions defined in Python
- **DSL for Program Construction**: Fluent interface for building PTO programs
- **Loop Constructs**: Single and nested loops with iteration counts derived from tile shapes
- **Loop Fusion Optimization**: Combines consecutive elementwise operations into single fused loops
- **Multi-Backend Code Generation**: ARM64 NEON, NVIDIA CUDA, Huawei Ascend 910B
- **Type Checking**: Validation of tile shapes and element types

## Architecture

```
PTO_ISA_Compiler/
├── pto_compile.py              # Unified compiler (frontend + optimizer + backends)
├── pto_isa_definition.py       # Complete PTO ISA instruction definitions
├── pto-isa-cheatsheet.pdf      # ISA reference
├── PTO_ISA_improvement_ideas.md # Improvement suggestions
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
└── examples/                   # Example programs and generated outputs
    ├── pto_isa_sinh.py               # sinh() Taylor expansion
    ├── pto_aten_ir_primitives.py     # 27 ATen IR primitives
    ├── pto_torch_functional.py       # 38 torch.nn.functional APIs
    ├── pto_torch_nn_operators.py     # 24 torch.nn operators
    ├── pto_torch_tensor.py           # 55 Tensor methods
    │
    ├── output_arm64/             # ARM64 NEON generated code
    │   ├── sinh_taylor/
    │   ├── aten_primitives/
    │   ├── torch_functional/
    │   ├── torch_nn/
    │   └── torch_tensor/
    ├── output_cuda/              # NVIDIA CUDA generated code
    │   └── ...
    ├── output_ascend910b/        # Huawei Ascend 910B generated code
    │   └── ...
    └── output_pto/               # PTO Assembly generated code
        └── ...
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Example

```python
from pto_compile import PTOFunctionBuilder, PTOCompiler, generate_all_backends
from pto_isa_definition import ElementType, MemorySpace

# Build a GELU activation program
program = (PTOFunctionBuilder("gelu")
    .tile("x", 8, 8, ElementType.F32)
    .tile("y", 8, 8, ElementType.F32)
    .memref("input", MemorySpace.GM, ElementType.F32)
    .memref("output", MemorySpace.GM, ElementType.F32)
    .load("x", "input")
    .mul("y", "x", "x")       # y = x²
    .muls("y", "y", 0.044715) # y = 0.044715 * x²
    .add("y", "x", "y")       # y = x + 0.044715 * x²
    .exp("y", "y")            # y = exp(...)
    .mul("y", "x", "y")       # y = x * exp(...)
    .store("y", "output")
    .build())

# Generate code for all backends
generate_all_backends(program, "gelu_output", ".")
```

### Run Examples

```bash
# Generate sinh() for ARM64, CUDA, Ascend
python3 examples/pto_isa_sinh.py

# Generate torch.nn.functional implementations
python3 examples/pto_torch_functional.py

# Generate Tensor methods
python3 examples/pto_torch_tensor.py
```

## Loop Fusion Optimization

The compiler automatically fuses consecutive elementwise operations:

```
Before fusion (5 separate loops):
for (row) for (col) { y = x * x; }
for (row) for (col) { y = y * 0.044715; }
for (row) for (col) { y = x + y; }
...

After fusion (1 fused loop):
for (row) for (col) {
    y = x * x;
    y = y * 0.044715;
    y = x + y;
    ...
}
```

**Benefits:**
- Reduces loop overhead by 80-95%
- Improves cache locality
- Significant code size reduction

## Supported Backends

| Backend | Output | Description |
|---------|--------|-------------|
| ARM64 NEON | `.c` | Apple Silicon, ARM servers |
| NVIDIA CUDA | `.cu` | GPU computing |
| Huawei Ascend 910B | `.cpp` | NPU/AI accelerator |
| PTO-AS | `.pto` | PTO assembly (portable) |

## PTO ISA Instructions

### Tile Instructions

| Category | Instructions |
|----------|-------------|
| Memory | `TLOAD`, `TSTORE` |
| Elementwise Unary | `TABS`, `TNEG`, `TEXP`, `TLOG`, `TSQRT`, `TRSQRT`, `TRECIP`, `TRELU` |
| Elementwise Binary | `TADD`, `TSUB`, `TMUL`, `TDIV`, `TMAX`, `TMIN` |
| Scalar Ops | `TADDS`, `TSUBS`, `TMULS`, `TDIVS` |
| Matrix | `TMATMUL`, `TMATMUL_ACC` |
| Reduction | `TROWSUM`, `TCOLSUM` |
| Broadcast | `TEXPANDS`, `TROWEXPAND`, `TCOLEXPAND`, `TROWEXPANDSUB`, `TROWEXPANDDIV`, `TROWEXPANDMUL` |

### Control Flow

| Category | Instructions |
|----------|-------------|
| Loops | `FOR`, `ENDFOR`, `WHILE`, `DO`, `ENDWHILE` |
| Conditional | `IF`, `ELSE`, `ENDIF` |

---

# PTOFunctionBuilder Reference Guide

## Overview

`PTOFunctionBuilder` is a fluent interface for constructing PTO programs. It provides a chainable API to declare tiles, memory references, and operations, then builds a `PTOProgram` object that can be compiled to multiple backends.

## Basic Structure

```python
from pto_compile import PTOFunctionBuilder, PTOCompiler
from pto_isa_definition import ElementType, MemorySpace

program = (PTOFunctionBuilder("program_name")
    # 1. Declare tiles (local tensor storage)
    .tile("name", rows, cols, dtype)
    
    # 2. Declare memory references (external memory)
    .memref("name", memory_space, dtype)
    
    # 3. Load data from memory to tiles
    .load("tile_name", "memref_name")
    
    # 4. Perform operations
    .add("dst", "src0", "src1")
    
    # 5. Store results back to memory
    .store("tile_name", "memref_name")
    
    # 6. Build the program
    .build())
```

## Declaration Methods

### `.tile(name, rows, cols, dtype)`

Declare a local tile (2D tensor in on-chip memory).

```python
.tile("x", 8, 8, ElementType.F32)      # 8x8 float32 tile
.tile("weight", 64, 128, ElementType.F16)  # 64x128 float16 tile
```

**Parameters:**
- `name`: Tile identifier (string)
- `rows`: Number of rows (int)
- `cols`: Number of columns (int)
- `dtype`: Element type (`ElementType.F32`, `F16`, `BF16`, `I32`, etc.)

### `.scalar(name, dtype)`

Declare a scalar variable.

```python
.scalar("alpha", ElementType.F32)
```

### `.memref(name, space, dtype, shape=None)`

Declare a memory reference (pointer to external memory).

```python
.memref("input", MemorySpace.GM, ElementType.F32)      # Global memory
.memref("weight", MemorySpace.L1, ElementType.F16)     # L1 cache
```

**Memory Spaces:**
- `MemorySpace.GM` - Global Memory
- `MemorySpace.L1` - L1 Cache
- `MemorySpace.L2` - L2 Cache
- `MemorySpace.UB` - Unified Buffer (Ascend)

## Memory Operations

### `.load(dst_tile, src_memref, row=0, col=0)`

Load data from memory into a tile.

```python
.load("x", "input")           # Load at offset (0, 0)
.load("x", "input", 8, 16)    # Load at offset (8, 16)
```

### `.store(src_tile, dst_memref, row=0, col=0)`

Store tile data to memory.

```python
.store("result", "output")    # Store at offset (0, 0)
```

## Arithmetic Operations

### Binary Operations

| Method | Operation | Description |
|--------|-----------|-------------|
| `.add(dst, src0, src1)` | `dst = src0 + src1` | Elementwise addition |
| `.sub(dst, src0, src1)` | `dst = src0 - src1` | Elementwise subtraction |
| `.mul(dst, src0, src1)` | `dst = src0 * src1` | Elementwise multiplication |
| `.div(dst, src0, src1)` | `dst = src0 / src1` | Elementwise division |
| `.max(dst, src0, src1)` | `dst = max(src0, src1)` | Elementwise maximum |
| `.min(dst, src0, src1)` | `dst = min(src0, src1)` | Elementwise minimum |

```python
.add("c", "a", "b")    # c = a + b
.mul("y", "x", "x")    # y = x * x (square)
```

### Scalar Operations

| Method | Operation | Description |
|--------|-----------|-------------|
| `.adds(dst, src, scalar)` | `dst = src + scalar` | Add scalar to all elements |
| `.muls(dst, src, scalar)` | `dst = src * scalar` | Multiply all elements by scalar |
| `.divs(dst, src, scalar)` | `dst = src / scalar` | Divide all elements by scalar |

```python
.adds("y", "x", 1.0)      # y = x + 1.0
.muls("y", "x", 0.5)      # y = x * 0.5
.divs("y", "x", 2.0)      # y = x / 2.0
```

### Unary Operations

| Method | Operation | Description |
|--------|-----------|-------------|
| `.neg(dst, src)` | `dst = -src` | Negation |
| `.abs(dst, src)` | `dst = \|src\|` | Absolute value |
| `.sqrt(dst, src)` | `dst = √src` | Square root |
| `.rsqrt(dst, src)` | `dst = 1/√src` | Reciprocal square root |
| `.recip(dst, src)` | `dst = 1/src` | Reciprocal |
| `.exp(dst, src)` | `dst = eˢʳᶜ` | Exponential |
| `.log(dst, src)` | `dst = ln(src)` | Natural logarithm |
| `.relu(dst, src)` | `dst = max(0, src)` | ReLU activation |

```python
.exp("exp_x", "x")       # exp_x = exp(x)
.relu("activated", "x")  # activated = ReLU(x)
```

## Reduction Operations

### `.rowsum(dst, src)`

Sum elements across each row. Output shape: `[rows, 1]`

```python
.tile("x", 8, 8)
.tile("row_sums", 8, 1)
.rowsum("row_sums", "x")  # row_sums[i][0] = sum(x[i][:])
```

### `.colsum(dst, src)`

Sum elements across each column. Output shape: `[1, cols]`

```python
.tile("x", 8, 8)
.tile("col_sums", 1, 8)
.colsum("col_sums", "x")  # col_sums[0][j] = sum(x[:][j])
```

## Broadcast Operations

### `.expands(dst, value)`

Broadcast a scalar value to fill a tile.

```python
.tile("ones", 8, 8)
.expands("ones", 1.0)     # Fill with 1.0
```

### `.rowexpandsub(dst, src0, src1)` / `.rowexpanddiv()` / `.rowexpandmul()`

Row-wise broadcast operation. `src1` must be `[rows, 1]` shape.

```python
.tile("x", 8, 8)
.tile("mean", 8, 1)
.tile("centered", 8, 8)
.rowsum("mean", "x")
.divs("mean", "mean", 8.0)
.rowexpandsub("centered", "x", "mean")  # centered = x - broadcast(mean)
```

## Matrix Operations

### `.matmul(dst, a, b)`

Matrix multiplication: `dst = a @ b`

```python
.tile("a", 64, 128)
.tile("b", 128, 64)
.tile("c", 64, 64)
.matmul("c", "a", "b")  # c[64,64] = a[64,128] @ b[128,64]
```

## Loop Constructs

### `.for_loop(iv_name, lb, ub, step=1)` / `.end_for()`

Create a FOR loop with explicit bounds.

```python
.for_loop("i", 0, 4, 1)
    .load("tile", "mem")
    .relu("tile", "tile")
    .store("tile", "mem")
.end_for()
```

### `.tile_loop(iv_name, tile_name, dimension, step=1)`

Loop based on tile dimensions.

```python
.tile("data", 64, 64)
.tile_loop("i", "data", "rows")  # Loop 0..64
    # operations
.end_for()
```

### `.nested_tile_loop(outer_iv, inner_iv, tile_name)` / `.end_nested_loop()`

2-level nested loop over tile dimensions.

```python
.tile("data", 64, 64)
.nested_tile_loop("i", "j", "data")  # Outer: rows, Inner: cols
    # operations
.end_nested_loop()
```

## Complete Examples

### Example 1: Sigmoid Activation

```python
def build_sigmoid():
    """sigmoid(x) = 1 / (1 + exp(-x))"""
    return (PTOFunctionBuilder("sigmoid")
        .tile("x", 8, 8, ElementType.F32)
        .tile("neg_x", 8, 8, ElementType.F32)
        .tile("exp_neg", 8, 8, ElementType.F32)
        .tile("one_plus", 8, 8, ElementType.F32)
        .tile("result", 8, 8, ElementType.F32)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        .load("x", "input")
        .neg("neg_x", "x")              # neg_x = -x
        .exp("exp_neg", "neg_x")        # exp_neg = exp(-x)
        .adds("one_plus", "exp_neg", 1.0)  # one_plus = 1 + exp(-x)
        .recip("result", "one_plus")    # result = 1 / (1 + exp(-x))
        .store("result", "output")
        .build())
```

### Example 2: Layer Normalization

```python
def build_layer_norm(rows=8, cols=8, eps=1e-5):
    """LayerNorm: (x - mean) / sqrt(var + eps)"""
    return (PTOFunctionBuilder("layer_norm")
        .tile("x", rows, cols, ElementType.F32)
        .tile("mean", rows, 1, ElementType.F32)
        .tile("centered", rows, cols, ElementType.F32)
        .tile("sq_centered", rows, cols, ElementType.F32)
        .tile("var", rows, 1, ElementType.F32)
        .tile("std", rows, 1, ElementType.F32)
        .tile("result", rows, cols, ElementType.F32)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        .load("x", "input")
        # Mean
        .rowsum("mean", "x")
        .divs("mean", "mean", float(cols))
        # Center
        .rowexpandsub("centered", "x", "mean")
        # Variance
        .mul("sq_centered", "centered", "centered")
        .rowsum("var", "sq_centered")
        .divs("var", "var", float(cols))
        # Std
        .adds("var", "var", eps)
        .sqrt("std", "var")
        # Normalize
        .rowexpanddiv("result", "centered", "std")
        .store("result", "output")
        .build())
```

### Example 3: Softmax

```python
def build_softmax(rows=8, cols=8):
    """Softmax: exp(x - max) / sum(exp(x - max))"""
    return (PTOFunctionBuilder("softmax")
        .tile("x", rows, cols, ElementType.F32)
        .tile("row_max", rows, 1, ElementType.F32)
        .tile("shifted", rows, cols, ElementType.F32)
        .tile("exp_x", rows, cols, ElementType.F32)
        .tile("row_sum", rows, 1, ElementType.F32)
        .tile("result", rows, cols, ElementType.F32)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        .load("x", "input")
        # Compute row max (simplified with mean)
        .rowsum("row_max", "x")
        .divs("row_max", "row_max", float(cols))
        # Shift for numerical stability
        .rowexpandsub("shifted", "x", "row_max")
        # Exp
        .exp("exp_x", "shifted")
        # Sum
        .rowsum("row_sum", "exp_x")
        # Normalize
        .rowexpanddiv("result", "exp_x", "row_sum")
        .store("result", "output")
        .build())
```

## Compiling Programs

### Generate PTO Assembly

```python
from pto_compile import PTOCompiler

program = build_sigmoid()
compiler = PTOCompiler()
pto_asm = compiler.compile(program)
print(pto_asm)
```

### Generate Multi-Backend Code

```python
from pto_compile import generate_all_backends

program = build_sigmoid()
results = generate_all_backends(program, "activations", "examples")
# Creates:
#   examples/output_arm64/activations/sigmoid.c
#   examples/output_cuda/activations/sigmoid.cu
#   examples/output_ascend910b/activations/sigmoid.cpp
#   examples/output_pto/activations/sigmoid.pto
```

### Generate Specific Backend

```python
from pto_compile import generate_arm64_code, generate_cuda_code, generate_ascend_code

program = build_sigmoid()

arm64_code = generate_arm64_code(program, enable_fusion=True)
cuda_code = generate_cuda_code(program, enable_fusion=True)
ascend_code = generate_ascend_code(program, enable_fusion=True)
```

## API Summary

| Category | Methods |
|----------|---------|
| Declaration | `tile()`, `scalar()`, `memref()` |
| Memory | `load()`, `store()` |
| Binary Ops | `add()`, `sub()`, `mul()`, `div()`, `max()`, `min()` |
| Scalar Ops | `adds()`, `muls()`, `divs()` |
| Unary Ops | `neg()`, `abs()`, `sqrt()`, `rsqrt()`, `recip()`, `exp()`, `log()`, `relu()` |
| Reduction | `rowsum()`, `colsum()` |
| Broadcast | `expands()`, `rowexpandsub()`, `rowexpanddiv()`, `rowexpandmul()` |
| Matrix | `matmul()`, `matmul_acc()` |
| Loops | `for_loop()`, `end_for()`, `tile_loop()`, `nested_tile_loop()`, `end_nested_loop()` |
| Build | `build()` |

## License

MIT License
