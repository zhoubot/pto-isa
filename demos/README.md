<p align="center">
  <img src="docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO Demos and Examples

This directory contains demonstrations and examples for the PTO (Parallel Tile Operation) ecosystem.

## Directory Structure

```
demos/
├── baseline/           # Baseline implementations
├── cpu/                # CPU simulation examples
├── python/             # Python-based examples
├── torch_jit/         # PyTorch JIT examples
└── ptodsl_ptoas_cpu/  # PTODSL + PTOAS CPU examples
    ├── add_static/             # Static addition kernel
    ├── add_dynamic_multicore/  # Dynamic multi-core addition
    ├── matmul_static_singlecore/   # Single-core GEMM
    ├── matmul_dynbatch_multicore/  # Dynamic batch multi-core GEMM
    └── relu_dynamic_multicore/    # Dynamic multi-core ReLU
```

## Quick Start with PTODSL

### Prerequisites

```bash
# Initialize submodules
git submodule update --init --recursive

# Build PTOAS
cd PTOAS && mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### Running Examples

```bash
# Run all PTODSL examples
cd demos/ptodsl_ptoas_cpu
./run_all.sh

# Run specific example
cd demos/ptodsl_ptoas_cpu/add_static
./run.sh
```

## Example: GEMM with PTODSL

```python
import pto

# Define a GEMM kernel
@pto.kernel
def gemm(A: pto.Tensor[(M, K), pto.f16],
         B: pto.Tensor[(K, N), pto.f16],
         C: pto.Tensor[(M, N), pto.f16]):
    for m in pto.range(0, M, tile_size=16):
        for k in pto.range(0, K, tile_size=16):
            for n in pto.range(0, N, tile_size=16):
                # Load tiles
                tA = pto.tload(A[m:m+16, k:k+16])
                tB = pto.tload(B[k:k+16, n:n+16])
                tC = pto.tload(C[m:m+16, n:n+16])
                
                # Matrix multiplication
                tC = pto.tmatmul(tA, tB, tC)
                
                # Store result
                pto.tstore(C[m:m+16, n:n+16], tC)
```

## Related Documentation

| Document | Description |
|----------|-------------|
| [PTOAS submodule](../PTOAS/) | Assembler and MLIR dialect |
| [PTODSL submodule](../PTODSL/) | Python DSL for kernel authoring |
| [docs/grammar/](../docs/grammar/) | PTO-AS specification |
| [docs/bytecode/](../docs/bytecode/) | PTO-BC bytecode specification |
| [docs/coding/tutorials/](../docs/coding/tutorials/) | Step-by-step tutorials |

## External Resources

| Project | Description | Link |
|---------|-------------|------|
| **pyPTO** | Python-first frontend | [GitCode](https://gitcode.com/cann/pypto/) |
| **PTODSL** | Python DSL | [GitHub](https://github.com/huawei-csl/pto-dsl) |
| **PTOAS** | Assembler and MLIR dialect | [GitHub](https://github.com/zhangstevenunity/PTOAS) |
| **TileLang Ascend** | High-level framework | [GitHub](https://github.com/tile-ai/tilelang-ascend/) |

---

For the main project README, see: [README.md](../README.md)
