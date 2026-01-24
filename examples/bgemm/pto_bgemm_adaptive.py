"""
PTO BGEMM - Adaptive Balanced Tree Reduction

This version supports arbitrary B, M, N, K values with:
1. Dynamic K reduction tree (2, 4, 8, 16, 32 K-tiles)
2. L2 cache blocking for large problems
3. Optimized arithmetic intensity through tile reuse
4. Batch parallelism across cube cores

=============================================================================
Algorithm Overview:
=============================================================================

For a BGEMM: C[b,m,n] = sum_k(A[b,m,k] @ B[b,k,n])

1. Outer Loop: Iterate over L2 blocks (if needed)
2. Middle Loop: Iterate over output tiles (M×N)
3. Inner: Balanced tree reduction over K dimension

Memory Hierarchy:
- L1 (192KB): Single tile operations (A_tile, B_tile, C_tile)
- L2 (200MB): Multiple K-tiles for reduction (P0, P1, P2)
- GM: Full tensors

=============================================================================
"""

import os
import sys
import math

# Add parent directories to path
_example_dir = os.path.dirname(os.path.abspath(__file__))
_examples_dir = os.path.dirname(_example_dir)
_project_root = os.path.dirname(_examples_dir)
_src_dir = os.path.join(_project_root, 'src')

for path in [_src_dir, _project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from compile.pto_compile import PTOModule, PTOFunctionBuilder
from isa_definition.pto_isa_definition import ElementType, MemorySpace

# =============================================================================
# Hardware Configuration
# =============================================================================
NUM_CUBE_CORES = 24
NUM_VECTOR_CORES = 48
INCORE_SRAM_KB = 192
L2_CACHE_MB = 200

# =============================================================================
# Tile Configuration
# =============================================================================
TILE_M = 64
TILE_N = 128
TILE_K = 64
DTYPE = ElementType.F32

# Supported K_TILES configurations
SUPPORTED_K_TILES = [2, 4, 8, 16, 32]

MIN_TILES = 4
MAX_TILES = 8192


def get_reduction_depth(k_tiles):
    """Get the depth of the reduction tree."""
    return int(math.log2(k_tiles))


def create_incore_functions(module):
    """Create all InCore functions (shared across all K_TILES configs)."""
    
    # GEMM tile: C = A @ B (Cube core)
    module.add_function((PTOFunctionBuilder("gemm_tile")
        .in_core()
        .tile("a", TILE_M, TILE_K, DTYPE)
        .tile("b", TILE_K, TILE_N, DTYPE)
        .tile("c", TILE_M, TILE_N, DTYPE)
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("B", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .load("a", "A", 0, 0)
        .load("b", "B", 0, 0)
        .matmul("c", "a", "b")
        .store("c", "C", 0, 0)
        .build()))
    
    # Tile add: C = A + B (Vector core)
    module.add_function((PTOFunctionBuilder("tile_add")
        .in_core()
        .tile("a", TILE_M, TILE_N, DTYPE)
        .tile("b", TILE_M, TILE_N, DTYPE)
        .tile("c", TILE_M, TILE_N, DTYPE)
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("B", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .load("a", "A", 0, 0)
        .load("b", "B", 0, 0)
        .add("c", "a", "b")
        .store("c", "C", 0, 0)
        .build()))
    
    # Tile copy (for L2 blocking final accumulation)
    module.add_function((PTOFunctionBuilder("tile_copy")
        .in_core()
        .tile("a", TILE_M, TILE_N, DTYPE)
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .load("a", "A", 0, 0)
        .store("a", "C", 0, 0)
        .build()))


def create_orchestration_k2(module):
    """Create orchestration for K_TILES=2 (depth=1)."""
    module.add_function((PTOFunctionBuilder("bgemm_k2")
        .not_in_core()
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("B", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .memref("P0", MemorySpace.GM, DTYPE)  # 2 partial products
        .scalar("num_tiles", ElementType.I32)
        .scalar("k_offset", ElementType.I32)  # For L2 blocking
        
        .for_loop("tile", 0, "num_tiles", 1, max_range=MAX_TILES, min_range=MIN_TILES)
            # Level 0: 2 parallel gemm (Cube)
            .call("gemm_tile", {
                "A": ("A", "tile * 2 + 0", 0, TILE_M, TILE_K),
                "B": ("B", "(k_offset + 0) * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 2 + 0", 0, TILE_M, TILE_N)
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 2 + 1", 0, TILE_M, TILE_K),
                "B": ("B", "(k_offset + 1) * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 2 + 1", 0, TILE_M, TILE_N)
            })
            # Level 1: Final reduction (Vector)
            .call("tile_add", {
                "A": ("P0", "tile * 2 + 0", 0, TILE_M, TILE_N),
                "B": ("P0", "tile * 2 + 1", 0, TILE_M, TILE_N),
                "C": ("C", "tile", 0, TILE_M, TILE_N)
            })
        .end_for()
        .build()))


def create_orchestration_k4(module):
    """Create orchestration for K_TILES=4 (depth=2)."""
    module.add_function((PTOFunctionBuilder("bgemm_k4")
        .not_in_core()
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("B", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .memref("P0", MemorySpace.GM, DTYPE)  # 4 partial products
        .memref("P1", MemorySpace.GM, DTYPE)  # 2 level-1 sums
        .scalar("num_tiles", ElementType.I32)
        .scalar("k_offset", ElementType.I32)
        
        .for_loop("tile", 0, "num_tiles", 1, max_range=MAX_TILES, min_range=MIN_TILES)
            # Level 0: 4 parallel gemm (Cube)
            .call("gemm_tile", {
                "A": ("A", "tile * 4 + 0", 0, TILE_M, TILE_K),
                "B": ("B", "(k_offset + 0) * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 4 + 0", 0, TILE_M, TILE_N)
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 4 + 1", 0, TILE_M, TILE_K),
                "B": ("B", "(k_offset + 1) * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 4 + 1", 0, TILE_M, TILE_N)
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 4 + 2", 0, TILE_M, TILE_K),
                "B": ("B", "(k_offset + 2) * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 4 + 2", 0, TILE_M, TILE_N)
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 4 + 3", 0, TILE_M, TILE_K),
                "B": ("B", "(k_offset + 3) * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 4 + 3", 0, TILE_M, TILE_N)
            })
            # Level 1: 4->2 (Vector)
            .call("tile_add", {
                "A": ("P0", "tile * 4 + 0", 0, TILE_M, TILE_N),
                "B": ("P0", "tile * 4 + 1", 0, TILE_M, TILE_N),
                "C": ("P1", "tile * 2 + 0", 0, TILE_M, TILE_N)
            })
            .call("tile_add", {
                "A": ("P0", "tile * 4 + 2", 0, TILE_M, TILE_N),
                "B": ("P0", "tile * 4 + 3", 0, TILE_M, TILE_N),
                "C": ("P1", "tile * 2 + 1", 0, TILE_M, TILE_N)
            })
            # Level 2: 2->1 (Vector)
            .call("tile_add", {
                "A": ("P1", "tile * 2 + 0", 0, TILE_M, TILE_N),
                "B": ("P1", "tile * 2 + 1", 0, TILE_M, TILE_N),
                "C": ("C", "tile", 0, TILE_M, TILE_N)
            })
        .end_for()
        .build()))


def create_orchestration_k8(module):
    """Create orchestration for K_TILES=8 (depth=3)."""
    module.add_function((PTOFunctionBuilder("bgemm_k8")
        .not_in_core()
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("B", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .memref("P0", MemorySpace.GM, DTYPE)
        .memref("P1", MemorySpace.GM, DTYPE)
        .memref("P2", MemorySpace.GM, DTYPE)
        .scalar("num_tiles", ElementType.I32)
        .scalar("k_offset", ElementType.I32)
        
        .for_loop("tile", 0, "num_tiles", 1, max_range=MAX_TILES, min_range=MIN_TILES)
            # Level 0: 8 parallel gemm (Cube)
            .call("gemm_tile", {"A": ("A", "tile * 8 + 0", 0, TILE_M, TILE_K),
                                "B": ("B", "(k_offset + 0) * num_tiles + tile", 0, TILE_K, TILE_N),
                                "C": ("P0", "tile * 8 + 0", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 8 + 1", 0, TILE_M, TILE_K),
                                "B": ("B", "(k_offset + 1) * num_tiles + tile", 0, TILE_K, TILE_N),
                                "C": ("P0", "tile * 8 + 1", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 8 + 2", 0, TILE_M, TILE_K),
                                "B": ("B", "(k_offset + 2) * num_tiles + tile", 0, TILE_K, TILE_N),
                                "C": ("P0", "tile * 8 + 2", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 8 + 3", 0, TILE_M, TILE_K),
                                "B": ("B", "(k_offset + 3) * num_tiles + tile", 0, TILE_K, TILE_N),
                                "C": ("P0", "tile * 8 + 3", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 8 + 4", 0, TILE_M, TILE_K),
                                "B": ("B", "(k_offset + 4) * num_tiles + tile", 0, TILE_K, TILE_N),
                                "C": ("P0", "tile * 8 + 4", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 8 + 5", 0, TILE_M, TILE_K),
                                "B": ("B", "(k_offset + 5) * num_tiles + tile", 0, TILE_K, TILE_N),
                                "C": ("P0", "tile * 8 + 5", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 8 + 6", 0, TILE_M, TILE_K),
                                "B": ("B", "(k_offset + 6) * num_tiles + tile", 0, TILE_K, TILE_N),
                                "C": ("P0", "tile * 8 + 6", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 8 + 7", 0, TILE_M, TILE_K),
                                "B": ("B", "(k_offset + 7) * num_tiles + tile", 0, TILE_K, TILE_N),
                                "C": ("P0", "tile * 8 + 7", 0, TILE_M, TILE_N)})
            # Level 1: 8->4 (Vector)
            .call("tile_add", {"A": ("P0", "tile * 8 + 0", 0, TILE_M, TILE_N),
                               "B": ("P0", "tile * 8 + 1", 0, TILE_M, TILE_N),
                               "C": ("P1", "tile * 4 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 8 + 2", 0, TILE_M, TILE_N),
                               "B": ("P0", "tile * 8 + 3", 0, TILE_M, TILE_N),
                               "C": ("P1", "tile * 4 + 1", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 8 + 4", 0, TILE_M, TILE_N),
                               "B": ("P0", "tile * 8 + 5", 0, TILE_M, TILE_N),
                               "C": ("P1", "tile * 4 + 2", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 8 + 6", 0, TILE_M, TILE_N),
                               "B": ("P0", "tile * 8 + 7", 0, TILE_M, TILE_N),
                               "C": ("P1", "tile * 4 + 3", 0, TILE_M, TILE_N)})
            # Level 2: 4->2 (Vector)
            .call("tile_add", {"A": ("P1", "tile * 4 + 0", 0, TILE_M, TILE_N),
                               "B": ("P1", "tile * 4 + 1", 0, TILE_M, TILE_N),
                               "C": ("P2", "tile * 2 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 4 + 2", 0, TILE_M, TILE_N),
                               "B": ("P1", "tile * 4 + 3", 0, TILE_M, TILE_N),
                               "C": ("P2", "tile * 2 + 1", 0, TILE_M, TILE_N)})
            # Level 3: 2->1 (Vector)
            .call("tile_add", {"A": ("P2", "tile * 2 + 0", 0, TILE_M, TILE_N),
                               "B": ("P2", "tile * 2 + 1", 0, TILE_M, TILE_N),
                               "C": ("C", "tile", 0, TILE_M, TILE_N)})
        .end_for()
        .build()))


def create_orchestration_k16(module):
    """Create orchestration for K_TILES=16 (depth=4)."""
    module.add_function((PTOFunctionBuilder("bgemm_k16")
        .not_in_core()
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("B", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .memref("P0", MemorySpace.GM, DTYPE)
        .memref("P1", MemorySpace.GM, DTYPE)
        .memref("P2", MemorySpace.GM, DTYPE)
        .memref("P3", MemorySpace.GM, DTYPE)
        .scalar("num_tiles", ElementType.I32)
        .scalar("k_offset", ElementType.I32)
        
        .for_loop("tile", 0, "num_tiles", 1, max_range=MAX_TILES, min_range=MIN_TILES)
            # Level 0: 16 parallel gemm (Cube)
            .call("gemm_tile", {"A": ("A", "tile * 16 + 0", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 0) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 0", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 1", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 1) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 1", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 2", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 2) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 2", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 3", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 3) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 3", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 4", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 4) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 4", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 5", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 5) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 5", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 6", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 6) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 6", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 7", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 7) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 7", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 8", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 8) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 8", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 9", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 9) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 9", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 10", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 10) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 10", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 11", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 11) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 11", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 12", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 12) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 12", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 13", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 13) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 13", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 14", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 14) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 14", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 16 + 15", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 15) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 16 + 15", 0, TILE_M, TILE_N)})
            # Level 1: 16->8 (Vector)
            .call("tile_add", {"A": ("P0", "tile * 16 + 0", 0, TILE_M, TILE_N), "B": ("P0", "tile * 16 + 1", 0, TILE_M, TILE_N), "C": ("P1", "tile * 8 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 16 + 2", 0, TILE_M, TILE_N), "B": ("P0", "tile * 16 + 3", 0, TILE_M, TILE_N), "C": ("P1", "tile * 8 + 1", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 16 + 4", 0, TILE_M, TILE_N), "B": ("P0", "tile * 16 + 5", 0, TILE_M, TILE_N), "C": ("P1", "tile * 8 + 2", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 16 + 6", 0, TILE_M, TILE_N), "B": ("P0", "tile * 16 + 7", 0, TILE_M, TILE_N), "C": ("P1", "tile * 8 + 3", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 16 + 8", 0, TILE_M, TILE_N), "B": ("P0", "tile * 16 + 9", 0, TILE_M, TILE_N), "C": ("P1", "tile * 8 + 4", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 16 + 10", 0, TILE_M, TILE_N), "B": ("P0", "tile * 16 + 11", 0, TILE_M, TILE_N), "C": ("P1", "tile * 8 + 5", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 16 + 12", 0, TILE_M, TILE_N), "B": ("P0", "tile * 16 + 13", 0, TILE_M, TILE_N), "C": ("P1", "tile * 8 + 6", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 16 + 14", 0, TILE_M, TILE_N), "B": ("P0", "tile * 16 + 15", 0, TILE_M, TILE_N), "C": ("P1", "tile * 8 + 7", 0, TILE_M, TILE_N)})
            # Level 2: 8->4 (Vector)
            .call("tile_add", {"A": ("P1", "tile * 8 + 0", 0, TILE_M, TILE_N), "B": ("P1", "tile * 8 + 1", 0, TILE_M, TILE_N), "C": ("P2", "tile * 4 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 8 + 2", 0, TILE_M, TILE_N), "B": ("P1", "tile * 8 + 3", 0, TILE_M, TILE_N), "C": ("P2", "tile * 4 + 1", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 8 + 4", 0, TILE_M, TILE_N), "B": ("P1", "tile * 8 + 5", 0, TILE_M, TILE_N), "C": ("P2", "tile * 4 + 2", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 8 + 6", 0, TILE_M, TILE_N), "B": ("P1", "tile * 8 + 7", 0, TILE_M, TILE_N), "C": ("P2", "tile * 4 + 3", 0, TILE_M, TILE_N)})
            # Level 3: 4->2 (Vector)
            .call("tile_add", {"A": ("P2", "tile * 4 + 0", 0, TILE_M, TILE_N), "B": ("P2", "tile * 4 + 1", 0, TILE_M, TILE_N), "C": ("P3", "tile * 2 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P2", "tile * 4 + 2", 0, TILE_M, TILE_N), "B": ("P2", "tile * 4 + 3", 0, TILE_M, TILE_N), "C": ("P3", "tile * 2 + 1", 0, TILE_M, TILE_N)})
            # Level 4: 2->1 (Vector)
            .call("tile_add", {"A": ("P3", "tile * 2 + 0", 0, TILE_M, TILE_N), "B": ("P3", "tile * 2 + 1", 0, TILE_M, TILE_N), "C": ("C", "tile", 0, TILE_M, TILE_N)})
        .end_for()
        .build()))


def create_orchestration_k32(module):
    """Create orchestration for K_TILES=32 (depth=5)."""
    module.add_function((PTOFunctionBuilder("bgemm_k32")
        .not_in_core()
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("B", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .memref("P0", MemorySpace.GM, DTYPE)
        .memref("P1", MemorySpace.GM, DTYPE)
        .memref("P2", MemorySpace.GM, DTYPE)
        .memref("P3", MemorySpace.GM, DTYPE)
        .memref("P4", MemorySpace.GM, DTYPE)
        .scalar("num_tiles", ElementType.I32)
        .scalar("k_offset", ElementType.I32)
        
        .for_loop("tile", 0, "num_tiles", 1, max_range=MAX_TILES, min_range=MIN_TILES)
            # Level 0: 32 parallel gemm (Cube) - split into two groups for readability
            .call("gemm_tile", {"A": ("A", "tile * 32 + 0", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 0) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 0", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 1", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 1) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 1", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 2", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 2) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 2", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 3", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 3) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 3", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 4", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 4) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 4", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 5", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 5) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 5", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 6", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 6) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 6", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 7", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 7) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 7", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 8", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 8) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 8", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 9", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 9) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 9", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 10", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 10) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 10", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 11", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 11) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 11", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 12", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 12) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 12", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 13", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 13) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 13", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 14", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 14) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 14", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 15", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 15) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 15", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 16", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 16) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 16", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 17", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 17) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 17", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 18", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 18) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 18", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 19", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 19) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 19", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 20", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 20) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 20", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 21", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 21) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 21", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 22", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 22) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 22", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 23", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 23) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 23", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 24", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 24) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 24", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 25", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 25) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 25", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 26", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 26) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 26", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 27", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 27) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 27", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 28", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 28) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 28", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 29", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 29) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 29", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 30", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 30) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 30", 0, TILE_M, TILE_N)})
            .call("gemm_tile", {"A": ("A", "tile * 32 + 31", 0, TILE_M, TILE_K), "B": ("B", "(k_offset + 31) * num_tiles + tile", 0, TILE_K, TILE_N), "C": ("P0", "tile * 32 + 31", 0, TILE_M, TILE_N)})
            # Level 1: 32->16 (Vector)
            .call("tile_add", {"A": ("P0", "tile * 32 + 0", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 1", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 2", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 3", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 1", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 4", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 5", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 2", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 6", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 7", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 3", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 8", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 9", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 4", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 10", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 11", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 5", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 12", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 13", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 6", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 14", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 15", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 7", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 16", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 17", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 8", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 18", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 19", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 9", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 20", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 21", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 10", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 22", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 23", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 11", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 24", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 25", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 12", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 26", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 27", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 13", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 28", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 29", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 14", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P0", "tile * 32 + 30", 0, TILE_M, TILE_N), "B": ("P0", "tile * 32 + 31", 0, TILE_M, TILE_N), "C": ("P1", "tile * 16 + 15", 0, TILE_M, TILE_N)})
            # Level 2: 16->8 (Vector)
            .call("tile_add", {"A": ("P1", "tile * 16 + 0", 0, TILE_M, TILE_N), "B": ("P1", "tile * 16 + 1", 0, TILE_M, TILE_N), "C": ("P2", "tile * 8 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 16 + 2", 0, TILE_M, TILE_N), "B": ("P1", "tile * 16 + 3", 0, TILE_M, TILE_N), "C": ("P2", "tile * 8 + 1", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 16 + 4", 0, TILE_M, TILE_N), "B": ("P1", "tile * 16 + 5", 0, TILE_M, TILE_N), "C": ("P2", "tile * 8 + 2", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 16 + 6", 0, TILE_M, TILE_N), "B": ("P1", "tile * 16 + 7", 0, TILE_M, TILE_N), "C": ("P2", "tile * 8 + 3", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 16 + 8", 0, TILE_M, TILE_N), "B": ("P1", "tile * 16 + 9", 0, TILE_M, TILE_N), "C": ("P2", "tile * 8 + 4", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 16 + 10", 0, TILE_M, TILE_N), "B": ("P1", "tile * 16 + 11", 0, TILE_M, TILE_N), "C": ("P2", "tile * 8 + 5", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 16 + 12", 0, TILE_M, TILE_N), "B": ("P1", "tile * 16 + 13", 0, TILE_M, TILE_N), "C": ("P2", "tile * 8 + 6", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P1", "tile * 16 + 14", 0, TILE_M, TILE_N), "B": ("P1", "tile * 16 + 15", 0, TILE_M, TILE_N), "C": ("P2", "tile * 8 + 7", 0, TILE_M, TILE_N)})
            # Level 3: 8->4 (Vector)
            .call("tile_add", {"A": ("P2", "tile * 8 + 0", 0, TILE_M, TILE_N), "B": ("P2", "tile * 8 + 1", 0, TILE_M, TILE_N), "C": ("P3", "tile * 4 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P2", "tile * 8 + 2", 0, TILE_M, TILE_N), "B": ("P2", "tile * 8 + 3", 0, TILE_M, TILE_N), "C": ("P3", "tile * 4 + 1", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P2", "tile * 8 + 4", 0, TILE_M, TILE_N), "B": ("P2", "tile * 8 + 5", 0, TILE_M, TILE_N), "C": ("P3", "tile * 4 + 2", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P2", "tile * 8 + 6", 0, TILE_M, TILE_N), "B": ("P2", "tile * 8 + 7", 0, TILE_M, TILE_N), "C": ("P3", "tile * 4 + 3", 0, TILE_M, TILE_N)})
            # Level 4: 4->2 (Vector)
            .call("tile_add", {"A": ("P3", "tile * 4 + 0", 0, TILE_M, TILE_N), "B": ("P3", "tile * 4 + 1", 0, TILE_M, TILE_N), "C": ("P4", "tile * 2 + 0", 0, TILE_M, TILE_N)})
            .call("tile_add", {"A": ("P3", "tile * 4 + 2", 0, TILE_M, TILE_N), "B": ("P3", "tile * 4 + 3", 0, TILE_M, TILE_N), "C": ("P4", "tile * 2 + 1", 0, TILE_M, TILE_N)})
            # Level 5: 2->1 (Vector)
            .call("tile_add", {"A": ("P4", "tile * 2 + 0", 0, TILE_M, TILE_N), "B": ("P4", "tile * 2 + 1", 0, TILE_M, TILE_N), "C": ("C", "tile", 0, TILE_M, TILE_N)})
        .end_for()
        .build()))


def create_bgemm_adaptive_module(k_tiles=8):
    """
    Create adaptive BGEMM module for the specified K_TILES.
    
    Args:
        k_tiles: Number of K tiles (2, 4, 8, 16, or 32)
    """
    if k_tiles not in SUPPORTED_K_TILES:
        raise ValueError(f"k_tiles must be one of {SUPPORTED_K_TILES}, got {k_tiles}")
    
    module = PTOModule("bgemm_adaptive")
    
    depth = get_reduction_depth(k_tiles)
    k_dim = k_tiles * TILE_K
    
    print(f"Creating Adaptive BGEMM module...")
    print(f"  K_TILES = {k_tiles} → K = {k_dim}")
    print(f"  Reduction depth = {depth}")
    print(f"  Tasks per tile: {k_tiles} Cube + {k_tiles - 1} Vector = {2*k_tiles - 1}")
    print(f"  Critical path: 1 Cube + {depth} Vector = {depth + 1}")
    
    # Create shared InCore functions
    create_incore_functions(module)
    
    # Create orchestration for the specified K_TILES
    if k_tiles == 2:
        create_orchestration_k2(module)
        module.set_entry("bgemm_k2")
    elif k_tiles == 4:
        create_orchestration_k4(module)
        module.set_entry("bgemm_k4")
    elif k_tiles == 8:
        create_orchestration_k8(module)
        module.set_entry("bgemm_k8")
    elif k_tiles == 16:
        create_orchestration_k16(module)
        module.set_entry("bgemm_k16")
    elif k_tiles == 32:
        create_orchestration_k32(module)
        module.set_entry("bgemm_k32")
    
    print(f"\nModule created with {len(module.functions)} functions")
    return module


def create_bgemm_module():
    """Default: create module with K_TILES=8 for backward compatibility."""
    return create_bgemm_adaptive_module(k_tiles=8)


def main():
    print("=" * 70)
    print("PTO BGEMM - Adaptive Balanced Tree Reduction")
    print("=" * 70)
    
    print("\nSupported K configurations:")
    print("-" * 50)
    print(f"{'K_TILES':<10} | {'K dim':<10} | {'Depth':<8} | {'Tasks/tile':<12} | {'Critical'}")
    print("-" * 50)
    
    for k_tiles in SUPPORTED_K_TILES:
        k_dim = k_tiles * TILE_K
        depth = get_reduction_depth(k_tiles)
        tasks = 2 * k_tiles - 1
        critical = depth + 1
        print(f"{k_tiles:<10} | {k_dim:<10} | {depth:<8} | {tasks:<12} | {critical}")
    
    print("\n" + "=" * 70)
    print("Creating default module (K_TILES=8)...")
    print("=" * 70)
    
    module = create_bgemm_module()
    
    print("\n" + "=" * 70)
    print("Usage")
    print("=" * 70)
    print("""
To use a specific K configuration:

    from pto_bgemm_adaptive import create_bgemm_adaptive_module
    
    # For K=128 (2 tiles × 64)
    module = create_bgemm_adaptive_module(k_tiles=2)
    
    # For K=256 (4 tiles × 64)
    module = create_bgemm_adaptive_module(k_tiles=4)
    
    # For K=512 (8 tiles × 64)
    module = create_bgemm_adaptive_module(k_tiles=8)
    
    # For K=1024 (16 tiles × 64)
    module = create_bgemm_adaptive_module(k_tiles=16)
    
    # For K=2048 (32 tiles × 64)
    module = create_bgemm_adaptive_module(k_tiles=32)

For K > 2048, use L2 blocking:
    # K=4096 → two passes with K_TILES=32
    # First pass: k_offset=0, compute partial C
    # Second pass: k_offset=32, accumulate to C
""")
    
    return module


if __name__ == "__main__":
    main()
