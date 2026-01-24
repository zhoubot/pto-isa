"""
PTO BGEMM (Batched General Matrix Multiply) Example

This example implements a highly optimized Batched GEMM with:
- Balanced tree reduction for K accumulation
- Pipelined Cube/Vector execution
- Multi-level tiling for data reuse

Algorithm: C[b,m,n] = sum_k(A[b,m,k] @ B[b,k,n])

=============================================================================
Reduction Strategies:
=============================================================================

1. Sequential Reduction (Old):
   C = 0
   for k in K_tiles:
       C += A[k] @ B[k]   # Serial dependency chain
   
   Problem: Each gemm_acc depends on previous result
   Latency: O(K) serial steps

2. Balanced Tree Reduction (New):
   Level 0 (Cube): Compute partial products in parallel
     P[0] = A[0] @ B[0]
     P[1] = A[1] @ B[1]
     ...
     P[K-1] = A[K-1] @ B[K-1]
   
   Level 1 (Vector): Pairwise addition
     S[0] = P[0] + P[1]
     S[1] = P[2] + P[3]
     ...
   
   Level 2 (Vector): Continue reduction
     T[0] = S[0] + S[1]
     ...
   
   Final: C = result
   
   Advantage: O(log K) critical path, Cube/Vector pipeline

=============================================================================
"""

import os
import sys

# Add parent directories to path for imports
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
# Configuration Constants
# =============================================================================

# Hardware configuration (Ascend A2/A3 / 910B)
NUM_CUBE_CORES = 24
NUM_VECTOR_CORES = 48
INCORE_SRAM_KB = 192

# Tile sizes (optimized for 192KB SRAM)
TILE_M = 64       # Output rows per tile
TILE_N = 128      # Output cols per tile  
TILE_K = 64       # Reduction dimension per tile

# K tiles for reduction (K=512 / TK=64 = 8)
K_TILES = 8

# Minimum tiles for binary expansion
MIN_TILES = 8
MAX_TILES = 4096

# Data type
DTYPE = ElementType.F32


def create_bgemm_module() -> PTOModule:
    """
    Create PTO module for Batched GEMM with balanced tree reduction.
    """
    
    module = PTOModule("bgemm")
    
    print("Creating BGEMM module with balanced tree reduction...")
    print(f"  Hardware: {NUM_CUBE_CORES} cube cores, {NUM_VECTOR_CORES} vector cores")
    print(f"  Tile sizes: TM={TILE_M}, TN={TILE_N}, TK={TILE_K}")
    print(f"  K tiles: {K_TILES} (log2 = {K_TILES.bit_length()-1} reduction levels)")
    
    # =========================================================================
    # Level 1: Basic Tile Operations (InCore)
    # =========================================================================
    print("\nAdding Level 1 InCore functions...")
    
    # GEMM tile: C = A @ B (single tile, NO accumulation)
    # Used for parallel partial product computation
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
    
    # Add tiles: C = A + B (for tree reduction)
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
    
    # Copy tile (for odd reduction)
    module.add_function((PTOFunctionBuilder("tile_copy")
        .in_core()
        .tile("a", TILE_M, TILE_N, DTYPE)
        .memref("A", MemorySpace.GM, DTYPE)
        .memref("C", MemorySpace.GM, DTYPE)
        .load("a", "A", 0, 0)
        .store("a", "C", 0, 0)
        .build()))
    
    # =========================================================================
    # Level 2: Orchestration - Balanced Tree Reduction
    # =========================================================================
    print("Adding orchestration with balanced tree reduction...")
    
    # For K_TILES=8, we have 3 reduction levels:
    # Level 0: 8 partial products (Cube)
    # Level 1: 4 sums (Vector) 
    # Level 2: 2 sums (Vector)
    # Level 3: 1 final sum (Vector)
    
    # We need temporary buffers for partial results
    # P0[0..7]: partial products from Cube
    # P1[0..3]: level 1 sums from Vector
    # P2[0..1]: level 2 sums from Vector
    # C: final result
    
    module.add_function((PTOFunctionBuilder("bgemm_dynamic")
        .not_in_core()
        .memref("A", MemorySpace.GM, DTYPE)      # Input A
        .memref("B", MemorySpace.GM, DTYPE)      # Input B
        .memref("C", MemorySpace.GM, DTYPE)      # Output C
        .memref("P0", MemorySpace.GM, DTYPE)     # Partial products (8 per tile)
        .memref("P1", MemorySpace.GM, DTYPE)     # Level 1 reduction (4 per tile)
        .memref("P2", MemorySpace.GM, DTYPE)     # Level 2 reduction (2 per tile)
        .scalar("seq_len", ElementType.I32)
        .scalar("tile_rows", ElementType.I32)
        .scalar("num_tiles", ElementType.I32)
        .scalar("zero", DTYPE)
        
        # Main loop over output tiles
        .for_loop("tile", 0, "num_tiles", 1, max_range=MAX_TILES, min_range=MIN_TILES)
            
            # =================================================================
            # Level 0: Compute 8 partial products in PARALLEL (Cube cores)
            # =================================================================
            # These can all run in parallel since they write to different P0 slots
            
            .call("gemm_tile", {
                "A": ("A", "tile * 8 + 0", 0, TILE_M, TILE_K),
                "B": ("B", "0 * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 8 + 0", 0, TILE_M, TILE_N)  # P0[0]
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 8 + 1", 0, TILE_M, TILE_K),
                "B": ("B", "1 * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 8 + 1", 0, TILE_M, TILE_N)  # P0[1]
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 8 + 2", 0, TILE_M, TILE_K),
                "B": ("B", "2 * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 8 + 2", 0, TILE_M, TILE_N)  # P0[2]
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 8 + 3", 0, TILE_M, TILE_K),
                "B": ("B", "3 * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 8 + 3", 0, TILE_M, TILE_N)  # P0[3]
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 8 + 4", 0, TILE_M, TILE_K),
                "B": ("B", "4 * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 8 + 4", 0, TILE_M, TILE_N)  # P0[4]
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 8 + 5", 0, TILE_M, TILE_K),
                "B": ("B", "5 * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 8 + 5", 0, TILE_M, TILE_N)  # P0[5]
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 8 + 6", 0, TILE_M, TILE_K),
                "B": ("B", "6 * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 8 + 6", 0, TILE_M, TILE_N)  # P0[6]
            })
            .call("gemm_tile", {
                "A": ("A", "tile * 8 + 7", 0, TILE_M, TILE_K),
                "B": ("B", "7 * num_tiles + tile", 0, TILE_K, TILE_N),
                "C": ("P0", "tile * 8 + 7", 0, TILE_M, TILE_N)  # P0[7]
            })
            
            # =================================================================
            # Level 1: Reduce 8 -> 4 (Vector cores)
            # =================================================================
            # P1[0] = P0[0] + P0[1]  (depends on P0[0], P0[1])
            # P1[1] = P0[2] + P0[3]  (depends on P0[2], P0[3])
            # P1[2] = P0[4] + P0[5]  (depends on P0[4], P0[5])
            # P1[3] = P0[6] + P0[7]  (depends on P0[6], P0[7])
            # These can start as soon as their inputs are ready!
            
            .call("tile_add", {
                "A": ("P0", "tile * 8 + 0", 0, TILE_M, TILE_N),
                "B": ("P0", "tile * 8 + 1", 0, TILE_M, TILE_N),
                "C": ("P1", "tile * 4 + 0", 0, TILE_M, TILE_N)
            })
            .call("tile_add", {
                "A": ("P0", "tile * 8 + 2", 0, TILE_M, TILE_N),
                "B": ("P0", "tile * 8 + 3", 0, TILE_M, TILE_N),
                "C": ("P1", "tile * 4 + 1", 0, TILE_M, TILE_N)
            })
            .call("tile_add", {
                "A": ("P0", "tile * 8 + 4", 0, TILE_M, TILE_N),
                "B": ("P0", "tile * 8 + 5", 0, TILE_M, TILE_N),
                "C": ("P1", "tile * 4 + 2", 0, TILE_M, TILE_N)
            })
            .call("tile_add", {
                "A": ("P0", "tile * 8 + 6", 0, TILE_M, TILE_N),
                "B": ("P0", "tile * 8 + 7", 0, TILE_M, TILE_N),
                "C": ("P1", "tile * 4 + 3", 0, TILE_M, TILE_N)
            })
            
            # =================================================================
            # Level 2: Reduce 4 -> 2 (Vector cores)
            # =================================================================
            # P2[0] = P1[0] + P1[1]
            # P2[1] = P1[2] + P1[3]
            
            .call("tile_add", {
                "A": ("P1", "tile * 4 + 0", 0, TILE_M, TILE_N),
                "B": ("P1", "tile * 4 + 1", 0, TILE_M, TILE_N),
                "C": ("P2", "tile * 2 + 0", 0, TILE_M, TILE_N)
            })
            .call("tile_add", {
                "A": ("P1", "tile * 4 + 2", 0, TILE_M, TILE_N),
                "B": ("P1", "tile * 4 + 3", 0, TILE_M, TILE_N),
                "C": ("P2", "tile * 2 + 1", 0, TILE_M, TILE_N)
            })
            
            # =================================================================
            # Level 3: Final reduction 2 -> 1 (Vector core)
            # =================================================================
            # C[tile] = P2[0] + P2[1]
            
            .call("tile_add", {
                "A": ("P2", "tile * 2 + 0", 0, TILE_M, TILE_N),
                "B": ("P2", "tile * 2 + 1", 0, TILE_M, TILE_N),
                "C": ("C", "tile", 0, TILE_M, TILE_N)
            })
            
        .end_for()
        .build()))
    
    # Set entry point
    module.set_entry("bgemm_dynamic")
    
    print(f"\nModule created with {len(module.functions)} functions:")
    for name, prog in module.functions.items():
        func_type = "Orchestration" if not getattr(prog, 'is_in_core', True) else "InCore"
        is_cube = "Cube" if getattr(prog, 'is_cube', False) else "Vector"
        print(f"  - {name}: {func_type} ({is_cube})")
    
    return module


def main():
    """Main entry point."""
    print("=" * 70)
    print("PTO BGEMM - Balanced Tree Reduction")
    print("=" * 70)
    print()
    
    module = create_bgemm_module()
    
    print()
    print("=" * 70)
    print("Algorithm: Balanced Tree Reduction")
    print("=" * 70)
    print(f"""
For K={K_TILES} partial products:

Depth 0 (Cube, parallel):  8 gemm_tile operations
  P0[0] = A[0] @ B[0]    |    P0[4] = A[4] @ B[4]
  P0[1] = A[1] @ B[1]    |    P0[5] = A[5] @ B[5]
  P0[2] = A[2] @ B[2]    |    P0[6] = A[6] @ B[6]
  P0[3] = A[3] @ B[3]    |    P0[7] = A[7] @ B[7]

Depth 1 (Vector, parallel): 4 tile_add operations
  P1[0] = P0[0] + P0[1]  |    P1[2] = P0[4] + P0[5]
  P1[1] = P0[2] + P0[3]  |    P1[3] = P0[6] + P0[7]

Depth 2 (Vector, parallel): 2 tile_add operations
  P2[0] = P1[0] + P1[1]  |    P2[1] = P1[2] + P1[3]

Depth 3 (Vector): 1 tile_add operation
  C = P2[0] + P2[1]

Total tasks per tile: 8 Cube + 7 Vector = 15 tasks
Critical path: 1 Cube + 3 Vector = 4 tasks (vs 9 for sequential)
Parallelism: Up to 8 Cube tasks + 4 Vector tasks simultaneously

Pipeline advantage:
  - Vector Level 1 can start when Cube pairs complete
  - Vector Level 2 can start when Level 1 pairs complete
  - Full overlap between Cube and Vector execution
""")
    
    print("Run with:")
    print("  python config_example.py --example bgemm --platform ascend_a2a3_sim --run")
    print()
    
    return module


if __name__ == "__main__":
    main()
