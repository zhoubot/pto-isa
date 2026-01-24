#!/usr/bin/env python3
"""
BGEMM Performance Analysis

Analyze the balanced tree reduction algorithm's performance across
different B, M, K, N configurations.
"""

import math
import sys

# =============================================================================
# Hardware Configuration (Ascend 910B)
# =============================================================================
NUM_CUBE_CORES = 24
NUM_VECTOR_CORES = 48
INCORE_SRAM_KB = 192
L2_CACHE_MB = 200

# Current implementation parameters
TILE_M = 64
TILE_N = 128
TILE_K = 64

# Adaptive version supports multiple K_TILES
SUPPORTED_K_TILES = [2, 4, 8, 16, 32]
K_TILES_FIXED = 8  # Default value

# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_problem(B, M, N, K, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K):
    """
    Analyze a single problem configuration.
    
    Returns dict with performance metrics and potential issues.
    """
    # Calculate tile counts
    m_tiles = math.ceil(M / tile_m)
    n_tiles = math.ceil(N / tile_n)
    k_tiles = math.ceil(K / tile_k)
    
    # Round k_tiles to power of 2 for balanced tree
    k_tiles_pow2 = 2 ** math.ceil(math.log2(max(k_tiles, 2)))
    reduction_depth = int(math.log2(k_tiles_pow2))
    
    # Output tiles = B × m_tiles × n_tiles
    output_tiles = B * m_tiles * n_tiles
    
    # Tasks per output tile
    cube_tasks = k_tiles_pow2
    vector_tasks = k_tiles_pow2 - 1
    tasks_per_tile = cube_tasks + vector_tasks
    
    # Total tasks
    total_cube = output_tiles * cube_tasks
    total_vector = output_tiles * vector_tasks
    total_tasks = total_cube + total_vector
    
    # Critical path length
    critical_path = 1 + reduction_depth  # 1 cube + log2(k) vector
    
    # Theoretical speedup vs sequential
    sequential_depth = 2 * k_tiles_pow2 - 1  # k matmuls + k-1 adds
    speedup = sequential_depth / critical_path
    
    # Memory analysis (per tile)
    bytes_per_element = 4  # float32
    mem_a_tile = tile_m * tile_k * bytes_per_element
    mem_b_tile = tile_k * tile_n * bytes_per_element
    mem_c_tile = tile_m * tile_n * bytes_per_element
    
    # Intermediate buffers for one output tile
    mem_p0 = k_tiles_pow2 * mem_c_tile  # All partial products
    mem_p1 = (k_tiles_pow2 // 2) * mem_c_tile
    mem_p2 = (k_tiles_pow2 // 4) * mem_c_tile if k_tiles_pow2 >= 4 else 0
    mem_intermediate = mem_p0 + mem_p1 + mem_p2
    
    # L1 check: Can a single tile operation fit in SRAM?
    l1_per_op = mem_a_tile + mem_b_tile + mem_c_tile
    fits_l1 = l1_per_op <= INCORE_SRAM_KB * 1024
    
    # L2 check: Can intermediate buffers for all tiles fit?
    total_intermediate = output_tiles * mem_intermediate
    fits_l2 = total_intermediate <= L2_CACHE_MB * 1024 * 1024
    
    # Parallelism analysis
    max_cube_parallel = min(output_tiles * k_tiles_pow2, NUM_CUBE_CORES)
    max_vector_parallel = min(output_tiles * (k_tiles_pow2 // 2), NUM_VECTOR_CORES)
    
    # Compute/memory ratio (arithmetic intensity)
    flops_per_tile = 2 * tile_m * tile_n * tile_k * k_tiles_pow2  # matmuls
    flops_per_tile += tile_m * tile_n * (k_tiles_pow2 - 1)  # additions
    bytes_per_tile = (k_tiles_pow2 * (mem_a_tile + mem_b_tile) +  # inputs
                      mem_c_tile +  # output
                      mem_intermediate)  # intermediates
    arithmetic_intensity = flops_per_tile / bytes_per_tile
    
    # Issues detection
    issues = []
    
    # Check if k_tiles_pow2 is supported by adaptive version
    if k_tiles_pow2 not in SUPPORTED_K_TILES:
        if k_tiles_pow2 < min(SUPPORTED_K_TILES):
            issues.append(f"K too small: needs {k_tiles_pow2} tiles, min supported is {min(SUPPORTED_K_TILES)} (K={K})")
        elif k_tiles_pow2 > max(SUPPORTED_K_TILES):
            issues.append(f"K too large: needs {k_tiles_pow2} tiles, max supported is {max(SUPPORTED_K_TILES)} (K={K}, need L2 blocking)")
    
    if not fits_l1:
        issues.append(f"L1 overflow: {l1_per_op/1024:.1f}KB > {INCORE_SRAM_KB}KB")
    
    if not fits_l2:
        issues.append(f"L2 overflow: {total_intermediate/1024/1024:.1f}MB > {L2_CACHE_MB}MB")
    
    if k_tiles_pow2 > 32:
        issues.append(f"Deep reduction tree: {reduction_depth} levels may have high overhead")
    
    if output_tiles < NUM_CUBE_CORES:
        issues.append(f"Low parallelism: {output_tiles} tiles < {NUM_CUBE_CORES} cube cores")
    
    if arithmetic_intensity < 10:
        issues.append(f"Memory bound: AI={arithmetic_intensity:.1f} FLOPS/byte (want >50)")
    
    return {
        'config': {'B': B, 'M': M, 'N': N, 'K': K},
        'tiles': {
            'm_tiles': m_tiles,
            'n_tiles': n_tiles,
            'k_tiles': k_tiles,
            'k_tiles_pow2': k_tiles_pow2,
            'output_tiles': output_tiles,
        },
        'tasks': {
            'cube_per_tile': cube_tasks,
            'vector_per_tile': vector_tasks,
            'total_cube': total_cube,
            'total_vector': total_vector,
            'total': total_tasks,
        },
        'performance': {
            'reduction_depth': reduction_depth,
            'critical_path': critical_path,
            'speedup_vs_seq': speedup,
            'max_cube_parallel': max_cube_parallel,
            'max_vector_parallel': max_vector_parallel,
            'arithmetic_intensity': arithmetic_intensity,
        },
        'memory': {
            'l1_per_op_kb': l1_per_op / 1024,
            'intermediate_per_tile_kb': mem_intermediate / 1024,
            'total_intermediate_mb': total_intermediate / 1024 / 1024,
            'fits_l1': fits_l1,
            'fits_l2': fits_l2,
        },
        'issues': issues,
        'supported': len(issues) == 0 or (len(issues) == 1 and 'K mismatch' not in issues[0]),
    }


def print_analysis(result):
    """Print analysis results."""
    cfg = result['config']
    print(f"\nB={cfg['B']}, M={cfg['M']}, N={cfg['N']}, K={cfg['K']}")
    print("-" * 60)
    
    tiles = result['tiles']
    print(f"  Tiles: {tiles['m_tiles']}×{tiles['n_tiles']} spatial, {tiles['k_tiles']} K-tiles → {tiles['k_tiles_pow2']} (pow2)")
    print(f"  Output tiles: {tiles['output_tiles']}")
    
    tasks = result['tasks']
    print(f"  Tasks: {tasks['total']:,} total ({tasks['total_cube']:,} cube + {tasks['total_vector']:,} vector)")
    
    perf = result['performance']
    print(f"  Reduction depth: {perf['reduction_depth']} levels")
    print(f"  Critical path: {perf['critical_path']} (speedup: {perf['speedup_vs_seq']:.1f}x vs sequential)")
    print(f"  Parallelism: up to {perf['max_cube_parallel']} cube, {perf['max_vector_parallel']} vector")
    print(f"  Arithmetic intensity: {perf['arithmetic_intensity']:.1f} FLOPS/byte")
    
    mem = result['memory']
    print(f"  Memory: L1={mem['l1_per_op_kb']:.1f}KB/op, intermediate={mem['total_intermediate_mb']:.1f}MB")
    print(f"  Fits L1: {'✓' if mem['fits_l1'] else '✗'}, Fits L2: {'✓' if mem['fits_l2'] else '✗'}")
    
    if result['issues']:
        print(f"  ⚠️  Issues:")
        for issue in result['issues']:
            print(f"      - {issue}")
    else:
        print(f"  ✓ No issues detected")


def main():
    print("=" * 70)
    print("BGEMM Balanced Tree Reduction - Performance Analysis")
    print("=" * 70)
    print(f"\nHardware: {NUM_CUBE_CORES} cube cores, {NUM_VECTOR_CORES} vector cores")
    print(f"L1 SRAM: {INCORE_SRAM_KB}KB, L2 Cache: {L2_CACHE_MB}MB")
    print(f"Current impl: TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}, K_TILES={K_TILES_FIXED}")
    print(f"Supported K: {K_TILES_FIXED * TILE_K} (fixed)")
    
    # Test cases representing different workloads
    test_cases = [
        # (B, M, N, K) - Description
        # Small problems
        (1, 64, 128, 512),       # Single tile, K=512 (supported)
        (1, 256, 256, 512),      # 4×2 tiles, K=512 (supported)
        
        # Medium problems (typical ML)
        (1, 512, 512, 512),      # 8×4 tiles, K=512 (supported)
        (1, 1024, 1024, 512),    # 16×8 tiles, K=512 (supported)
        (1, 1024, 1024, 1024),   # K=1024 (needs 16 K-tiles)
        
        # Large problems (LLaMA-like)
        (1, 4096, 4096, 512),    # Large M,N, K=512 (supported)
        (1, 4096, 4096, 4096),   # Large all dims (K needs 64 tiles)
        (1, 4096, 11008, 4096),  # LLaMA MLP gate proj
        
        # Batched problems
        (8, 512, 512, 512),      # 8 batches
        (32, 256, 256, 512),     # 32 batches, smaller matrices
        (64, 128, 128, 512),     # 64 batches, small matrices
        
        # Edge cases
        (1, 64, 64, 64),         # Very small (K=1 tile)
        (1, 64, 64, 2048),       # Small spatial, large K
        (1, 8192, 8192, 256),    # Large spatial, small K
    ]
    
    print("\n" + "=" * 70)
    print("Analysis Results")
    print("=" * 70)
    
    supported_count = 0
    for B, M, N, K in test_cases:
        result = analyze_problem(B, M, N, K)
        print_analysis(result)
        if result['supported']:
            supported_count += 1
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nTested {len(test_cases)} configurations")
    print(f"  Fully supported: {supported_count}")
    print(f"  Has issues: {len(test_cases) - supported_count}")
    
    print("\n" + "=" * 70)
    print("Current Implementation Limitations")
    print("=" * 70)
    print("""
1. K Dimension Fixed:
   - Only K=512 supported (K_TILES=8 hardcoded)
   - Needs dynamic K_TILES selection for K=256, 1024, 2048, 4096, ...

2. Tile Sizes Fixed:
   - TILE_M=64, TILE_N=128, TILE_K=64 may not be optimal
   - Small problems: tiles too large → low utilization
   - Large problems: may need L2 blocking

3. Batch Not Explicit:
   - B is folded into spatial tiles, not separate dimension
   - May miss batch-level parallelism opportunities

4. Memory Buffers:
   - P0, P1, P2 sizes fixed for K_TILES=8
   - Different K needs different buffer counts

5. No L2 Blocking:
   - Large problems overflow L2 cache
   - Need hierarchical tiling for K > 2048
""")
    
    print("\n" + "=" * 70)
    print("Recommended Improvements")
    print("=" * 70)
    print("""
1. Dynamic K_TILES:
   - Accept K as runtime parameter
   - Generate reduction tree based on actual K
   - Support K_TILES = 2, 4, 8, 16, 32, 64

2. Adaptive Tile Sizes:
   - Small K (≤256): Use smaller tiles
   - Large M,N: Use larger tiles for better cache reuse
   - Profile to find optimal per problem class

3. Explicit Batch Parallelism:
   - Distribute batches across cube cores first
   - Each core handles subset of batches

4. L2 Cache Blocking:
   - For K > 2048: Split into L2-sized chunks
   - Partial tree reduction per chunk
   - Final reduction across chunks

5. Memory Optimization:
   - Double buffering for P0 (pingpong)
   - Reuse P1/P2 buffers across tiles
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
