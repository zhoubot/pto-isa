"""
PTO ISA Example: sinh() using Taylor Expansion with Dynamic Tiling

This file demonstrates the PTO Python DSL for computing sinh(x) on tiles
with dynamic input tensor sizes.

Taylor expansion for sinh(x):
    sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + ...
            = x + x³/6 + x⁵/120 + x⁷/5040 + ...

Tile Shape Computation Rules:
1) col should be multiples of VECTOR_LANES of the given physical ISA
2) row should be multiple of PHYSICAL_ROW_SIZE
3) byte size of the TILE should be no greater than 16KB

Algorithm:
    - Compute optimal tile shape based on data type and target ISA
    - Use scalar control loop to iterate over input tensor
    - Handle tail tiles that don't exactly match tile size
"""

import os
import sys

# Add parent directory for imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_compile import PTOFunctionBuilder, PTOCompiler
from pto_isa_definition import (
    ElementType, MemorySpace, CompareMode,
    ARM64_VECTOR_LANES, ARM64_PHYSICAL_ROW_SIZE,
    CUDA_VECTOR_LANES, CUDA_PHYSICAL_ROW_SIZE,
    ASCEND_VECTOR_LANES, ASCEND_PHYSICAL_ROW_SIZE,
)
from pto_dynamic_tiling import (
    compute_tile_shape, get_tile_info, 
    MAX_TILE_BYTES, ELEMENT_BYTES,
)


# =============================================================================
# Tile Shape Computation
# =============================================================================

# Maximum tile size in bytes (16KB)
MAX_TILE_BYTES = 16 * 1024

# Element size in bytes
ELEMENT_BYTES = {
    ElementType.F32: 4,
    ElementType.F16: 2,
    ElementType.F64: 8,
    ElementType.BF16: 2,
    ElementType.I32: 4,
    ElementType.I8: 1,
    ElementType.I16: 2,
    ElementType.I64: 8,
}

def compute_tile_shape(dtype: ElementType, target_isa: str = "arm64") -> tuple:
    """
    Compute optimal tile shape based on data type and target ISA.
    
    Rules:
    1) col should be multiples of VECTOR_LANES
    2) row should be multiple of PHYSICAL_ROW_SIZE
    3) byte size of the TILE should be no greater than 16KB
    
    Returns:
        (rows, cols) tuple
    """
    dtype_str = dtype.value
    
    # Get ISA-specific parameters
    if target_isa == "arm64":
        vector_lanes = ARM64_VECTOR_LANES.get(dtype_str, 4)
        physical_row_size = ARM64_PHYSICAL_ROW_SIZE
    elif target_isa == "cuda":
        vector_lanes = CUDA_VECTOR_LANES.get(dtype_str, 4)
        physical_row_size = CUDA_PHYSICAL_ROW_SIZE
    elif target_isa in ("ascend_a2a3", "ascend_a5", "ascend910b"):
        vector_lanes = ASCEND_VECTOR_LANES.get(dtype_str, 8)
        physical_row_size = ASCEND_PHYSICAL_ROW_SIZE
    else:
        # Default to ARM64
        vector_lanes = ARM64_VECTOR_LANES.get(dtype_str, 4)
        physical_row_size = ARM64_PHYSICAL_ROW_SIZE
    
    element_bytes = ELEMENT_BYTES.get(dtype, 4)
    
    # Start with col = vector_lanes (minimum aligned column count)
    # Try to maximize cols as multiples of vector_lanes while staying under 16KB
    
    # Maximum elements that fit in 16KB
    max_elements = MAX_TILE_BYTES // element_bytes
    
    # Start with a reasonable row count based on physical_row_size
    # For Ascend, we want 32 rows; for ARM64/CUDA, we want 1 row minimum
    # but we'll try to increase rows to fill the tile size
    
    # Strategy: Use cols as multiple of vector_lanes
    # Compute how many columns we can have
    # cols = N * vector_lanes, where N is a power of 2 for alignment
    
    # Try different row configurations
    best_rows = physical_row_size
    best_cols = vector_lanes
    best_total = best_rows * best_cols
    
    for row_mult in [1, 2, 4, 8, 16, 32, 64, 128]:
        rows = physical_row_size * row_mult
        
        # Compute max cols for this row count
        max_cols_for_rows = max_elements // rows
        
        # Round down to multiple of vector_lanes
        cols = (max_cols_for_rows // vector_lanes) * vector_lanes
        
        if cols < vector_lanes:
            break  # Too many rows, can't fit even one vector width
        
        total = rows * cols
        if total > best_total and total * element_bytes <= MAX_TILE_BYTES:
            best_rows = rows
            best_cols = cols
            best_total = total
    
    return best_rows, best_cols


# =============================================================================
# Build sinh() PTO Program with Dynamic Tiling
# =============================================================================

def build_sinh_program_dynamic(dtype: ElementType = ElementType.F32,
                                target_isa: str = "arm64"):
    """
    Build the sinh() computation using PTO Python DSL with dynamic tiling.
    
    sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + ...
    
    Uses 7 terms of Taylor expansion for good accuracy.
    
    The program uses scalar control loops to:
    1. Iterate over the input tensor in tile-sized chunks
    2. Handle the tail tile that may not fill the full tile size
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    print(f"  Target ISA: {target_isa}")
    print(f"  Data type: {dtype.value}")
    print(f"  Tile shape: {rows}x{cols} = {tile_elements} elements")
    print(f"  Tile bytes: {tile_elements * ELEMENT_BYTES.get(dtype, 4)} bytes")
    
    program = (PTOFunctionBuilder("sinh_taylor")
        # ====================================================================
        # Tile Declarations
        # ====================================================================
        .tile("x", rows, cols, dtype)              # Input tile
        .tile("x_squared", rows, cols, dtype)      # x²
        .tile("term", rows, cols, dtype)           # Current Taylor term
        .tile("result", rows, cols, dtype)         # Accumulated result
        
        # Memory references (dynamic size - no shape specified)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        # Scalar declarations for loop control
        .scalar("total_elements", ElementType.I32)  # Total input elements
        .scalar("tile_size", ElementType.I32)       # Elements per tile
        .scalar("num_full_tiles", ElementType.I32)  # Number of full tiles
        .scalar("tail_elements", ElementType.I32)   # Remaining elements
        .scalar("offset", ElementType.I32)          # Current offset in elements
        .scalar("has_tail", ElementType.U1)         # Whether tail exists
        .scalar("zero", ElementType.I32)            # Constant zero for comparison
        
        # Initialize constants
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        # ====================================================================
        # Main Loop: Process full tiles
        # FOR offset = 0 to num_full_tiles * tile_size, step = tile_size
        # ====================================================================
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            # Compute offset = tile_idx * tile_size
            # Note: In real implementation, we'd need scalar multiply
            # For now, we use tile_idx directly as offset multiplier
            
            # Load tile from input[offset]
            .load("x", "input", "tile_idx", 0)
            
            # ================================================================
            # sinh Taylor expansion computation
            # ================================================================
            # result = x (first term of Taylor series)
            .muls("result", "x", 1.0)
            
            # x_squared = x * x
            .mul("x_squared", "x", "x")
            
            # term = x (starting term)
            .muls("term", "x", 1.0)
            
            # Term 2: x³/3! = x³/6
            .mul("term", "term", "x_squared")  # term = x * x² = x³
            .divs("term", "term", 6.0)         # term = x³/6
            .add("result", "result", "term")   # result = x + x³/6
            
            # Term 3: x⁵/5! = x⁵/120
            .mul("term", "term", "x_squared")  # term = x⁵/6
            .divs("term", "term", 20.0)        # term = x⁵/120
            .add("result", "result", "term")   # result += x⁵/120
            
            # Term 4: x⁷/7! = x⁷/5040
            .mul("term", "term", "x_squared")  # term = x⁷/120
            .divs("term", "term", 42.0)        # term = x⁷/5040
            .add("result", "result", "term")   # result += x⁷/5040
            
            # Term 5: x⁹/9! = x⁹/362880
            .mul("term", "term", "x_squared")
            .divs("term", "term", 72.0)
            .add("result", "result", "term")
            
            # Term 6: x¹¹/11!
            .mul("term", "term", "x_squared")
            .divs("term", "term", 110.0)
            .add("result", "result", "term")
            
            # Term 7: x¹³/13!
            .mul("term", "term", "x_squared")
            .divs("term", "term", 156.0)
            .add("result", "result", "term")
            
            # Store result to output[offset]
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        # ====================================================================
        # Handle Tail: Process remaining elements (if any)
        # ====================================================================
        # Check if tail_elements > 0
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            # Load partial tile (hardware handles masking)
            .load("x", "input", "num_full_tiles", 0)
            
            # sinh Taylor expansion (same computation)
            .muls("result", "x", 1.0)
            .mul("x_squared", "x", "x")
            .muls("term", "x", 1.0)
            
            .mul("term", "term", "x_squared")
            .divs("term", "term", 6.0)
            .add("result", "result", "term")
            
            .mul("term", "term", "x_squared")
            .divs("term", "term", 20.0)
            .add("result", "result", "term")
            
            .mul("term", "term", "x_squared")
            .divs("term", "term", 42.0)
            .add("result", "result", "term")
            
            .mul("term", "term", "x_squared")
            .divs("term", "term", 72.0)
            .add("result", "result", "term")
            
            .mul("term", "term", "x_squared")
            .divs("term", "term", 110.0)
            .add("result", "result", "term")
            
            .mul("term", "term", "x_squared")
            .divs("term", "term", 156.0)
            .add("result", "result", "term")
            
            # Store partial result (hardware handles masking)
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())
    
    return program


# =============================================================================
# Main: Generate Multi-Backend Code
# =============================================================================

if __name__ == "__main__":
    import os
    import sys
    
    # Add parent directory for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from pto_compile import generate_all_backends, generate_arm64_code, BACKENDS
    
    print("=" * 70)
    print("PTO ISA sinh() - Dynamic Tiling Multi-Backend Code Generation")
    print("=" * 70)
    
    # Base output directory  
    OUTPUT_PREFIX = "sinh_taylor"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Build programs for different ISAs
    for isa in ["arm64", "cuda", "ascend_a2a3", "ascend_a5"]:
        print(f"\n{'='*70}")
        print(f"Building sinh() program for {isa.upper()}")
        print(f"{'='*70}")
        
        program = build_sinh_program_dynamic(ElementType.F32, isa)
        
        print(f"\nGenerating code...")
        results = generate_all_backends(
            program, 
            OUTPUT_PREFIX,
            output_base_dir=SCRIPT_DIR,
            enable_fusion=True
        )
    
    # =========================================================================
    # Show tile shape for different ISAs and data types
    # =========================================================================
    print("\n" + "=" * 70)
    print("Tile Shape Summary for Different ISAs and Data Types")
    print("=" * 70)
    
    for isa in ["arm64", "cuda", "ascend_a2a3", "ascend_a5"]:
        print(f"\n{isa.upper()}:")
        for dtype in [ElementType.F32, ElementType.F16, ElementType.F64]:
            rows, cols = compute_tile_shape(dtype, isa)
            tile_bytes = rows * cols * ELEMENT_BYTES[dtype]
            print(f"  {dtype.value}: {rows}x{cols} = {rows*cols} elements ({tile_bytes} bytes)")
    
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print(f"Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{OUTPUT_PREFIX}/")
    print(f"  - output_pto/{OUTPUT_PREFIX}/")
    print("=" * 70)
