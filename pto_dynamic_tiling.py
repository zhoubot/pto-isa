"""
PTO Dynamic Tiling Helper Module

This module provides shared utilities for dynamic tensor shape handling in PTO programs.

Tile Shape Computation Rules:
1) col should be multiples of VECTOR_LANES of the given physical ISA
2) row should be multiple of PHYSICAL_ROW_SIZE
3) byte size of the TILE should be no greater than 16KB

The module provides:
- compute_tile_shape(): Calculate optimal tile shape for given dtype and ISA
- DynamicTiledProgram: Helper class to build programs with dynamic tiling
"""

from pto_compile import PTOFunctionBuilder
from pto_isa_definition import (
    ElementType, MemorySpace, CompareMode,
    ARM64_VECTOR_LANES, ARM64_PHYSICAL_ROW_SIZE,
    CUDA_VECTOR_LANES, CUDA_PHYSICAL_ROW_SIZE,
    ASCEND_VECTOR_LANES, ASCEND_PHYSICAL_ROW_SIZE,
)

# Re-export ISA constants for convenience
__all__ = [
    'compute_tile_shape', 'get_tile_info', 'DynamicTiledProgram',
    'build_unary_op', 'build_binary_op', 'build_scalar_op',
    'print_tile_shapes',
    'MAX_TILE_BYTES', 'ELEMENT_BYTES', 'DEFAULT_DTYPE',
    'ARM64_VECTOR_LANES', 'ARM64_PHYSICAL_ROW_SIZE',
    'CUDA_VECTOR_LANES', 'CUDA_PHYSICAL_ROW_SIZE',
    'ASCEND_VECTOR_LANES', 'ASCEND_PHYSICAL_ROW_SIZE',
]


# =============================================================================
# Constants
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

# Default data type
DEFAULT_DTYPE = ElementType.F32


# =============================================================================
# Tile Shape Computation
# =============================================================================

def compute_tile_shape(dtype: ElementType = ElementType.F32, 
                       target_isa: str = "arm64") -> tuple:
    """
    Compute optimal tile shape based on data type and target ISA.
    
    Rules:
    1) col should be multiples of VECTOR_LANES
    2) row should be multiple of PHYSICAL_ROW_SIZE
    3) byte size of the TILE should be no greater than 16KB
    
    Args:
        dtype: Element data type
        target_isa: Target ISA ("arm64", "cuda", "ascend_a2a3", "ascend_a5")
    
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
    
    # Maximum elements that fit in 16KB
    max_elements = MAX_TILE_BYTES // element_bytes
    
    # Find best tile configuration
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


def get_tile_info(dtype: ElementType = ElementType.F32,
                  target_isa: str = "arm64") -> dict:
    """
    Get tile information including shape and element count.
    
    Returns:
        dict with 'rows', 'cols', 'elements', 'bytes'
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    elements = rows * cols
    bytes_size = elements * ELEMENT_BYTES.get(dtype, 4)
    return {
        'rows': rows,
        'cols': cols,
        'elements': elements,
        'bytes': bytes_size
    }


# =============================================================================
# Dynamic Tiling Program Builder Helper
# =============================================================================

class DynamicTiledProgram:
    """
    Helper class to build PTO programs with dynamic tiling support.
    
    This wraps the basic tile computation with:
    - Scalar control loop to iterate over tiles
    - Tail tile handling for non-aligned tensor sizes
    
    Example usage:
        builder = DynamicTiledProgram("my_relu", dtype=ElementType.F32, target_isa="arm64")
        builder.add_input("input")
        builder.add_output("output")
        builder.add_tile("x")
        builder.add_tile("result")
        
        # Define computation (applied to each tile)
        def compute(b):
            return (b
                .load("x", "input", "tile_idx", 0)
                .relu("result", "x")
                .store("result", "output", "tile_idx", 0))
        
        program = builder.build(compute)
    """
    
    def __init__(self, name: str, dtype: ElementType = ElementType.F32, 
                 target_isa: str = "arm64"):
        self.name = name
        self.dtype = dtype
        self.target_isa = target_isa
        self.rows, self.cols = compute_tile_shape(dtype, target_isa)
        self.tile_elements = self.rows * self.cols
        
        self.inputs = []  # List of input memref names
        self.outputs = []  # List of output memref names
        self.tiles = []  # List of tile names
        self.extra_tiles = []  # Tiles with custom shapes
    
    def add_input(self, name: str) -> "DynamicTiledProgram":
        """Add an input memory reference."""
        self.inputs.append(name)
        return self
    
    def add_output(self, name: str) -> "DynamicTiledProgram":
        """Add an output memory reference."""
        self.outputs.append(name)
        return self
    
    def add_tile(self, name: str) -> "DynamicTiledProgram":
        """Add a tile with default shape."""
        self.tiles.append(name)
        return self
    
    def add_tile_custom(self, name: str, rows: int, cols: int) -> "DynamicTiledProgram":
        """Add a tile with custom shape."""
        self.extra_tiles.append((name, rows, cols))
        return self
    
    def build(self, compute_fn) -> "PTOProgram":
        """
        Build the program with dynamic tiling.
        
        Args:
            compute_fn: A function that takes PTOFunctionBuilder and adds tile operations.
                       The function should use "tile_idx" as the offset variable.
        
        Returns:
            PTOProgram
        """
        builder = PTOFunctionBuilder(self.name)
        
        # Declare tiles
        for tile_name in self.tiles:
            builder.tile(tile_name, self.rows, self.cols, self.dtype)
        
        for tile_name, rows, cols in self.extra_tiles:
            builder.tile(tile_name, rows, cols, self.dtype)
        
        # Declare memory references
        for input_name in self.inputs:
            builder.memref(input_name, MemorySpace.GM, self.dtype)
        
        for output_name in self.outputs:
            builder.memref(output_name, MemorySpace.GM, self.dtype)
        
        # Declare scalar variables for control flow
        builder.scalar("num_full_tiles", ElementType.I32)
        builder.scalar("tail_elements", ElementType.I32)
        builder.scalar("has_tail", ElementType.U1)
        builder.scalar("zero", ElementType.I32)
        builder.scalar("tile_size", ElementType.I32)
        
        # Initialize constants
        builder.scalar_li("tile_size", self.tile_elements)
        builder.scalar_li("zero", 0)
        
        # Main loop: process full tiles
        builder.for_loop("tile_idx", 0, "num_full_tiles", 1)
        builder = compute_fn(builder)
        builder.end_for()
        
        # Handle tail
        builder.scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        builder.if_then("has_tail")
        # Use num_full_tiles as offset for tail
        builder = compute_fn(builder, tail=True)
        builder.endif()
        
        return builder.build()


# =============================================================================
# Convenience Functions for Common Patterns
# =============================================================================

def build_unary_op(name: str, op_method: str, dtype: ElementType = ElementType.F32,
                   target_isa: str = "arm64"):
    """
    Build a dynamic-tiled unary operation program.
    
    Args:
        name: Program name
        op_method: PTOFunctionBuilder method name (e.g., "relu", "exp", "neg")
        dtype: Element data type
        target_isa: Target ISA
    
    Returns:
        PTOProgram
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    builder = (PTOFunctionBuilder(name)
        .tile("x", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0))
    
    # Apply the operation
    builder = getattr(builder, op_method)("result", "x")
    
    builder = (builder
            .store("result", "output", "tile_idx", 0)
        .end_for()
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0))
    
    builder = getattr(builder, op_method)("result", "x")
    
    builder = (builder
            .store("result", "output", "num_full_tiles", 0)
        .endif())
    
    return builder.build()


def build_binary_op(name: str, op_method: str, dtype: ElementType = ElementType.F32,
                    target_isa: str = "arm64"):
    """
    Build a dynamic-tiled binary operation program.
    
    Args:
        name: Program name
        op_method: PTOFunctionBuilder method name (e.g., "add", "sub", "mul", "div")
        dtype: Element data type
        target_isa: Target ISA
    
    Returns:
        PTOProgram
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    builder = (PTOFunctionBuilder(name)
        .tile("x", rows, cols, dtype)
        .tile("y", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_y", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input_x", "tile_idx", 0)
            .load("y", "input_y", "tile_idx", 0))
    
    # Apply the operation
    builder = getattr(builder, op_method)("result", "x", "y")
    
    builder = (builder
            .store("result", "output", "tile_idx", 0)
        .end_for()
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input_x", "num_full_tiles", 0)
            .load("y", "input_y", "num_full_tiles", 0))
    
    builder = getattr(builder, op_method)("result", "x", "y")
    
    builder = (builder
            .store("result", "output", "num_full_tiles", 0)
        .endif())
    
    return builder.build()


def build_scalar_op(name: str, op_method: str, scalar_value: float,
                    dtype: ElementType = ElementType.F32, target_isa: str = "arm64"):
    """
    Build a dynamic-tiled tile-scalar operation program.
    
    Args:
        name: Program name
        op_method: PTOFunctionBuilder method name (e.g., "adds", "muls", "divs")
        scalar_value: The scalar value for the operation
        dtype: Element data type
        target_isa: Target ISA
    
    Returns:
        PTOProgram
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    builder = (PTOFunctionBuilder(name)
        .tile("x", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0))
    
    # Apply the operation
    builder = getattr(builder, op_method)("result", "x", scalar_value)
    
    builder = (builder
            .store("result", "output", "tile_idx", 0)
        .end_for()
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0))
    
    builder = getattr(builder, op_method)("result", "x", scalar_value)
    
    builder = (builder
            .store("result", "output", "num_full_tiles", 0)
        .endif())
    
    return builder.build()


# =============================================================================
# Tile Shape Summary Printer
# =============================================================================

def print_tile_shapes():
    """Print tile shape summary for all ISAs and common data types."""
    print("=" * 70)
    print("Tile Shape Summary for Different ISAs and Data Types")
    print("=" * 70)
    
    for isa in ["arm64", "cuda", "ascend_a2a3", "ascend_a5"]:
        print(f"\n{isa.upper()}:")
        for dtype in [ElementType.F32, ElementType.F16, ElementType.F64]:
            info = get_tile_info(dtype, isa)
            print(f"  {dtype.value}: {info['rows']}x{info['cols']} = "
                  f"{info['elements']} elements ({info['bytes']} bytes)")


if __name__ == "__main__":
    print_tile_shapes()
