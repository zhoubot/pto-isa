"""
PTO ISA Example: sinh() using Taylor Expansion (Python Version)

This file demonstrates the PTO Python DSL for computing sinh(x) on tiles.

Taylor expansion for sinh(x):
    sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + ...
            = x + x³/6 + x⁵/120 + x⁷/5040 + ...

Algorithm:
    result = x
    term = x
    x_squared = x * x
    for n = 1 to N:
        term = term * x_squared / ((2n)(2n+1))
        result = result + term
"""

import os
import sys

# Add parent directory for imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_compile import PTOFunctionBuilder, PTOCompiler
from pto_isa_definition import ElementType, MemorySpace


# =============================================================================
# Build sinh() PTO Program using Python DSL
# =============================================================================

def build_sinh_program():
    """
    Build the sinh() computation using PTO Python DSL.
    
    sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + ...
    
    Uses 7 terms of Taylor expansion for good accuracy.
    """
    program = (PTOFunctionBuilder("sinh_taylor")
        # ====================================================================
        # Tile Declarations
        # ====================================================================
        .tile("x", 8, 8, ElementType.F32)           # Input tile
        .tile("x_squared", 8, 8, ElementType.F32)   # x²
        .tile("term", 8, 8, ElementType.F32)        # Current Taylor term
        .tile("result", 8, 8, ElementType.F32)      # Accumulated result
        
        # Memory references
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # ====================================================================
        # Load input
        # ====================================================================
        .load("x", "input", 0, 0)
        
        # ====================================================================
        # Initialize
        # ====================================================================
        # result = x (first term of Taylor series)
        .muls("result", "x", 1.0)          # result = x * 1.0 = x
        
        # x_squared = x * x
        .mul("x_squared", "x", "x")        # x_squared = x²
        
        # term = x (starting term)
        .muls("term", "x", 1.0)            # term = x
        
        # ====================================================================
        # Taylor Expansion Terms
        # ====================================================================
        
        # Term 2: x³/3! = x³/6
        .mul("term", "term", "x_squared")  # term = x * x² = x³
        .divs("term", "term", 6.0)         # term = x³/6
        .add("result", "result", "term")   # result = x + x³/6
        
        # Term 3: x⁵/5! = x⁵/120
        # term = (x³/6) * x² / 20 = x⁵/120
        .mul("term", "term", "x_squared")  # term = x⁵/6
        .divs("term", "term", 20.0)        # term = x⁵/120
        .add("result", "result", "term")   # result += x⁵/120
        
        # Term 4: x⁷/7! = x⁷/5040
        # term = (x⁵/120) * x² / 42 = x⁷/5040
        .mul("term", "term", "x_squared")  # term = x⁷/120
        .divs("term", "term", 42.0)        # term = x⁷/5040
        .add("result", "result", "term")   # result += x⁷/5040
        
        # Term 5: x⁹/9! = x⁹/362880
        # term = (x⁷/5040) * x² / 72 = x⁹/362880
        .mul("term", "term", "x_squared")
        .divs("term", "term", 72.0)
        .add("result", "result", "term")
        
        # Term 6: x¹¹/11!
        # term = term * x² / 110
        .mul("term", "term", "x_squared")
        .divs("term", "term", 110.0)
        .add("result", "result", "term")
        
        # Term 7: x¹³/13!
        # term = term * x² / 156
        .mul("term", "term", "x_squared")
        .divs("term", "term", 156.0)
        .add("result", "result", "term")
        
        # ====================================================================
        # Store result
        # ====================================================================
        .store("result", "output", 0, 0)
        
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
    print("PTO ISA sinh() - Multi-Backend Code Generation")
    print("=" * 70)
    
    # Base output directory  
    OUTPUT_PREFIX = "sinh_taylor"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Build the program
    program = build_sinh_program()
    
    # =========================================================================
    # Generate Code for All Backends
    # =========================================================================
    print(f"\nGenerating sinh() for {len(BACKENDS)} backends...")
    print(f"Backends: {', '.join(BACKENDS.keys())}")
    print()
    
    print("[sinh_taylor]")
    results = generate_all_backends(
        program, 
        OUTPUT_PREFIX,
        output_base_dir=SCRIPT_DIR,
        enable_fusion=True
    )
    
    # =========================================================================
    # Also show comparison with/without fusion
    # =========================================================================
    print("\n" + "=" * 70)
    print("Loop Fusion Comparison (ARM64)")
    print("=" * 70)
    
    arm64_code_fused = generate_arm64_code(program, enable_fusion=True)
    arm64_code_nofusion = generate_arm64_code(program, enable_fusion=False)
    
    fused_lines = len(arm64_code_fused.split('\n'))
    nofused_lines = len(arm64_code_nofusion.split('\n'))
    
    print(f"Lines with fusion:    {fused_lines}")
    print(f"Lines without fusion: {nofused_lines}")
    if nofused_lines > 0:
        print(f"Reduction:            {nofused_lines - fused_lines} lines ({100*(nofused_lines-fused_lines)//nofused_lines}%)")
    
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print(f"Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{OUTPUT_PREFIX}/")
    print(f"  - output_pto/{OUTPUT_PREFIX}/")
    print("=" * 70)
