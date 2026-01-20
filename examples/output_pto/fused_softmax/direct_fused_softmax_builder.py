"""
Auto-generated Python code from PTO Assembly
Module: parsed_module
Entry: None

This code uses PTOFunctionBuilder to construct the same program
as the original .pto assembly file.
"""

import sys
import os

# Add project root to path for imports
# This handles nested directory structures like output_pto/llama7b/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = _script_dir
while _project_root and not os.path.exists(os.path.join(_project_root, "pto_compile.py")):
    _project_root = os.path.dirname(_project_root)
if _project_root:
    sys.path.insert(0, _project_root)

from pto_compile import (
    PTOFunctionBuilder, PTOModule, PTOModuleCompiler,
    MultiBackendCodeGenerator, ElementType, MemorySpace
)


def create_direct_fused_softmax(module=None):
    """
    Create the direct_fused_softmax function.
    Type: InCore
    """
    return (PTOFunctionBuilder("direct_fused_softmax", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 8, 8, ElementType.F32)
        .tile("row_max", 8, 1, ElementType.F32)
        .tile("x_shifted", 8, 8, ElementType.F32)
        .tile("exp_x", 8, 8, ElementType.F32)
        .tile("row_sum", 8, 1, ElementType.F32)
        .tile("result", 8, 8, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .rowmax("row_max", "x")
        .rowexpandsub("x_shifted", "x", "row_max")
        .exp("exp_x", "x_shifted")
        .rowsum("row_sum", "exp_x")
        .rowexpanddiv("result", "exp_x", "row_sum")
        .store("result", "output", 0, 0)
        .build())


def create_parsed_module_module():
    """Create the parsed_module module."""
    module = PTOModule("parsed_module")

    # Add InCore functions
    module.add_function(create_direct_fused_softmax(module))

    return module


def main():
    """Create and compile the parsed_module module."""
    module = create_parsed_module_module()

    print(f"Module: {module.name}")
    print(f"Functions: {len(module.functions)}")
    for name, func in module.functions.items():
        func_type = "InCore" if func.is_in_core else "Orchestration"
        print(f"  - {name}: {func_type}")

    # Compile to PTO assembly
    compiler = PTOModuleCompiler()
    pto_code = compiler.compile(module)
    print("\n--- PTO Assembly ---")
    print(pto_code[:2000] + "..." if len(pto_code) > 2000 else pto_code)


if __name__ == "__main__":
    main()
