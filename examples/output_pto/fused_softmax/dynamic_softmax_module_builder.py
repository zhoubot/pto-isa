"""
Auto-generated Python code from PTO Assembly
Module: dynamic_softmax_module
Entry: dynamic_softmax

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


def create_rowmax(module=None):
    """
    Create the rowmax function.
    Type: InCore
    """
    return (PTOFunctionBuilder("rowmax", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 8, 8, ElementType.F32)
        .tile("result", 8, 1, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .rowmax("result", "x")
        .store("result", "output", 0, 0)
        .build())

def create_rowexpandsub(module=None):
    """
    Create the rowexpandsub function.
    Type: InCore
    """
    return (PTOFunctionBuilder("rowexpandsub", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_x", MemorySpace.GM, ElementType.F32)
        .memref("input_row", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 8, 8, ElementType.F32)
        .tile("row_vals", 8, 1, ElementType.F32)
        .tile("result", 8, 8, ElementType.F32)
        
        # Instructions
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandsub("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())

def create_elem_exp(module=None):
    """
    Create the elem_exp function.
    Type: InCore
    """
    return (PTOFunctionBuilder("elem_exp", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 8, 8, ElementType.F32)
        .tile("result", 8, 8, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .exp("result", "x")
        .store("result", "output", 0, 0)
        .build())

def create_rowsum(module=None):
    """
    Create the rowsum function.
    Type: InCore
    """
    return (PTOFunctionBuilder("rowsum", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 8, 8, ElementType.F32)
        .tile("result", 8, 1, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .rowsum("result", "x")
        .store("result", "output", 0, 0)
        .build())

def create_rowexpanddiv(module=None):
    """
    Create the rowexpanddiv function.
    Type: InCore
    """
    return (PTOFunctionBuilder("rowexpanddiv", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_x", MemorySpace.GM, ElementType.F32)
        .memref("input_row", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 8, 8, ElementType.F32)
        .tile("row_vals", 8, 1, ElementType.F32)
        .tile("result", 8, 8, ElementType.F32)
        
        # Instructions
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpanddiv("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())

def create_dynamic_softmax(module=None):
    """
    Create the dynamic_softmax function.
    Type: Orchestration
    """
    return (PTOFunctionBuilder("dynamic_softmax", module=module)
        .not_in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        .memref("temp_rowmax", MemorySpace.GM, ElementType.F32)
        .memref("temp_shifted", MemorySpace.GM, ElementType.F32)
        .memref("temp_exp", MemorySpace.GM, ElementType.F32)
        .memref("temp_rowsum", MemorySpace.GM, ElementType.F32)
        
        # Scalar declarations
        .scalar("total_rows", ElementType.I32)
        .scalar("tile_rows", ElementType.I32)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_rows", ElementType.I32)
        .scalar("has_tail", ElementType.I32)
        .scalar("zero", ElementType.I32)
        
        # Instructions
        .scalar_li("tile_rows", 8)
        .scalar_li("zero", 0)
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
        .call("rowmax", {"input": "input", "output": "temp_rowmax"})
        .call("rowexpandsub", {"input_x": "input", "input_row": "temp_rowmax", "output": "temp_shifted"})
        .call("elem_exp", {"input": "temp_shifted", "output": "temp_exp"})
        .call("rowsum", {"input": "temp_exp", "output": "temp_rowsum"})
        .call("rowexpanddiv", {"input_x": "temp_exp", "input_row": "temp_rowsum", "output": "output"})
        .end_for()
        .if_then("has_tail")
        .call("rowmax", {"input": "input", "output": "temp_rowmax"})
        .call("rowexpandsub", {"input_x": "input", "input_row": "temp_rowmax", "output": "temp_shifted"})
        .call("elem_exp", {"input": "temp_shifted", "output": "temp_exp"})
        .call("rowsum", {"input": "temp_exp", "output": "temp_rowsum"})
        .call("rowexpanddiv", {"input_x": "temp_exp", "input_row": "temp_rowsum", "output": "output"})
        .endif()
        .build())


def create_dynamic_softmax_module_module():
    """Create the dynamic_softmax_module module."""
    module = PTOModule("dynamic_softmax_module")

    # Add InCore functions
    module.add_function(create_rowmax(module))
    module.add_function(create_rowexpandsub(module))
    module.add_function(create_elem_exp(module))
    module.add_function(create_rowsum(module))
    module.add_function(create_rowexpanddiv(module))

    # Add Orchestration functions
    module.add_function(create_dynamic_softmax(module))

    module.set_entry("dynamic_softmax")

    return module


def main():
    """Create and compile the dynamic_softmax_module module."""
    module = create_dynamic_softmax_module_module()

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
