#!/usr/bin/env python3
"""
PTO Assembly Parser and Python Code Generator

This module provides:
1. PTOParser: Parses .pto assembly files into internal data structures
2. PythonCodeGenerator: Generates equivalent PTOFunctionBuilder Python code

Usage:
    python pto_parser.py <input_dir> [--output-dir <output_dir>]
    
Example:
    python pto_parser.py examples/output_pto/llama7b/
    
This will:
1. Read all .pto files in the input directory
2. Parse them into PTOProgram/PTOModule structures
3. Generate equivalent Python code using PTOFunctionBuilder API
4. Save the generated Python code to the same directory (or output_dir if specified)
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum


# =============================================================================
# Data Structures for Parsed PTO
# =============================================================================

@dataclass
class ParsedTile:
    """Parsed tile declaration."""
    name: str
    rows: int
    cols: int
    dtype: str  # e.g., "f32", "f16"


@dataclass
class ParsedMemref:
    """Parsed memref declaration."""
    name: str
    memory_space: str  # "gm" for global memory
    dtype: str


@dataclass
class ParsedScalar:
    """Parsed scalar declaration."""
    name: str
    dtype: str  # "i32", "f32", etc.


@dataclass
class ParsedInstruction:
    """Parsed instruction."""
    opcode: str
    dst: Optional[str] = None
    operands: List[Any] = field(default_factory=list)
    type_info: Optional[str] = None
    # For CALL instructions
    callee: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    # For control flow
    loop_var: Optional[str] = None
    lb: Any = None
    ub: Any = None
    step: Any = None
    max_range: Optional[int] = None  # For binary-expanded loops
    min_range: Optional[int] = None  # Minimum power-of-2 block for binary expansion


@dataclass
class ParsedFunction:
    """Parsed function."""
    name: str
    is_in_core: bool
    memrefs: List[ParsedMemref] = field(default_factory=list)
    tiles: List[ParsedTile] = field(default_factory=list)
    scalars: List[ParsedScalar] = field(default_factory=list)
    instructions: List[ParsedInstruction] = field(default_factory=list)


@dataclass
class ParsedModule:
    """Parsed module."""
    name: str
    functions: Dict[str, ParsedFunction] = field(default_factory=dict)
    entry_function: Optional[str] = None


# =============================================================================
# PTO Parser
# =============================================================================

class PTOParser:
    """
    Parser for .pto assembly files.
    
    Parses the text representation of PTO programs into structured data.
    """
    
    def __init__(self):
        self.module: Optional[ParsedModule] = None
        self.current_function: Optional[ParsedFunction] = None
        self.lines: List[str] = []
        self.pos: int = 0
    
    def parse_file(self, filepath: str) -> ParsedModule:
        """Parse a .pto file and return a ParsedModule."""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse(content)
    
    def parse(self, content: str) -> ParsedModule:
        """Parse PTO assembly content."""
        self.lines = content.split('\n')
        self.pos = 0
        self.module = ParsedModule(name="parsed_module")
        
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//'):
                # Check for module metadata in comments
                if line.startswith('// PTO Module:'):
                    self.module.name = line.split(':')[1].strip()
                elif line.startswith('// Entry:'):
                    entry = line.split(':')[1].strip()
                    if entry.startswith('@'):
                        entry = entry[1:]
                    self.module.entry_function = entry
                self.pos += 1
                continue
            
            # Parse function definition
            if line.startswith('func @'):
                self._parse_function()
            else:
                self.pos += 1
        
        return self.module
    
    def _parse_function(self):
        """Parse a function definition."""
        line = self.lines[self.pos].strip()
        
        # Check if InCore or Orchestration (from previous comment line)
        is_in_core = True
        if self.pos > 0:
            prev_line = self.lines[self.pos - 1].strip()
            if 'Orchestration' in prev_line:
                is_in_core = False
        
        # Parse function signature: func @name(%param1: type, %param2: type, ...) {
        match = re.match(r'func @(\w+)\((.*?)\)\s*\{', line)
        if not match:
            self.pos += 1
            return
        
        func_name = match.group(1)
        params_str = match.group(2)
        
        self.current_function = ParsedFunction(
            name=func_name,
            is_in_core=is_in_core
        )
        
        # Parse parameters (memrefs)
        if params_str.strip():
            params = self._split_params(params_str)
            for param in params:
                memref = self._parse_param(param)
                if memref:
                    self.current_function.memrefs.append(memref)
        
        self.pos += 1
        
        # Parse function body
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            
            if line == '}':
                self.pos += 1
                break
            
            # Skip comments and empty lines
            if not line or line.startswith('//'):
                self.pos += 1
                continue
            
            # Parse declarations and instructions
            if line.startswith('%') and '= alloc_tile' in line:
                self._parse_tile_decl(line)
            elif line.startswith('%') and '= alloc_scalar' in line:
                self._parse_scalar_decl(line)
            elif line.startswith('return'):
                self.current_function.instructions.append(
                    ParsedInstruction(opcode='RETURN')
                )
            elif line.startswith('FOR '):
                self._parse_for_loop(line)
            elif line.startswith('ENDFOR'):
                self.current_function.instructions.append(
                    ParsedInstruction(opcode='ENDFOR')
                )
            elif line.startswith('IF '):
                self._parse_if(line)
            elif line.startswith('ENDIF'):
                self.current_function.instructions.append(
                    ParsedInstruction(opcode='ENDIF')
                )
            elif line.startswith('CALL '):
                self._parse_call(line)
            elif line.startswith('LI '):
                self._parse_load_immediate(line)
            elif line.startswith('%') and '=' in line:
                self._parse_instruction(line)
            elif line.startswith('tstore'):
                self._parse_store(line)
            
            self.pos += 1
        
        self.module.functions[func_name] = self.current_function
    
    def _split_params(self, params_str: str) -> List[str]:
        """Split parameter string, handling nested brackets."""
        params = []
        depth = 0
        current = ""
        for c in params_str:
            if c in '(<':
                depth += 1
                current += c
            elif c in ')>':
                depth -= 1
                current += c
            elif c == ',' and depth == 0:
                if current.strip():
                    params.append(current.strip())
                current = ""
            else:
                current += c
        if current.strip():
            params.append(current.strip())
        return params
    
    def _parse_param(self, param: str) -> Optional[ParsedMemref]:
        """Parse a function parameter."""
        # Format: %name: !pto.memref<gm,...,f32>
        match = re.match(r'%(\w+):\s*!pto\.memref<(\w+),.*?,(\w+)>', param)
        if match:
            return ParsedMemref(
                name=match.group(1),
                memory_space=match.group(2),
                dtype=match.group(3)
            )
        return None
    
    def _parse_tile_decl(self, line: str):
        """Parse tile declaration."""
        # Format: %name = alloc_tile : !pto.tile<ROWSxCOLSxDTYPE>
        match = re.match(r'%(\w+)\s*=\s*alloc_tile\s*:\s*!pto\.tile<(\d+)x(\d+)x(\w+)>', line)
        if match:
            self.current_function.tiles.append(ParsedTile(
                name=match.group(1),
                rows=int(match.group(2)),
                cols=int(match.group(3)),
                dtype=match.group(4)
            ))
    
    def _parse_scalar_decl(self, line: str):
        """Parse scalar declaration."""
        # Format: %name = alloc_scalar : dtype
        match = re.match(r'%(\w+)\s*=\s*alloc_scalar\s*:\s*(\w+)', line)
        if match:
            self.current_function.scalars.append(ParsedScalar(
                name=match.group(1),
                dtype=match.group(2)
            ))
    
    def _parse_for_loop(self, line: str):
        """Parse FOR loop."""
        # Format: FOR %iv:idx, lb:idx, ub:idx, step:idx [max_range=N] [min_range=M]
        match = re.match(r'FOR\s+%(\w+):idx,\s*(\S+):idx,\s*%?(\S+):idx,\s*(\S+):idx(?:\s+max_range=(\d+))?(?:\s+min_range=(\d+))?', line)
        if match:
            iv = match.group(1)
            lb = match.group(2)
            ub = match.group(3)
            step = match.group(4)
            max_range_str = match.group(5)
            min_range_str = match.group(6)
            
            # Convert to int if possible
            try:
                lb = int(lb)
            except ValueError:
                pass
            try:
                step = int(step)
            except ValueError:
                pass
            
            # Parse max_range and min_range
            max_range = int(max_range_str) if max_range_str else None
            min_range = int(min_range_str) if min_range_str else None
            
            self.current_function.instructions.append(ParsedInstruction(
                opcode='FOR',
                loop_var=iv,
                lb=lb,
                ub=ub,
                step=step,
                max_range=max_range,
                min_range=min_range
            ))
    
    def _parse_if(self, line: str):
        """Parse IF statement."""
        # Format: IF %condition
        match = re.match(r'IF\s+%(\w+)', line)
        if match:
            self.current_function.instructions.append(ParsedInstruction(
                opcode='IF',
                operands=[match.group(1)]
            ))
    
    def _parse_call(self, line: str):
        """Parse CALL instruction."""
        # Format: CALL @func(%param -> %arg, %param -> (%arg, offset, 0), ...)
        match = re.match(r'CALL\s+@(\w+)\((.*)\)$', line)
        if match:
            callee = match.group(1)
            args_str = match.group(2)
            
            args = {}
            if args_str.strip():
                # Parse each argument - need to handle nested parens
                arg_parts = self._split_call_args(args_str)
                for part in arg_parts:
                    # Format: %param -> %arg or %param -> (%arg, offset, 0)
                    if ' -> ' in part:
                        arrow_idx = part.index(' -> ')
                        param = part[:arrow_idx].strip().lstrip('%')
                        value = part[arrow_idx + 4:].strip()
                        
                        # Check if tuple format
                        if value.startswith('(') and value.endswith(')'):
                            # Parse tuple: (%tensor, offset, col)
                            inner = value[1:-1]
                            tuple_parts = [p.strip().lstrip('%') for p in inner.split(',')]
                            if len(tuple_parts) >= 2:
                                tensor = tuple_parts[0]
                                row_off = tuple_parts[1]
                                col_off = tuple_parts[2] if len(tuple_parts) > 2 else "0"
                                try:
                                    col_off = int(col_off)
                                except ValueError:
                                    pass
                                args[param] = (tensor, row_off, col_off)
                            else:
                                args[param] = value.lstrip('%')
                        else:
                            args[param] = value.lstrip('%')
            
            self.current_function.instructions.append(ParsedInstruction(
                opcode='CALL',
                callee=callee,
                args=args
            ))
    
    def _split_call_args(self, args_str: str) -> List[str]:
        """Split CALL arguments, handling nested parentheses."""
        args = []
        depth = 0
        current = ""
        for c in args_str:
            if c == '(':
                depth += 1
                current += c
            elif c == ')':
                depth -= 1
                current += c
            elif c == ',' and depth == 0:
                if current.strip():
                    args.append(current.strip())
                current = ""
            else:
                current += c
        if current.strip():
            args.append(current.strip())
        return args
    
    def _parse_load_immediate(self, line: str):
        """Parse LI (load immediate) instruction."""
        # Format: LI %var:type, value
        match = re.match(r'LI\s+%(\w+):(\w+),\s*(\S+)', line)
        if match:
            self.current_function.instructions.append(ParsedInstruction(
                opcode='LI',
                dst=match.group(1),
                operands=[match.group(3)],
                type_info=match.group(2)
            ))
    
    def _parse_instruction(self, line: str):
        """Parse a general instruction."""
        # Format: %dst = opcode %src1, %src2 : type_info
        # or: %dst = opcode %src : type_info
        # or: %dst = tload %memref[row, col] : type_info
        
        # Remove type annotations for simpler parsing
        if ' : ' in line:
            line_parts = line.split(' : ')
            line = line_parts[0]
            type_info = line_parts[1]
        else:
            type_info = None
        
        # Parse destination
        if '=' not in line:
            return
        
        dst_part, rest = line.split('=', 1)
        dst = dst_part.strip().lstrip('%')
        rest = rest.strip()
        
        # Handle TLOAD specially: %dst = tload %src[row, col]
        # Row and col can be numbers or variable names (e.g., %tile_idx or tile_idx)
        tload_match = re.match(r'tload\s+%(\w+)\[%?(\w+),\s*%?(\w+)\]', rest)
        if tload_match:
            src = tload_match.group(1)
            row = tload_match.group(2)
            col = tload_match.group(3)
            self.current_function.instructions.append(ParsedInstruction(
                opcode='TLOAD',
                dst=dst,
                operands=[src, row, col],
                type_info=type_info
            ))
            return
        
        # Parse opcode and operands
        parts = rest.split()
        if not parts:
            return
        
        opcode = parts[0].upper()
        
        # Parse operands
        operands_str = ' '.join(parts[1:])
        operands = []
        
        # Handle comma-separated operands
        if operands_str:
            op_parts = operands_str.split(',')
            for op in op_parts:
                op = op.strip().lstrip('%')
                operands.append(op)
        
        self.current_function.instructions.append(ParsedInstruction(
            opcode=opcode,
            dst=dst,
            operands=operands,
            type_info=type_info
        ))
    
    def _parse_store(self, line: str):
        """Parse TSTORE instruction."""
        # Format: tstore %src, %dst[row, col] - row/col can be numbers or variables
        match = re.match(r'tstore\s+%(\w+),\s*%(\w+)\[%?(\w+),\s*%?(\w+)\]', line)
        if match:
            self.current_function.instructions.append(ParsedInstruction(
                opcode='TSTORE',
                operands=[match.group(1), match.group(2), match.group(3), match.group(4)]
            ))


# =============================================================================
# Python Code Generator
# =============================================================================

class PythonCodeGenerator:
    """
    Generates Python code that uses PTOFunctionBuilder to recreate the parsed program.
    """
    
    # Mapping from PTO opcodes to PTOFunctionBuilder methods
    OPCODE_TO_METHOD = {
        'TLOAD': 'load',
        'TSTORE': 'store',
        'TADD': 'add',
        'TSUB': 'sub',
        'TMUL': 'mul',
        'TDIV': 'div',
        'TNEG': 'neg',
        'TEXP': 'exp',
        'TLOG': 'log',
        'TSQRT': 'sqrt',
        'TRSQRT': 'rsqrt',
        'TRECIP': 'recip',
        'TMATMUL': 'matmul',
        'TMATMUL_ACC': 'matmul_acc',
        'TROWSUM': 'rowsum',
        'TROWMAX': 'rowmax',
        'TCOLSUM': 'colsum',
        'TROWEXPANDSUB': 'rowexpandsub',
        'TROWEXPANDDIV': 'rowexpanddiv',
        'TROWEXPANDMUL': 'rowexpandmul',
        'TADDS': 'adds',
        'TMULS': 'muls',
        'TSILU': 'silu',
    }
    
    DTYPE_MAP = {
        'f32': 'ElementType.F32',
        'f16': 'ElementType.F16',
        'i32': 'ElementType.I32',
        'i64': 'ElementType.I64',
    }
    
    def __init__(self, module: ParsedModule):
        self.module = module
    
    def generate(self) -> str:
        """Generate Python code for the entire module."""
        lines = []
        
        # Header
        lines.append('"""')
        lines.append(f'Auto-generated Python code from PTO Assembly')
        lines.append(f'Module: {self.module.name}')
        lines.append(f'Entry: {self.module.entry_function}')
        lines.append('')
        lines.append('This code uses PTOFunctionBuilder to construct the same program')
        lines.append('as the original .pto assembly file.')
        lines.append('"""')
        lines.append('')
        
        # Imports
        lines.append('import sys')
        lines.append('import os')
        lines.append('')
        lines.append('# Add project root to path for imports')
        lines.append('# This handles nested directory structures like output_pto/llama7b/')
        lines.append('_script_dir = os.path.dirname(os.path.abspath(__file__))')
        lines.append('_project_root = _script_dir')
        lines.append('while _project_root and not os.path.exists(os.path.join(_project_root, "pto_compile.py")):')
        lines.append('    _project_root = os.path.dirname(_project_root)')
        lines.append('if _project_root:')
        lines.append('    sys.path.insert(0, _project_root)')
        lines.append('')
        lines.append('from pto_compile import (')
        lines.append('    PTOFunctionBuilder, PTOModule, PTOModuleCompiler,')
        lines.append('    MultiBackendCodeGenerator, ElementType, MemorySpace')
        lines.append(')')
        lines.append('')
        lines.append('')
        
        # Generate function creators
        for func_name, func in self.module.functions.items():
            lines.extend(self._generate_function_creator(func))
            lines.append('')
        
        # Generate module creation function
        lines.append('')
        lines.append(f'def create_{self.module.name}_module():')
        lines.append(f'    """Create the {self.module.name} module."""')
        lines.append(f'    module = PTOModule("{self.module.name}")')
        lines.append('')
        
        # Add functions in order (InCore first, then Orchestration)
        incore_funcs = [f for f in self.module.functions.values() if f.is_in_core]
        orch_funcs = [f for f in self.module.functions.values() if not f.is_in_core]
        
        lines.append('    # Add InCore functions')
        for func in incore_funcs:
            lines.append(f'    module.add_function(create_{func.name}(module))')
        
        if orch_funcs:
            lines.append('')
            lines.append('    # Add Orchestration functions')
            for func in orch_funcs:
                lines.append(f'    module.add_function(create_{func.name}(module))')
        
        if self.module.entry_function:
            lines.append('')
            lines.append(f'    module.set_entry("{self.module.entry_function}")')
        
        lines.append('')
        lines.append('    return module')
        lines.append('')
        
        # Main function
        lines.append('')
        lines.append('def main():')
        lines.append(f'    """Create and compile the {self.module.name} module."""')
        lines.append(f'    module = create_{self.module.name}_module()')
        lines.append('')
        lines.append('    print(f"Module: {module.name}")')
        lines.append('    print(f"Functions: {len(module.functions)}")')
        lines.append('    for name, func in module.functions.items():')
        lines.append('        func_type = "InCore" if func.is_in_core else "Orchestration"')
        lines.append('        print(f"  - {name}: {func_type}")')
        lines.append('')
        lines.append('    # Compile to PTO assembly')
        lines.append('    compiler = PTOModuleCompiler()')
        lines.append('    pto_code = compiler.compile(module)')
        lines.append('    print("\\n--- PTO Assembly ---")')
        lines.append('    print(pto_code[:2000] + "..." if len(pto_code) > 2000 else pto_code)')
        lines.append('')
        lines.append('')
        lines.append('if __name__ == "__main__":')
        lines.append('    main()')
        lines.append('')
        
        return '\n'.join(lines)
    
    def _generate_function_creator(self, func: ParsedFunction) -> List[str]:
        """Generate a function creator for a single function."""
        lines = []
        
        # Function signature
        lines.append(f'def create_{func.name}(module=None):')
        lines.append(f'    """')
        lines.append(f'    Create the {func.name} function.')
        lines.append(f'    Type: {"InCore" if func.is_in_core else "Orchestration"}')
        lines.append(f'    """')
        
        # Start builder
        lines.append(f'    return (PTOFunctionBuilder("{func.name}", module=module)')
        
        # Set InCore/Orchestration
        if func.is_in_core:
            lines.append('        .in_core()')
        else:
            lines.append('        .not_in_core()')
        
        # Add memrefs - they need to be declared before tiles
        if func.memrefs:
            lines.append('        ')
            lines.append('        # Memref declarations (function parameters)')
            for memref in func.memrefs:
                dtype = self.DTYPE_MAP.get(memref.dtype, 'ElementType.F32')
                lines.append(f'        .memref("{memref.name}", MemorySpace.GM, {dtype})')
        
        # Add tiles
        if func.tiles:
            lines.append('        ')
            lines.append('        # Tile declarations')
            for tile in func.tiles:
                dtype = self.DTYPE_MAP.get(tile.dtype, 'ElementType.F32')
                lines.append(f'        .tile("{tile.name}", {tile.rows}, {tile.cols}, {dtype})')
        
        # Add scalars
        if func.scalars:
            lines.append('        ')
            lines.append('        # Scalar declarations')
            for scalar in func.scalars:
                dtype = self.DTYPE_MAP.get(scalar.dtype, 'ElementType.I32')
                lines.append(f'        .scalar("{scalar.name}", {dtype})')
        
        # Add blank line before instructions
        lines.append('        ')
        lines.append('        # Instructions')
        
        # Generate instructions
        for instr in func.instructions:
            instr_code = self._generate_instruction(instr)
            if instr_code:
                lines.append(f'        {instr_code}')
        
        # End builder
        lines.append('        .build())')
        
        return lines
    
    def _generate_instruction(self, instr: ParsedInstruction) -> Optional[str]:
        """Generate code for a single instruction."""
        opcode = instr.opcode
        
        # Control flow
        if opcode == 'FOR':
            lb = instr.lb if isinstance(instr.lb, int) else f'"{instr.lb}"'
            ub = f'"{instr.ub}"' if isinstance(instr.ub, str) else instr.ub
            step = instr.step if isinstance(instr.step, int) else f'"{instr.step}"'
            # Build optional keyword arguments
            kwargs = []
            if instr.max_range is not None:
                kwargs.append(f'max_range={instr.max_range}')
            if instr.min_range is not None:
                kwargs.append(f'min_range={instr.min_range}')
            if kwargs:
                return f'.for_loop("{instr.loop_var}", {lb}, {ub}, {step}, {", ".join(kwargs)})'
            return f'.for_loop("{instr.loop_var}", {lb}, {ub}, {step})'
        
        elif opcode == 'ENDFOR':
            return '.end_for()'
        
        elif opcode == 'IF':
            return f'.if_then("{instr.operands[0]}")'
        
        elif opcode == 'ENDIF':
            return '.endif()'
        
        elif opcode == 'RETURN':
            return None  # build() handles return
        
        # Function call
        elif opcode == 'CALL':
            args_str = self._format_call_args(instr.args)
            return f'.call("{instr.callee}", {{{args_str}}})'
        
        # Load immediate
        elif opcode == 'LI':
            value = instr.operands[0] if instr.operands else '0'
            return f'.scalar_li("{instr.dst}", {value})'
        
        # Memory operations
        elif opcode == 'TLOAD':
            # Format: dst = tload src[row, col] -> operands: [src, row, col]
            src = instr.operands[0] if instr.operands else 'input'
            row_val = instr.operands[1] if len(instr.operands) > 1 else '0'
            col_val = instr.operands[2] if len(instr.operands) > 2 else '0'
            # Check if row/col are numeric or variable
            try:
                row = int(row_val)
            except ValueError:
                row = f'"{row_val}"'
            try:
                col = int(col_val)
            except ValueError:
                col = f'"{col_val}"'
            return f'.load("{instr.dst}", "{src}", {row}, {col})'
        
        elif opcode == 'TSTORE':
            # Format: tstore src, dst[row, col] -> operands: [src, dst, row, col]
            # PTOFunctionBuilder.store(src_tile, dst_memref, row, col)
            src = instr.operands[0] if instr.operands else 'result'
            dst = instr.operands[1] if len(instr.operands) > 1 else 'output'
            row_val = instr.operands[2] if len(instr.operands) > 2 else '0'
            col_val = instr.operands[3] if len(instr.operands) > 3 else '0'
            # Check if row/col are numeric or variable
            try:
                row = int(row_val)
            except ValueError:
                row = f'"{row_val}"'
            try:
                col = int(col_val)
            except ValueError:
                col = f'"{col_val}"'
            return f'.store("{src}", "{dst}", {row}, {col})'
        
        # Arithmetic operations
        elif opcode in self.OPCODE_TO_METHOD:
            method = self.OPCODE_TO_METHOD[opcode]
            
            if opcode in ['TADDS', 'TMULS']:
                # Scalar operations: dst = op src, scalar
                src = instr.operands[0] if instr.operands else 'x'
                scalar = instr.operands[1] if len(instr.operands) > 1 else '1.0'
                # Check if scalar is a literal or variable
                if scalar.startswith('%') or (not scalar.replace('.', '').replace('-', '').isdigit()):
                    scalar = f'"{scalar}"'
                return f'.{method}("{instr.dst}", "{src}", {scalar})'
            
            elif opcode in ['TNEG', 'TEXP', 'TLOG', 'TSQRT', 'TRSQRT', 'TRECIP', 'TROWSUM', 'TROWMAX', 'TCOLSUM', 'TSILU']:
                # Unary operations
                src = instr.operands[0] if instr.operands else 'x'
                return f'.{method}("{instr.dst}", "{src}")'
            
            elif opcode in ['TROWEXPANDSUB', 'TROWEXPANDDIV', 'TROWEXPANDMUL']:
                # Row expand operations: dst = op tile, row_vec
                tile = instr.operands[0] if instr.operands else 'x'
                row_vec = instr.operands[1] if len(instr.operands) > 1 else 'row'
                return f'.{method}("{instr.dst}", "{tile}", "{row_vec}")'
            
            elif opcode == 'TMATMUL':
                # Matrix multiply
                a = instr.operands[0] if instr.operands else 'a'
                b = instr.operands[1] if len(instr.operands) > 1 else 'b'
                return f'.{method}("{instr.dst}", "{a}", "{b}")'
            
            elif opcode == 'TMATMUL_ACC':
                # Matrix multiply accumulate
                a = instr.operands[0] if instr.operands else 'a'
                b = instr.operands[1] if len(instr.operands) > 1 else 'b'
                c = instr.operands[2] if len(instr.operands) > 2 else 'c'
                return f'.{method}("{instr.dst}", "{a}", "{b}", "{c}")'
            
            else:
                # Binary operations: dst = op src1, src2
                src1 = instr.operands[0] if instr.operands else 'a'
                src2 = instr.operands[1] if len(instr.operands) > 1 else 'b'
                return f'.{method}("{instr.dst}", "{src1}", "{src2}")'
        
        # Unknown instruction - add as comment
        return f'# Unknown: {opcode} {instr.dst} {instr.operands}'
    
    def _format_call_args(self, args: Dict[str, Any]) -> str:
        """Format CALL arguments as Python dict literal."""
        parts = []
        for param, value in args.items():
            if isinstance(value, tuple):
                # Tuple format: (tensor, offset, col)
                tensor, row_off, col_off = value
                parts.append(f'"{param}": ("{tensor}", "{row_off}", {col_off})')
            else:
                parts.append(f'"{param}": "{value}"')
        return ', '.join(parts)


# =============================================================================
# Main Entry Point
# =============================================================================

def process_pto_directory(input_dir: str, output_dir: Optional[str] = None):
    """
    Process all .pto files in a directory.
    
    Args:
        input_dir: Directory containing .pto files
        output_dir: Output directory for generated Python files (default: same as input)
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .pto files
    pto_files = [f for f in os.listdir(input_dir) if f.endswith('.pto')]
    
    if not pto_files:
        print(f"No .pto files found in {input_dir}")
        return
    
    print(f"Found {len(pto_files)} .pto file(s) in {input_dir}")
    
    for pto_file in pto_files:
        input_path = os.path.join(input_dir, pto_file)
        output_name = pto_file.replace('.pto', '_builder.py')
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\nProcessing: {pto_file}")
        
        try:
            # Parse
            parser = PTOParser()
            module = parser.parse_file(input_path)
            
            print(f"  Module: {module.name}")
            print(f"  Functions: {len(module.functions)}")
            print(f"    InCore: {sum(1 for f in module.functions.values() if f.is_in_core)}")
            print(f"    Orchestration: {sum(1 for f in module.functions.values() if not f.is_in_core)}")
            
            # Generate Python code
            generator = PythonCodeGenerator(module)
            python_code = generator.generate()
            
            # Write output
            with open(output_path, 'w') as f:
                f.write(python_code)
            
            print(f"  Generated: {output_path}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse .pto files and generate equivalent PTOFunctionBuilder Python code'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing .pto files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for generated Python files (default: same as input)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)
    
    process_pto_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
