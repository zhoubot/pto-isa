"""
PTO ISA Compiler

This module provides the compiler infrastructure for the PTO (Programmable Tensor Operations)
Domain Specific Language. It handles parsing, validation, optimization, and code generation
for PTO programs.

Key Features:
- DSL parsing and AST construction
- Type checking and validation
- Loop unrolling and optimization
- Code generation to PTO assembly
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from enum import Enum
import re

from pto_isa_definition import (
    # Types
    ElementType, MemorySpace, CompareMode, RoundMode, TMovMode,
    TileShape, TileType, MemRefType, EventType,
    
    # Operands
    TileOperand, ScalarOperand, MemRefOperand, IndexOperand, ImmediateOperand, Operand,
    
    # Base classes
    PTOInstruction, TileInstruction, ScalarInstruction, ControlFlowInstruction,
    
    # All instructions
    ALL_INSTRUCTIONS, TILE_INSTRUCTIONS, SCALAR_INSTRUCTIONS, CONTROL_FLOW_INSTRUCTIONS,
    
    # Loop constructs
    TileLoop, NestedTileLoop, FOR, ENDFOR, WHILE, DO, ENDWHILE, IF, ELSE, ENDIF,
    
    # Tile instructions
    TLOAD, TSTORE, TADD, TSUB, TMUL, TDIV, TMATMUL, TMATMUL_ACC,
    TROWSUM, TCOLSUM, TRELU, TSQRT, TEXP, TLOG,
    # Additional unary operations
    TABS, TNEG, TRSQRT, TRECIP,
    # Max/Min operations
    TMAX, TMIN,
    # Broadcast operations
    TEXPANDS, TROWEXPAND, TCOLEXPAND,
    TROWEXPANDSUB, TROWEXPANDDIV, TROWEXPANDMUL,
    # Scalar operations
    TADDS, TMULS, TDIVS,
    
    # Helper functions
    tile, scalar, index, memref, imm,
)


# =============================================================================
# Compiler Error Types
# =============================================================================

class CompilerError(Exception):
    """Base class for compiler errors."""
    pass


class ParseError(CompilerError):
    """Error during parsing."""
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(f"Parse error at line {line}, col {col}: {message}")


class TypeError(CompilerError):
    """Type checking error."""
    def __init__(self, message: str, instruction: Optional[PTOInstruction] = None):
        self.instruction = instruction
        super().__init__(f"Type error: {message}")


class ValidationError(CompilerError):
    """Validation error."""
    pass


# =============================================================================
# Symbol Table
# =============================================================================

@dataclass
class Symbol:
    """A symbol in the symbol table."""
    name: str
    symbol_type: str  # "tile", "scalar", "memref", "index"
    data_type: Any    # TileType, ElementType, MemRefType, etc.
    is_const: bool = False
    value: Optional[Any] = None


class SymbolTable:
    """
    Symbol table for managing variables and their types.
    
    Supports nested scopes for loop constructs.
    """
    
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]
    
    def push_scope(self):
        """Enter a new scope."""
        self.scopes.append({})
    
    def pop_scope(self):
        """Exit current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()
    
    def define(self, name: str, symbol: Symbol):
        """Define a symbol in the current scope."""
        self.scopes[-1][name] = symbol
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol, searching from innermost to outermost scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def is_defined(self, name: str) -> bool:
        """Check if a symbol is defined in any scope."""
        return self.lookup(name) is not None
    
    def is_defined_in_current_scope(self, name: str) -> bool:
        """Check if a symbol is defined in the current scope."""
        return name in self.scopes[-1]


# =============================================================================
# Program Representation
# =============================================================================

@dataclass
class PTOProgram:
    """
    A complete PTO program.
    
    Contains declarations, instructions, and metadata.
    """
    name: str = "main"
    tile_declarations: Dict[str, TileType] = field(default_factory=dict)
    scalar_declarations: Dict[str, ElementType] = field(default_factory=dict)
    memref_declarations: Dict[str, MemRefType] = field(default_factory=dict)
    instructions: List[PTOInstruction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_tile(self, name: str, rows: int, cols: int, dtype: ElementType = ElementType.F32):
        """Declare a tile variable."""
        self.tile_declarations[name] = TileType.create(rows, cols, dtype)
    
    def add_scalar(self, name: str, dtype: ElementType = ElementType.F32):
        """Declare a scalar variable."""
        self.scalar_declarations[name] = dtype
    
    def add_memref(self, name: str, space: MemorySpace, dtype: ElementType,
                   shape: Optional[TileShape] = None):
        """Declare a memory reference."""
        self.memref_declarations[name] = MemRefType(space, dtype, shape)
    
    def add_instruction(self, instr: PTOInstruction):
        """Add an instruction to the program."""
        self.instructions.append(instr)
    
    def add_loop(self, loop: Union[TileLoop, NestedTileLoop]):
        """Add a loop construct by expanding it to instructions."""
        self.instructions.extend(loop.to_instructions())


# =============================================================================
# DSL Builder - Fluent Interface
# =============================================================================

class PTOFunctionBuilder:
    """
    Fluent interface for building PTO programs.
    
    Example:
        program = (PTOFunctionBuilder("matmul")
            .tile("a", 64, 64, ElementType.F16)
            .tile("b", 64, 64, ElementType.F16)
            .tile("c", 64, 64, ElementType.F32)
            .memref("mem_a", MemorySpace.GM, ElementType.F16)
            .load("a", "mem_a", 0, 0)
            .matmul("c", "a", "b")
            .store("c", "mem_c", 0, 0)
            .build())
    """
    
    def __init__(self, name: str = "main"):
        self.program = PTOProgram(name=name)
        self.symbol_table = SymbolTable()
        self._loop_stack: List[List[PTOInstruction]] = []
    
    def _get_tile(self, name: str) -> TileOperand:
        """Get a tile operand by name."""
        if name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{name}' not declared")
        return TileOperand(name, self.program.tile_declarations[name])
    
    def _get_scalar(self, name: str) -> ScalarOperand:
        """Get a scalar operand by name."""
        if name not in self.program.scalar_declarations:
            raise ValidationError(f"Scalar '{name}' not declared")
        return ScalarOperand(name, self.program.scalar_declarations[name])
    
    def _get_memref(self, name: str) -> MemRefOperand:
        """Get a memref operand by name."""
        if name not in self.program.memref_declarations:
            raise ValidationError(f"MemRef '{name}' not declared")
        return MemRefOperand(name, self.program.memref_declarations[name])
    
    def _add_instr(self, instr: PTOInstruction):
        """Add instruction to current context (main or loop body)."""
        if self._loop_stack:
            self._loop_stack[-1].append(instr)
        else:
            self.program.add_instruction(instr)
    
    # Declaration methods
    def tile(self, name: str, rows: int, cols: int, 
             dtype: ElementType = ElementType.F32) -> "PTOFunctionBuilder":
        """Declare a tile variable."""
        self.program.add_tile(name, rows, cols, dtype)
        self.symbol_table.define(name, Symbol(name, "tile", TileType.create(rows, cols, dtype)))
        return self
    
    def scalar(self, name: str, dtype: ElementType = ElementType.F32) -> "PTOFunctionBuilder":
        """Declare a scalar variable."""
        self.program.add_scalar(name, dtype)
        self.symbol_table.define(name, Symbol(name, "scalar", dtype))
        return self
    
    def memref(self, name: str, space: MemorySpace = MemorySpace.GM,
               dtype: ElementType = ElementType.F32,
               shape: Optional[Tuple[int, int]] = None) -> "PTOFunctionBuilder":
        """Declare a memory reference."""
        tile_shape = TileShape(*shape) if shape else None
        self.program.add_memref(name, space, dtype, tile_shape)
        self.symbol_table.define(name, Symbol(name, "memref", MemRefType(space, dtype, tile_shape)))
        return self
    
    # Tile memory operations
    def load(self, dst: str, src_mem: str, row: int = 0, col: int = 0) -> "PTOFunctionBuilder":
        """Load data from memory into a tile."""
        self._add_instr(TLOAD(
            dst=self._get_tile(dst),
            src_mem=self._get_memref(src_mem),
            row_offset=ImmediateOperand(row),
            col_offset=ImmediateOperand(col)
        ))
        return self
    
    def store(self, src: str, dst_mem: str, row: int = 0, col: int = 0) -> "PTOFunctionBuilder":
        """Store data from a tile into memory."""
        self._add_instr(TSTORE(
            src=self._get_tile(src),
            dst_mem=self._get_memref(dst_mem),
            row_offset=ImmediateOperand(row),
            col_offset=ImmediateOperand(col)
        ))
        return self
    
    # Tile arithmetic operations
    def add(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise add of two tiles."""
        self._add_instr(TADD(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def sub(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise subtract of two tiles."""
        self._add_instr(TSUB(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def mul(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise multiply of two tiles."""
        self._add_instr(TMUL(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def div(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise divide of two tiles."""
        self._add_instr(TDIV(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    # Scalar operations (tile op scalar)
    def _make_scalar_operand(self, value: float) -> ScalarOperand:
        """Create a scalar operand for an immediate value."""
        # Create a unique name for the immediate value
        name = f"_imm_{abs(hash(value)) % 10000}"
        return ScalarOperand(name=str(value), element_type=ElementType.F32)
    
    def adds(self, dst: str, src: str, scalar: float) -> "PTOFunctionBuilder":
        """Add scalar to all elements of tile."""
        self._add_instr(TADDS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    def muls(self, dst: str, src: str, scalar: float) -> "PTOFunctionBuilder":
        """Multiply all elements of tile by scalar."""
        self._add_instr(TMULS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    def divs(self, dst: str, src: str, scalar: float) -> "PTOFunctionBuilder":
        """Divide all elements of tile by scalar."""
        self._add_instr(TDIVS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    # Matrix operations
    def matmul(self, dst: str, a: str, b: str) -> "PTOFunctionBuilder":
        """Matrix multiply."""
        self._add_instr(TMATMUL(
            dst=self._get_tile(dst),
            a=self._get_tile(a),
            b=self._get_tile(b)
        ))
        return self
    
    def matmul_acc(self, dst: str, acc: str, a: str, b: str) -> "PTOFunctionBuilder":
        """Matrix multiply with accumulator."""
        self._add_instr(TMATMUL_ACC(
            dst=self._get_tile(dst),
            acc=self._get_tile(acc),
            a=self._get_tile(a),
            b=self._get_tile(b)
        ))
        return self
    
    # Activation functions
    def relu(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply ReLU activation."""
        self._add_instr(TRELU(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def exp(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply exponential."""
        self._add_instr(TEXP(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def log(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply natural logarithm."""
        self._add_instr(TLOG(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def sqrt(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply square root."""
        self._add_instr(TSQRT(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def abs(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply absolute value."""
        self._add_instr(TABS(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def neg(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply negation."""
        self._add_instr(TNEG(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def rsqrt(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply reciprocal square root (1/sqrt(x))."""
        self._add_instr(TRSQRT(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def recip(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply reciprocal (1/x)."""
        self._add_instr(TRECIP(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    # Binary max/min operations
    def max(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise maximum of two tiles."""
        self._add_instr(TMAX(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def min(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise minimum of two tiles."""
        self._add_instr(TMIN(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    # Broadcast operations
    def expands(self, dst: str, value: float) -> "PTOFunctionBuilder":
        """Broadcast a scalar value to all elements of a tile."""
        self._add_instr(TEXPANDS(
            dst=self._get_tile(dst),
            scalar=self._make_scalar_operand(value)
        ))
        return self
    
    def rowexpand(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Broadcast first element of each row across the row."""
        self._add_instr(TROWEXPAND(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def colexpand(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Broadcast first element of each column across the column."""
        self._add_instr(TCOLEXPAND(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    # Row-wise broadcast operations (src1 is [N,1], broadcast to [N,M])
    def rowexpandsub(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Row-wise broadcast subtract: dst = src0 - broadcast(src1)."""
        self._add_instr(TROWEXPANDSUB(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def rowexpanddiv(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Row-wise broadcast divide: dst = src0 / broadcast(src1)."""
        self._add_instr(TROWEXPANDDIV(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def rowexpandmul(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Row-wise broadcast multiply: dst = src0 * broadcast(src1)."""
        self._add_instr(TROWEXPANDMUL(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    # Reduction operations
    def rowsum(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Sum reduction across rows."""
        self._add_instr(TROWSUM(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def colsum(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Sum reduction across columns."""
        self._add_instr(TCOLSUM(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    # Loop constructs
    def for_loop(self, iv_name: str, lb: int, ub: int, step: int = 1) -> "PTOFunctionBuilder":
        """Begin a FOR loop."""
        self.symbol_table.push_scope()
        self.symbol_table.define(iv_name, Symbol(iv_name, "index", ElementType.INDEX))
        
        self._add_instr(FOR(
            iv=IndexOperand(iv_name),
            lb=ImmediateOperand(lb),
            ub=ImmediateOperand(ub),
            step=ImmediateOperand(step)
        ))
        self._loop_stack.append([])
        return self
    
    def end_for(self) -> "PTOFunctionBuilder":
        """End a FOR loop."""
        if not self._loop_stack:
            raise ValidationError("ENDFOR without matching FOR")
        
        loop_body = self._loop_stack.pop()
        
        # If there's still a parent loop, add body and ENDFOR to parent
        # Otherwise, add to program
        if self._loop_stack:
            # We're inside another loop, add to parent loop's body
            for instr in loop_body:
                self._loop_stack[-1].append(instr)
            self._loop_stack[-1].append(ENDFOR())
        else:
            # No parent loop, add directly to program
            for instr in loop_body:
                self.program.add_instruction(instr)
            self.program.add_instruction(ENDFOR())
        
        self.symbol_table.pop_scope()
        return self
    
    def tile_loop(self, iv_name: str, tile_name: str, 
                  dimension: str = "rows", step: int = 1) -> "PTOFunctionBuilder":
        """
        Begin a loop that iterates based on tile dimensions.
        
        Args:
            iv_name: Name of induction variable
            tile_name: Name of tile to get dimensions from
            dimension: "rows" or "cols"
            step: Loop step
        """
        if tile_name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{tile_name}' not declared")
        
        tile_type = self.program.tile_declarations[tile_name]
        bound = tile_type.shape.rows if dimension == "rows" else tile_type.shape.cols
        
        return self.for_loop(iv_name, 0, bound, step)
    
    def nested_tile_loop(self, outer_iv: str, inner_iv: str, tile_name: str,
                         outer_step: int = 1, inner_step: int = 1) -> "PTOFunctionBuilder":
        """
        Begin a 2-level nested loop that iterates over tile dimensions.
        
        Outer loop iterates over rows, inner loop over columns.
        """
        if tile_name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{tile_name}' not declared")
        
        tile_type = self.program.tile_declarations[tile_name]
        
        # Start outer loop
        self.for_loop(outer_iv, 0, tile_type.shape.rows, outer_step)
        # Start inner loop
        self.for_loop(inner_iv, 0, tile_type.shape.cols, inner_step)
        
        return self
    
    def end_nested_loop(self) -> "PTOFunctionBuilder":
        """End a 2-level nested loop."""
        self.end_for()  # End inner
        self.end_for()  # End outer
        return self
    
    # Build the program
    def build(self) -> PTOProgram:
        """Build and return the program."""
        if self._loop_stack:
            raise ValidationError("Unclosed loop constructs")
        return self.program


# =============================================================================
# Type Checker
# =============================================================================

class TypeChecker:
    """
    Type checker for PTO programs.
    
    Validates that all operations have compatible types.
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.errors: List[str] = []
    
    def check(self) -> bool:
        """Run type checking, return True if no errors."""
        self.errors = []
        
        for instr in self.program.instructions:
            self._check_instruction(instr)
        
        return len(self.errors) == 0
    
    def _check_instruction(self, instr: PTOInstruction):
        """Type check a single instruction."""
        if isinstance(instr, TADD) or isinstance(instr, TSUB) or isinstance(instr, TMUL):
            self._check_binary_tile_op(instr)
        elif isinstance(instr, TMATMUL):
            self._check_matmul(instr)
    
    def _check_binary_tile_op(self, instr):
        """Check binary tile operation types match."""
        if hasattr(instr, 'src0') and hasattr(instr, 'src1'):
            src0_type = instr.src0.tile_type
            src1_type = instr.src1.tile_type
            
            if src0_type.shape != src1_type.shape:
                self.errors.append(
                    f"{instr.opcode}: Shape mismatch - {src0_type.shape} vs {src1_type.shape}"
                )
            
            if src0_type.element_type != src1_type.element_type:
                self.errors.append(
                    f"{instr.opcode}: Element type mismatch - "
                    f"{src0_type.element_type.value} vs {src1_type.element_type.value}"
                )
    
    def _check_matmul(self, instr: TMATMUL):
        """Check matrix multiply dimension compatibility."""
        a_type = instr.a.tile_type
        b_type = instr.b.tile_type
        
        if a_type.shape.cols != b_type.shape.rows:
            self.errors.append(
                f"TMATMUL: Incompatible dimensions - "
                f"A cols ({a_type.shape.cols}) != B rows ({b_type.shape.rows})"
            )
    
    def get_errors(self) -> List[str]:
        """Get list of type errors."""
        return self.errors


# =============================================================================
# Code Generator
# =============================================================================

class CodeGenerator:
    """
    Code generator for PTO programs.
    
    Generates PTO assembly from a validated program.
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.output: List[str] = []
        self.indent_level = 0
    
    def generate(self) -> str:
        """Generate PTO assembly code."""
        self.output = []
        self._emit_header()
        self._emit_function_start()
        self._emit_declarations()
        self._emit_instructions()
        self._emit_footer()
        return "\n".join(self.output)
    
    def _emit(self, line: str):
        """Emit a line of code with proper indentation."""
        indent = "  " * self.indent_level
        self.output.append(f"{indent}{line}")
    
    def _emit_header(self):
        """Emit program header."""
        self._emit(f"// PTO Program: {self.program.name}")
        self._emit(f"// Generated by PTO ISA Compiler")
        self._emit("")
    
    def _emit_function_start(self):
        """Emit function definition with name and parameters."""
        # Collect memory references for function parameters
        memref_params = []
        for name, memref_type in self.program.memref_declarations.items():
            memref_params.append(f"%{name}: {memref_type}")
        
        # Generate function signature
        if memref_params:
            params_str = ", ".join(memref_params)
            self._emit(f"func @{self.program.name}({params_str}) {{")
        else:
            self._emit(f"func @{self.program.name}() {{")
        
        self.indent_level += 1
    
    def _emit_declarations(self):
        """Emit variable declarations."""
        self._emit("// Tile Declarations")
        for name, tile_type in self.program.tile_declarations.items():
            self._emit(f"%{name} = alloc_tile : {tile_type}")
        
        if self.program.scalar_declarations:
            self._emit("")
            self._emit("// Scalar Declarations")
            for name, dtype in self.program.scalar_declarations.items():
                self._emit(f"%{name} = alloc_scalar : {dtype.value}")
        
        self._emit("")
    
    def _emit_instructions(self):
        """Emit program instructions."""
        self._emit("// Instructions")
        for instr in self.program.instructions:
            self._emit_instruction(instr)
    
    def _emit_instruction(self, instr: PTOInstruction):
        """Emit a single instruction."""
        if isinstance(instr, FOR):
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ENDFOR):
            self.indent_level = max(1, self.indent_level - 1)  # Keep at least 1 for function body
            self._emit(instr.to_pto_as())
        elif isinstance(instr, IF):
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ELSE):
            self.indent_level = max(1, self.indent_level - 1)
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ENDIF):
            self.indent_level = max(1, self.indent_level - 1)
            self._emit(instr.to_pto_as())
        else:
            self._emit(instr.to_pto_as())
    
    def _emit_footer(self):
        """Emit program footer."""
        self._emit("")
        self._emit("return")
        self.indent_level = 0
        self._emit("}")


# =============================================================================
# Optimizer
# =============================================================================

class Optimizer:
    """
    Basic optimizer for PTO programs.
    
    Performs simple optimizations like:
    - Dead code elimination
    - Constant folding
    - Loop unrolling
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
    
    def optimize(self, unroll_threshold: int = 4) -> PTOProgram:
        """Apply optimizations and return optimized program."""
        optimized = PTOProgram(
            name=self.program.name,
            tile_declarations=self.program.tile_declarations.copy(),
            scalar_declarations=self.program.scalar_declarations.copy(),
            memref_declarations=self.program.memref_declarations.copy(),
            metadata=self.program.metadata.copy()
        )
        
        optimized.instructions = self._optimize_instructions(
            self.program.instructions,
            unroll_threshold
        )
        
        return optimized
    
    def _optimize_instructions(self, instructions: List[PTOInstruction], 
                                unroll_threshold: int) -> List[PTOInstruction]:
        """Optimize a list of instructions."""
        result = []
        i = 0
        
        while i < len(instructions):
            instr = instructions[i]
            
            # Check for small loops that can be unrolled
            if isinstance(instr, FOR):
                loop_end, loop_body = self._find_loop_end(instructions, i)
                if loop_end > i:
                    iterations = self._get_loop_iterations(instr)
                    if iterations is not None and iterations <= unroll_threshold:
                        # Unroll the loop
                        unrolled = self._unroll_loop(instr, loop_body, iterations)
                        result.extend(unrolled)
                        i = loop_end + 1
                        continue
            
            result.append(instr)
            i += 1
        
        return result
    
    def _find_loop_end(self, instructions: List[PTOInstruction], 
                       start: int) -> Tuple[int, List[PTOInstruction]]:
        """Find the matching ENDFOR and extract loop body."""
        depth = 1
        body = []
        i = start + 1
        
        while i < len(instructions) and depth > 0:
            instr = instructions[i]
            if isinstance(instr, FOR):
                depth += 1
            elif isinstance(instr, ENDFOR):
                depth -= 1
                if depth == 0:
                    return i, body
            
            if depth > 0:
                body.append(instr)
            i += 1
        
        return -1, []
    
    def _get_loop_iterations(self, for_instr: FOR) -> Optional[int]:
        """Get the number of iterations for a FOR loop if computable."""
        if isinstance(for_instr.lb, ImmediateOperand) and \
           isinstance(for_instr.ub, ImmediateOperand) and \
           isinstance(for_instr.step, ImmediateOperand):
            lb = for_instr.lb.value
            ub = for_instr.ub.value
            step = for_instr.step.value
            if step > 0:
                return (ub - lb + step - 1) // step
        return None
    
    def _unroll_loop(self, for_instr: FOR, body: List[PTOInstruction], 
                     iterations: int) -> List[PTOInstruction]:
        """Unroll a loop by duplicating body instructions."""
        result = []
        
        # For now, just return the loop as-is (full unrolling is complex)
        # A real implementation would substitute the induction variable
        result.append(for_instr)
        result.extend(body)
        result.append(ENDFOR())
        
        return result


# =============================================================================
# Compiler Driver
# =============================================================================

class PTOCompiler:
    """
    Main compiler driver for PTO programs.
    
    Orchestrates parsing, type checking, optimization, and code generation.
    """
    
    def __init__(self, optimize: bool = True, unroll_threshold: int = 4):
        self.optimize = optimize
        self.unroll_threshold = unroll_threshold
    
    def compile(self, program: PTOProgram) -> str:
        """
        Compile a PTO program to assembly.
        
        Args:
            program: The PTO program to compile
            
        Returns:
            Generated PTO assembly code
            
        Raises:
            TypeError: If type checking fails
            ValidationError: If validation fails
        """
        # Type check
        type_checker = TypeChecker(program)
        if not type_checker.check():
            errors = type_checker.get_errors()
            raise TypeError(f"Type checking failed:\n" + "\n".join(errors))
        
        # Optimize if enabled
        if self.optimize:
            optimizer = Optimizer(program)
            program = optimizer.optimize(self.unroll_threshold)
        
        # Generate code
        generator = CodeGenerator(program)
        return generator.generate()
    
    def compile_and_save(self, program: PTOProgram, output_path: str):
        """Compile a program and save to file."""
        code = self.compile(program)
        with open(output_path, 'w') as f:
            f.write(code)


# =============================================================================
# Loop Fusion - Instruction Classification
# =============================================================================

class OpCategory(Enum):
    """Categories of PTO operations for fusion analysis."""
    ELEMENTWISE_BINARY = "binary"    # TADD, TSUB, TMUL, TDIV, TMAX, TMIN
    ELEMENTWISE_UNARY = "unary"      # TABS, TNEG, TRECIP, TEXP, TLOG, TSQRT, TRELU
    ELEMENTWISE_SCALAR = "scalar"    # TADDS, TMULS, TDIVS, etc.
    BROADCAST = "broadcast"          # TEXPANDS
    REDUCTION = "reduction"          # TROWSUM, TCOLSUM, TSUM
    MATMUL = "matmul"               # TMATMUL
    MEMORY = "memory"               # TLOAD, TSTORE
    CONTROL_FLOW = "control"        # FOR, ENDFOR
    DECLARATION = "decl"            # TILE_DECL, SCALAR_DECL
    OTHER = "other"                 # Unknown/passthrough


# Classify opcodes into categories
OPCODE_CATEGORY = {
    # Elementwise binary
    "TADD": OpCategory.ELEMENTWISE_BINARY,
    "TSUB": OpCategory.ELEMENTWISE_BINARY,
    "TMUL": OpCategory.ELEMENTWISE_BINARY,
    "TDIV": OpCategory.ELEMENTWISE_BINARY,
    "TMAX": OpCategory.ELEMENTWISE_BINARY,
    "TMIN": OpCategory.ELEMENTWISE_BINARY,
    "TAND": OpCategory.ELEMENTWISE_BINARY,
    "TOR": OpCategory.ELEMENTWISE_BINARY,
    "TXOR": OpCategory.ELEMENTWISE_BINARY,
    
    # Elementwise unary
    "TABS": OpCategory.ELEMENTWISE_UNARY,
    "TNEG": OpCategory.ELEMENTWISE_UNARY,
    "TRECIP": OpCategory.ELEMENTWISE_UNARY,
    "TEXP": OpCategory.ELEMENTWISE_UNARY,
    "TLOG": OpCategory.ELEMENTWISE_UNARY,
    "TSQRT": OpCategory.ELEMENTWISE_UNARY,
    "TRSQRT": OpCategory.ELEMENTWISE_UNARY,
    "TRELU": OpCategory.ELEMENTWISE_UNARY,
    
    # Elementwise with scalar
    "TADDS": OpCategory.ELEMENTWISE_SCALAR,
    "TSUBS": OpCategory.ELEMENTWISE_SCALAR,
    "TMULS": OpCategory.ELEMENTWISE_SCALAR,
    "TDIVS": OpCategory.ELEMENTWISE_SCALAR,
    "TMAXS": OpCategory.ELEMENTWISE_SCALAR,
    "TMINS": OpCategory.ELEMENTWISE_SCALAR,
    
    # Broadcast
    "TEXPANDS": OpCategory.BROADCAST,
    
    # Reduction (fusion barrier)
    "TROWSUM": OpCategory.REDUCTION,
    "TCOLSUM": OpCategory.REDUCTION,
    "TSUM": OpCategory.REDUCTION,
    
    # Matrix ops (fusion barrier)
    "TMATMUL": OpCategory.MATMUL,
    
    # Memory
    "TLOAD": OpCategory.MEMORY,
    "TSTORE": OpCategory.MEMORY,
    
    # Control flow
    "FOR": OpCategory.CONTROL_FLOW,
    "ENDFOR": OpCategory.CONTROL_FLOW,
    
    # Declarations
    "TILE_DECL": OpCategory.DECLARATION,
    "SCALAR_DECL": OpCategory.DECLARATION,
}


def get_category(opcode: str) -> OpCategory:
    """Get the category of an opcode."""
    return OPCODE_CATEGORY.get(opcode, OpCategory.OTHER)


def is_fusable(opcode: str) -> bool:
    """Check if an operation can be fused (elementwise operations + memory)."""
    category = get_category(opcode)
    return category in {
        OpCategory.ELEMENTWISE_BINARY,
        OpCategory.ELEMENTWISE_UNARY,
        OpCategory.ELEMENTWISE_SCALAR,
        OpCategory.BROADCAST,
        OpCategory.MEMORY,  # TLOAD/TSTORE can be fused with same-shape ops
    }


def is_fusion_barrier(opcode: str) -> bool:
    """Check if an operation is a fusion barrier (stops fusion)."""
    category = get_category(opcode)
    return category in {
        OpCategory.REDUCTION,
        OpCategory.MATMUL,
        OpCategory.CONTROL_FLOW,
    }


# =============================================================================
# Loop Fusion - IR Classes
# =============================================================================

@dataclass
class FusionTileShape:
    """Shape of a tile for fusion analysis."""
    rows: int
    cols: int
    dtype: str = "f32"
    
    def __hash__(self):
        return hash((self.rows, self.cols, self.dtype))
    
    def __eq__(self, other):
        if not isinstance(other, FusionTileShape):
            return False
        return self.rows == other.rows and self.cols == other.cols and self.dtype == other.dtype


# Alias for backward compatibility
TileShape = FusionTileShape


@dataclass
class FusableOp:
    """
    A single fusable operation.
    
    Attributes:
        opcode: The operation code (TADD, TMUL, etc.)
        dst: Destination tile name
        operands: List of operand names/values
        shape: Shape of the operation
        raw_instr: Original parsed instruction
    """
    opcode: str
    dst: str
    operands: List[str]
    shape: FusionTileShape
    raw_instr: Any = None
    
    def get_reads(self) -> set:
        """Get the set of tiles read by this operation."""
        reads = set()
        category = get_category(self.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            reads.add(self.operands[0])
            reads.add(self.operands[1])
        elif category == OpCategory.ELEMENTWISE_UNARY:
            reads.add(self.operands[0])
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            reads.add(self.operands[0])
        elif category == OpCategory.BROADCAST:
            pass  # scalar only
        elif category == OpCategory.MEMORY:
            if self.opcode == "TLOAD":
                pass  # reads from memory, not tile
            elif self.opcode == "TSTORE":
                reads.add(self.operands[0])  # source tile
        
        return reads
    
    def get_writes(self) -> set:
        """Get the set of tiles written by this operation."""
        if self.opcode == "TSTORE":
            return set()  # writes to memory, not tile
        if self.dst:
            return {self.dst}
        return set()


@dataclass
class FusedLoop:
    """
    A fused loop containing multiple operations.
    
    All operations in a fused loop:
    - Have the same shape
    - Are elementwise operations
    - Can be executed in order within the same loop iteration
    """
    shape: FusionTileShape
    operations: List[FusableOp] = field(default_factory=list)
    
    def add_op(self, op: FusableOp):
        """Add an operation to the fused loop."""
        self.operations.append(op)
    
    def can_fuse(self, op: FusableOp) -> bool:
        """Check if an operation can be fused into this loop."""
        if op.shape != self.shape:
            return False
        if not is_fusable(op.opcode):
            return False
        return True
    
    def __len__(self):
        return len(self.operations)


@dataclass  
class FusionBarrier:
    """Represents a non-fusable operation that acts as a barrier."""
    raw_instr: Any
    opcode: str


# =============================================================================
# Loop Fusion - Optimizer
# =============================================================================

class LoopFusionOptimizer:
    """
    Optimizes PTO instruction sequences by fusing consecutive loops.
    
    Algorithm:
    1. Parse instructions into a sequence of FusableOp and FusionBarrier
    2. Group consecutive FusableOps with same shape into FusedLoops
    3. Respect data dependencies within fused groups
    """
    
    def __init__(self, tile_info: Dict[str, Any]):
        """
        Initialize the optimizer.
        
        Args:
            tile_info: Dictionary mapping tile names to tile info objects
        """
        self.tile_info = tile_info
        self.stats = {
            "original_ops": 0,
            "fused_loops": 0,
            "fusion_savings": 0,
        }
    
    def get_shape(self, tile_name: str) -> Optional[FusionTileShape]:
        """Get the shape of a tile."""
        info = self.tile_info.get(tile_name)
        if info:
            return FusionTileShape(info.rows, info.cols, info.dtype)
        return None
    
    def instr_to_fusable_op(self, instr) -> Optional[FusableOp]:
        """Convert a parsed instruction to a FusableOp if possible."""
        if not is_fusable(instr.opcode):
            return None
        
        # Handle TLOAD/TSTORE specially
        if instr.opcode == "TLOAD":
            shape = self.get_shape(instr.dst)
            if shape is None:
                return None
            return FusableOp(
                opcode=instr.opcode,
                dst=instr.dst,
                operands=list(instr.operands),
                shape=shape,
                raw_instr=instr
            )
        elif instr.opcode == "TSTORE":
            src_tile = instr.operands[0] if instr.operands else None
            shape = self.get_shape(src_tile) if src_tile else None
            if shape is None:
                return None
            return FusableOp(
                opcode=instr.opcode,
                dst=instr.dst,
                operands=list(instr.operands),
                shape=shape,
                raw_instr=instr
            )
        
        # Get shape from destination tile
        shape = self.get_shape(instr.dst) if instr.dst else None
        if shape is None:
            for op in instr.operands:
                shape = self.get_shape(op)
                if shape:
                    break
        
        if shape is None:
            shape = FusionTileShape(4, 4, "f32")
        
        return FusableOp(
            opcode=instr.opcode,
            dst=instr.dst,
            operands=list(instr.operands),
            shape=shape,
            raw_instr=instr
        )
    
    def check_dependency(self, group: List[FusableOp], new_op: FusableOp) -> bool:
        """Check if adding new_op to group would violate data dependencies."""
        return True  # Elementwise ops can always be fused if same shape
    
    def fuse_instructions(self, instructions: List) -> List:
        """Fuse consecutive fusable instructions into FusedLoops."""
        result = []
        current_fused: Optional[FusedLoop] = None
        
        self.stats["original_ops"] = len(instructions)
        
        for instr in instructions:
            if instr.opcode in ("TILE_DECL", "SCALAR_DECL", "PASSTHROUGH"):
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(instr)
                continue
            
            if is_fusion_barrier(instr.opcode):
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(FusionBarrier(raw_instr=instr, opcode=instr.opcode))
                continue
            
            fusable_op = self.instr_to_fusable_op(instr)
            
            if fusable_op is None:
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(FusionBarrier(raw_instr=instr, opcode=instr.opcode))
                continue
            
            if current_fused is None:
                current_fused = FusedLoop(shape=fusable_op.shape)
                current_fused.add_op(fusable_op)
            elif current_fused.can_fuse(fusable_op) and \
                 self.check_dependency(current_fused.operations, fusable_op):
                current_fused.add_op(fusable_op)
            else:
                result.append(current_fused)
                current_fused = FusedLoop(shape=fusable_op.shape)
                current_fused.add_op(fusable_op)
        
        if current_fused and len(current_fused) > 0:
            result.append(current_fused)
        
        self.stats["fused_loops"] = sum(1 for r in result if isinstance(r, FusedLoop))
        fused_op_count = sum(len(r.operations) for r in result if isinstance(r, FusedLoop))
        self.stats["fusion_savings"] = fused_op_count - self.stats["fused_loops"]
        
        return result
    
    def optimize(self, instructions: List) -> List:
        """Apply loop fusion optimization."""
        return self.fuse_instructions(instructions)
    
    def print_stats(self):
        """Print optimization statistics."""
        print(f"Loop Fusion Statistics:")
        print(f"  Original operations: {self.stats['original_ops']}")
        print(f"  Fused loops created: {self.stats['fused_loops']}")
        print(f"  Loop overhead saved: {self.stats['fusion_savings']}")


# =============================================================================
# Loop Fusion - Code Generator
# =============================================================================

class FusedCodeGenerator:
    """Generates ARM64 NEON code for fused loops."""
    
    def __init__(self):
        self.indent_level = 0
        self.var_counter = 0
    
    def _indent(self) -> str:
        return "    " * self.indent_level
    
    def _get_unique_var(self, prefix: str = "_v") -> str:
        name = f"{prefix}{self.var_counter}"
        self.var_counter += 1
        return name
    
    def generate_fused_loop(self, fused: FusedLoop) -> List[str]:
        """Generate code for a fused loop."""
        lines = []
        indent = self._indent()
        
        rows = fused.shape.rows
        cols = fused.shape.cols
        dtype = fused.shape.dtype
        
        vec_lanes = 4 if dtype == "f32" else 8 if dtype == "f16" else 4
        suffix = "f32" if dtype == "f32" else "f16" if dtype == "f16" else "f32"
        vec_type = f"float32x4_t" if dtype == "f32" else f"float16x8_t" if dtype == "f16" else "float32x4_t"
        
        op_names = [f"{op.dst}={op.opcode}({','.join(op.operands)})" for op in fused.operations]
        lines.append(f"{indent}// FUSED LOOP ({len(fused.operations)} ops): {'; '.join(op_names)}")
        
        # Pre-compute scalar broadcast vectors
        scalar_vars = {}
        for op in fused.operations:
            if get_category(op.opcode) == OpCategory.ELEMENTWISE_SCALAR:
                scalar_val = op.operands[1]
                if scalar_val not in scalar_vars:
                    var_name = self._get_unique_var("_vs")
                    scalar_vars[scalar_val] = var_name
                    lines.append(f"{indent}{vec_type} {var_name} = vdupq_n_{suffix}({scalar_val});")
            elif op.opcode == "TEXPANDS":
                scalar_val = op.operands[0]
                if scalar_val not in scalar_vars:
                    var_name = self._get_unique_var("_vs")
                    scalar_vars[scalar_val] = var_name
                    lines.append(f"{indent}{vec_type} {var_name} = vdupq_n_{suffix}({scalar_val});")
        
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        self.indent_level += 1
        indent = self._indent()
        
        lines.append(f"{indent}int _col;")
        lines.append(f"{indent}// Vectorized loop")
        lines.append(f"{indent}for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        self.indent_level += 1
        
        for op in fused.operations:
            vec_lines = self._gen_vectorized_op(op, suffix, vec_type, scalar_vars)
            lines.extend(vec_lines)
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        lines.append(f"{indent}// Scalar cleanup")
        lines.append(f"{indent}for (; _col < {cols}; _col++) {{")
        self.indent_level += 1
        
        for op in fused.operations:
            scalar_lines = self._gen_scalar_op(op)
            lines.extend(scalar_lines)
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        return lines
    
    def _gen_vectorized_op(self, op: FusableOp, suffix: str, vec_type: str,
                           scalar_vars: Dict[str, str]) -> List[str]:
        """Generate vectorized code for a single operation within a fused loop."""
        lines = []
        indent = self._indent()
        category = get_category(op.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            v0, v1, vr = self._get_unique_var("_v"), self._get_unique_var("_v"), self._get_unique_var("_vr")
            neon_op = self._get_neon_binary_op(op.opcode, suffix)
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {v1} = vld1q_{suffix}(&{op.operands[1]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_op}({v0}, {v1});")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.ELEMENTWISE_UNARY:
            v0, vr = self._get_unique_var("_v"), self._get_unique_var("_vr")
            neon_code = self._get_neon_unary_op(op.opcode, suffix, v0, vec_type)
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_code};")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            v0, vr = self._get_unique_var("_v"), self._get_unique_var("_vr")
            vs = scalar_vars.get(op.operands[1], "_vs")
            neon_op = self._get_neon_scalar_op(op.opcode, suffix)
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_op}({v0}, {vs});")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.BROADCAST:
            vs = scalar_vars.get(op.operands[0], "_vs")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vs});")
        
        elif category == OpCategory.MEMORY:
            if op.opcode == "TLOAD":
                vr = self._get_unique_var("_vl")
                memref = op.operands[0]
                lines.append(f"{indent}{vec_type} {vr} = vld1q_{suffix}(&{memref}[_row * {op.shape.cols} + _col]);")
                lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
            elif op.opcode == "TSTORE":
                vs = self._get_unique_var("_vs")
                src_tile, memref = op.operands[0], op.dst
                lines.append(f"{indent}{vec_type} {vs} = vld1q_{suffix}(&{src_tile}[_row][_col]);")
                lines.append(f"{indent}vst1q_{suffix}(&{memref}[_row * {op.shape.cols} + _col], {vs});")
        
        return lines
    
    def _gen_scalar_op(self, op: FusableOp) -> List[str]:
        """Generate scalar code for a single operation within a fused loop."""
        lines = []
        indent = self._indent()
        category = get_category(op.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            c_op = self._get_c_binary_op(op.opcode)
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]}[_row][_col] {c_op} {op.operands[1]}[_row][_col];")
        
        elif category == OpCategory.ELEMENTWISE_UNARY:
            c_expr = self._get_c_unary_expr(op.opcode, f"{op.operands[0]}[_row][_col]")
            lines.append(f"{indent}{op.dst}[_row][_col] = {c_expr};")
        
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            c_op = self._get_c_scalar_op(op.opcode)
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]}[_row][_col] {c_op} {op.operands[1]};")
        
        elif category == OpCategory.BROADCAST:
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]};")
        
        elif category == OpCategory.MEMORY:
            if op.opcode == "TLOAD":
                memref = op.operands[0]
                lines.append(f"{indent}{op.dst}[_row][_col] = {memref}[_row * {op.shape.cols} + _col];")
            elif op.opcode == "TSTORE":
                src_tile, memref = op.operands[0], op.dst
                lines.append(f"{indent}{memref}[_row * {op.shape.cols} + _col] = {src_tile}[_row][_col];")
        
        return lines
    
    def _get_neon_binary_op(self, opcode: str, suffix: str) -> str:
        ops = {"TADD": f"vaddq_{suffix}", "TSUB": f"vsubq_{suffix}", 
               "TMUL": f"vmulq_{suffix}", "TDIV": f"vdivq_{suffix}",
               "TMAX": f"vmaxq_{suffix}", "TMIN": f"vminq_{suffix}"}
        return ops.get(opcode, f"vaddq_{suffix}")
    
    def _get_neon_unary_op(self, opcode: str, suffix: str, v: str, vec_type: str) -> str:
        ops = {"TABS": f"vabsq_{suffix}({v})", "TNEG": f"vnegq_{suffix}({v})",
               "TSQRT": f"vsqrtq_{suffix}({v})", "TRSQRT": f"vrsqrteq_{suffix}({v})",
               "TRELU": f"vmaxq_{suffix}({v}, vdupq_n_{suffix}(0.0f))"}
        return ops.get(opcode, f"{v}")
    
    def _get_neon_scalar_op(self, opcode: str, suffix: str) -> str:
        ops = {"TADDS": f"vaddq_{suffix}", "TSUBS": f"vsubq_{suffix}",
               "TMULS": f"vmulq_{suffix}", "TDIVS": f"vdivq_{suffix}",
               "TMAXS": f"vmaxq_{suffix}", "TMINS": f"vminq_{suffix}"}
        return ops.get(opcode, f"vaddq_{suffix}")
    
    def _get_c_binary_op(self, opcode: str) -> str:
        ops = {"TADD": "+", "TSUB": "-", "TMUL": "*", "TDIV": "/"}
        return ops.get(opcode, "+")
    
    def _get_c_unary_expr(self, opcode: str, operand: str) -> str:
        exprs = {"TABS": f"fabsf({operand})", "TNEG": f"-{operand}",
                 "TRECIP": f"1.0f / {operand}", "TEXP": f"expf({operand})",
                 "TLOG": f"logf({operand})", "TSQRT": f"sqrtf({operand})",
                 "TRSQRT": f"1.0f / sqrtf({operand})", "TRELU": f"fmaxf({operand}, 0.0f)"}
        return exprs.get(opcode, operand)
    
    def _get_c_scalar_op(self, opcode: str) -> str:
        ops = {"TADDS": "+", "TSUBS": "-", "TMULS": "*", "TDIVS": "/"}
        return ops.get(opcode, "+")


# =============================================================================
# Multi-Backend Code Generation
# =============================================================================

# Import additional types for code generation
from pto_isa_definition import (
    ARM64_TYPE_MAP, CUDA_TYPE_MAP, ASCEND_TYPE_MAP,
    arm64_generate_header, cuda_generate_header, ascend_generate_header,
)

# Backend Configuration
BACKENDS = {
    "arm64": {
        "name": "ARM64 NEON",
        "suffix": "_arm64",
        "extension": ".c",
        "header_func": arm64_generate_header,
        "type_map": ARM64_TYPE_MAP,
    },
    "cuda": {
        "name": "NVIDIA CUDA",
        "suffix": "_cuda",
        "extension": ".cu",
        "header_func": cuda_generate_header,
        "type_map": CUDA_TYPE_MAP,
    },
    "ascend910b": {
        "name": "Huawei Ascend 910B",
        "suffix": "_ascend910b",
        "extension": ".cpp",
        "header_func": ascend_generate_header,
        "type_map": ASCEND_TYPE_MAP,
    },
}


@dataclass
class MockTileInfo:
    """Tile info for code generation."""
    name: str
    rows: int
    cols: int
    dtype: str


@dataclass
class MockInstruction:
    """Mock instruction for fusion optimizer."""
    opcode: str
    dst: str
    operands: list
    raw_line: str = ""


def convert_program_to_mock_instructions(program):
    """Convert PTOProgram to mock instructions for fusion optimizer."""
    tile_info = {}
    for name, tile_type in program.tile_declarations.items():
        tile_info[name] = MockTileInfo(
            name=name,
            rows=tile_type.shape.rows,
            cols=tile_type.shape.cols,
            dtype=tile_type.element_type.value
        )
    
    mock_instructions = []
    for instr in program.instructions:
        opcode = instr.__class__.__name__
        
        if opcode in ("TADD", "TSUB", "TMUL", "TDIV", "TMAX", "TMIN"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src0.name, instr.src1.name]
            ))
        elif opcode in ("TADDS", "TSUBS", "TMULS", "TDIVS"):
            scalar_str = instr.scalar.name
            if not scalar_str.endswith("f"):
                scalar_str += "f"
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src.name, scalar_str]
            ))
        elif opcode in ("TEXP", "TLOG", "TSQRT", "TRSQRT", "TRELU", "TABS", "TNEG", "TRECIP"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src.name]
            ))
        elif opcode == "TEXPANDS":
            scalar_str = instr.scalar.name
            if not scalar_str.endswith("f"):
                scalar_str += "f"
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[scalar_str]
            ))
        elif opcode in ("TROWSUM", "TCOLSUM"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src.name]
            ))
        elif opcode in ("TROWEXPANDSUB", "TROWEXPANDDIV", "TROWEXPANDMUL"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src0.name, instr.src1.name]
            ))
        elif opcode == "TMATMUL":
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.a.name, instr.b.name]
            ))
        elif opcode == "TLOAD":
            mock_instructions.append(MockInstruction(
                opcode="TLOAD", dst=instr.dst.name,
                operands=[instr.src_mem.name, "0", "0"]
            ))
        elif opcode == "TSTORE":
            mock_instructions.append(MockInstruction(
                opcode="TSTORE", dst=instr.dst_mem.name,
                operands=[instr.src.name, "0", "0"]
            ))
    
    return tile_info, mock_instructions


def _gen_arm64_barrier_op(instr, rows, cols, dtype, tile_info):
    """Generate ARM64 code for barrier operations (non-fusable)."""
    lines = []
    c_type = ARM64_TYPE_MAP.get(dtype, "float")
    
    if instr.opcode == "TLOAD":
        dst, src_mem = instr.dst, instr.operands[0]
        lines.append(f"// TLOAD: {dst} = load({src_mem})")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"        {dst}[_row][_col] = {src_mem}[_row * {cols} + _col];")
        lines.append(f"    }}}}")
        
    elif instr.opcode == "TSTORE":
        dst_mem, src = instr.dst, instr.operands[0]
        lines.append(f"// TSTORE: store({src}) -> {dst_mem}")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"        {dst_mem}[_row * {cols} + _col] = {src}[_row][_col];")
        lines.append(f"    }}}}")
        
    elif instr.opcode == "TROWSUM":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_cols = src_info.cols if src_info else cols
        lines.append(f"// TROWSUM: {dst} = rowsum({src})")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    {c_type} _sum = 0.0f;")
        lines.append(f"    for (int _col = 0; _col < {src_cols}; _col++) {{")
        lines.append(f"        _sum += {src}[_row][_col];")
        lines.append(f"    }}")
        lines.append(f"    {dst}[_row][0] = _sum;}}")
        
    elif instr.opcode == "TCOLSUM":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_rows = src_info.rows if src_info else rows
        lines.append(f"// TCOLSUM: {dst} = colsum({src})")
        lines.append(f"for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"    {c_type} _sum = 0.0f;")
        lines.append(f"    for (int _row = 0; _row < {src_rows}; _row++) {{")
        lines.append(f"        _sum += {src}[_row][_col];")
        lines.append(f"    }}")
        lines.append(f"    {dst}[0][_col] = _sum;}}")
        
    elif instr.opcode in ("TROWEXPANDSUB", "TROWEXPANDDIV", "TROWEXPANDMUL"):
        dst, src0, src1 = instr.dst, instr.operands[0], instr.operands[1]
        op_map = {"TROWEXPANDSUB": "-", "TROWEXPANDDIV": "/", "TROWEXPANDMUL": "*"}
        op = op_map.get(instr.opcode, "-")
        lines.append(f"// {instr.opcode}: {dst} = {src0} {op} broadcast({src1})")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    {c_type} _broadcast_val = {src1}[_row][0];")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"        {dst}[_row][_col] = {src0}[_row][_col] {op} _broadcast_val;")
        lines.append(f"    }}}}")
        
    elif instr.opcode == "TMATMUL":
        dst, a, b = instr.dst, instr.operands[0], instr.operands[1]
        a_info = tile_info.get(a)
        k = a_info.cols if a_info else 8
        lines.append(f"// TMATMUL: {dst} = {a} @ {b}")
        lines.append(f"for (int _i = 0; _i < {rows}; _i++) {{")
        lines.append(f"    for (int _j = 0; _j < {cols}; _j++) {{")
        lines.append(f"        {c_type} _sum = 0.0f;")
        lines.append(f"        for (int _k = 0; _k < {k}; _k++) {{")
        lines.append(f"            _sum += {a}[_i][_k] * {b}[_k][_j];}}")
        lines.append(f"        {dst}[_i][_j] = _sum;}}}}")
    else:
        lines.append(f"// {instr.opcode}: Not implemented")
    
    return lines


def _gen_cuda_single_op(instr, tile_info):
    """Generate a single CUDA operation."""
    op, dst = instr.opcode, f"{instr.dst}[_row][_col]"
    src0 = src1 = ""
    
    if len(instr.operands) >= 1:
        src0 = f"{instr.operands[0]}[_row][_col]"
    if len(instr.operands) >= 2:
        src1 = instr.operands[1]
        if not src1.endswith("f") and not src1.replace(".", "").replace("-", "").isdigit():
            src1 = f"{src1}[_row][_col]"
    
    # Get tile shape for memory operations
    dst_info = tile_info.get(instr.dst)
    cols = dst_info.cols if dst_info else 8
    
    if op == "TADD": return f"{dst} = {src0} + {src1};"
    elif op == "TSUB": return f"{dst} = {src0} - {src1};"
    elif op == "TMUL": return f"{dst} = {src0} * {src1};"
    elif op == "TDIV": return f"{dst} = {src0} / {src1};"
    elif op == "TMAX": return f"{dst} = fmaxf({src0}, {src1});"
    elif op == "TMIN": return f"{dst} = fminf({src0}, {src1});"
    elif op == "TABS": return f"{dst} = fabsf({src0});"
    elif op == "TNEG": return f"{dst} = -{src0};"
    elif op == "TRECIP": return f"{dst} = 1.0f / {src0};"
    elif op == "TEXP": return f"{dst} = __expf({src0});"
    elif op == "TLOG": return f"{dst} = __logf({src0});"
    elif op == "TSQRT": return f"{dst} = __fsqrt_rn({src0});"
    elif op == "TRSQRT": return f"{dst} = __frsqrt_rn({src0});"
    elif op == "TRELU": return f"{dst} = fmaxf({src0}, 0.0f);"
    elif op == "TADDS": return f"{dst} = {src0} + {src1};"
    elif op == "TSUBS": return f"{dst} = {src0} - {src1};"
    elif op == "TMULS": return f"{dst} = {src0} * {src1};"
    elif op == "TDIVS": return f"{dst} = {src0} / {src1};"
    elif op == "TEXPANDS": return f"{dst} = {instr.operands[0]};"
    elif op == "TLOAD": return f"{dst} = {instr.operands[0]}[_row * {cols} + _col];"
    elif op == "TSTORE": 
        src_info = tile_info.get(instr.operands[0])
        src_cols = src_info.cols if src_info else cols
        return f"{instr.dst}[_row * {src_cols} + _col] = {instr.operands[0]}[_row][_col];"
    return f"// Unknown op: {op}"


def _gen_ascend_single_op(instr, tile_info):
    """Generate a single Ascend C vector operation."""
    op, dst = instr.opcode, instr.dst
    src0 = instr.operands[0] if len(instr.operands) >= 1 else ""
    src1 = instr.operands[1] if len(instr.operands) >= 2 else ""
    
    ops_map = {
        "TADD": f"Add({dst}, {src0}, {src1}, 64);",
        "TSUB": f"Sub({dst}, {src0}, {src1}, 64);",
        "TMUL": f"Mul({dst}, {src0}, {src1}, 64);",
        "TDIV": f"Div({dst}, {src0}, {src1}, 64);",
        "TMAX": f"Max({dst}, {src0}, {src1}, 64);",
        "TMIN": f"Min({dst}, {src0}, {src1}, 64);",
        "TABS": f"Abs({dst}, {src0}, 64);",
        "TNEG": f"Neg({dst}, {src0}, 64);",
        "TRECIP": f"Reciprocal({dst}, {src0}, 64);",
        "TEXP": f"Exp({dst}, {src0}, 64);",
        "TLOG": f"Ln({dst}, {src0}, 64);",
        "TSQRT": f"Sqrt({dst}, {src0}, 64);",
        "TRSQRT": f"Rsqrt({dst}, {src0}, 64);",
        "TRELU": f"Relu({dst}, {src0}, 64);",
        "TADDS": f"Adds({dst}, {src0}, {src1}, 64);",
        "TSUBS": f"Subs({dst}, {src0}, {src1}, 64);",
        "TMULS": f"Muls({dst}, {src0}, {src1}, 64);",
        "TDIVS": f"Divs({dst}, {src0}, {src1}, 64);",
        "TEXPANDS": f"Duplicate({dst}, {src0}, 64);",
    }
    return ops_map.get(op, f"// {op}: Operation")


class MultiBackendCodeGenerator:
    """
    Unified multi-backend code generator for PTO programs.
    
    Generates optimized code for multiple target architectures:
    - ARM64 NEON (Apple Silicon, ARM servers)
    - NVIDIA CUDA (GPU computing)
    - Huawei Ascend 910B (NPU/AI accelerator)
    """
    
    def __init__(self, enable_fusion: bool = True):
        self.enable_fusion = enable_fusion
    
    def generate_arm64(self, program) -> str:
        """Generate ARM64 NEON code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        lines = [f"// PTO Program: {program.name}", arm64_generate_header()]
        
        # Collect memory references for function parameters
        memref_params = []
        for name, memref_type in program.memref_declarations.items():
            c_type = ARM64_TYPE_MAP.get(memref_type.element_type.value, "float")
            memref_params.append(f"{c_type}* {name}")
        
        # Generate function signature
        if memref_params:
            func_params = ", ".join(memref_params)
            lines.append(f"void {program.name}({func_params}) {{")
        else:
            lines.append(f"void {program.name}(void) {{")
        
        # Declare tiles as local variables inside the function
        for name, info in tile_info.items():
            c_type = ARM64_TYPE_MAP.get(info.dtype, "float")
            lines.append(f"    {c_type} {name}[{info.rows}][{info.cols}];")
        lines.append("")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"    // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            fused_codegen = FusedCodeGenerator()
            for item in fused_result:
                if isinstance(item, FusedLoop):
                    # Indent the fused loop code
                    fused_lines = fused_codegen.generate_fused_loop(item)
                    for fused_line in fused_lines:
                        lines.append(f"    {fused_line}" if fused_line else "")
                    lines.append("")
                elif isinstance(item, FusionBarrier):
                    instr = item.raw_instr
                    info = tile_info.get(instr.dst)
                    rows = info.rows if info else 8
                    cols = info.cols if info else 8
                    dtype = info.dtype if info else "f32"
                    # Indent the barrier code
                    barrier_lines = _gen_arm64_barrier_op(instr, rows, cols, dtype, tile_info)
                    for barrier_line in barrier_lines:
                        lines.append(f"    {barrier_line}" if barrier_line else "")
                    lines.append("")
        
        lines.append("}")
        return "\n".join(lines)
    
    def generate_cuda(self, program) -> str:
        """Generate NVIDIA CUDA code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        lines = [f"// PTO Program: {program.name}", cuda_generate_header()]
        
        # Declare tiles as __device__ arrays
        for name, info in tile_info.items():
            c_type = CUDA_TYPE_MAP.get(info.dtype, "float")
            lines.append(f"__device__ {c_type} {name}[{info.rows}][{info.cols}];")
        lines.append("")
        
        # Collect memory references for kernel parameters
        memref_params = []
        memref_types = {}
        for name, memref_type in program.memref_declarations.items():
            c_type = CUDA_TYPE_MAP.get(memref_type.element_type.value, "float")
            memref_params.append(f"{c_type}* {name}")
            memref_types[name] = c_type
        
        # Generate kernel signature with memory reference parameters
        if memref_params:
            kernel_params = ", ".join(memref_params)
            lines.append(f"__global__ void {program.name}_kernel({kernel_params}) {{")
        else:
            lines.append(f"__global__ void {program.name}_kernel() {{")
        lines.append("    int _row = threadIdx.y + blockIdx.y * blockDim.y;")
        lines.append("    int _col = threadIdx.x + blockIdx.x * blockDim.x;\n")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"    // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            for item in fused_result:
                if isinstance(item, FusedLoop):
                    ops_desc = "; ".join([f"{op.dst}={op.opcode}(...)" for op in item.operations])
                    lines.append(f"    // FUSED ({len(item.operations)} ops): {ops_desc}")
                    lines.append(f"    if (_row < {item.shape.rows} && _col < {item.shape.cols}) {{")
                    for op in item.operations:
                        lines.append(f"        {_gen_cuda_single_op(op, tile_info)}")
                    lines.append("    }\n")
                elif isinstance(item, FusionBarrier):
                    instr = item.raw_instr
                    lines.append(f"    // BARRIER: {instr.opcode}\n")
        
        lines.append("}\n")
        
        # Generate host wrapper function with memory reference parameters
        if memref_params:
            wrapper_params = ", ".join(memref_params)
            kernel_args = ", ".join(program.memref_declarations.keys())
            lines.append(f"void {program.name}({wrapper_params}) {{")
            lines.append("    dim3 block(8, 8);")
            lines.append("    dim3 grid(1, 1);")
            lines.append(f"    {program.name}_kernel<<<grid, block>>>({kernel_args});")
            lines.append("    cudaDeviceSynchronize();\n}")
        else:
            lines.append(f"void {program.name}() {{")
            lines.append("    dim3 block(8, 8);")
            lines.append("    dim3 grid(1, 1);")
            lines.append(f"    {program.name}_kernel<<<grid, block>>>();")
            lines.append("    cudaDeviceSynchronize();\n}")
        
        return "\n".join(lines)
    
    def generate_ascend(self, program) -> str:
        """Generate Huawei Ascend 910B (Ascend C) code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        lines = [f"// PTO Program: {program.name}", ascend_generate_header()]
        
        lines.append(f"class {program.name}Kernel {{")
        lines.append("public:")
        lines.append(f"    __aicore__ inline {program.name}Kernel() {{}}")
        lines.append("    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {")
        lines.append("        inputGm.SetGlobalBuffer((__gm__ float*)input);")
        lines.append("        outputGm.SetGlobalBuffer((__gm__ float*)output);")
        lines.append("        pipe.InitBuffer(inQueueX, 1, 8 * 8 * sizeof(float));")
        lines.append("        pipe.InitBuffer(outQueueY, 1, 8 * 8 * sizeof(float));")
        lines.append("    }\n")
        lines.append("    __aicore__ inline void Process() {")
        lines.append("        CopyIn(); Compute(); CopyOut();")
        lines.append("    }\n")
        lines.append("private:")
        lines.append("    __aicore__ inline void CopyIn() {")
        lines.append("        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();")
        lines.append("        DataCopy(xLocal, inputGm, 64);")
        lines.append("        inQueueX.EnQue(xLocal);")
        lines.append("    }\n")
        lines.append("    __aicore__ inline void Compute() {")
        lines.append("        LocalTensor<float> xLocal = inQueueX.DeQue<float>();")
        lines.append("        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();\n")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"        // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            for item in fused_result:
                if isinstance(item, FusedLoop):
                    ops_desc = "; ".join([op.opcode for op in item.operations])
                    lines.append(f"        // FUSED ({len(item.operations)} ops): {ops_desc}")
                    for op in item.operations:
                        lines.append(f"        {_gen_ascend_single_op(op, tile_info)}")
                    lines.append("")
                elif isinstance(item, FusionBarrier):
                    lines.append(f"        // BARRIER: {item.raw_instr.opcode}\n")
        
        lines.append("        outQueueY.EnQue(yLocal);")
        lines.append("        inQueueX.FreeTensor(xLocal);")
        lines.append("    }\n")
        lines.append("    __aicore__ inline void CopyOut() {")
        lines.append("        LocalTensor<float> yLocal = outQueueY.DeQue<float>();")
        lines.append("        DataCopy(outputGm, yLocal, 64);")
        lines.append("        outQueueY.FreeTensor(yLocal);")
        lines.append("    }\n")
        lines.append("private:")
        lines.append("    TPipe pipe;")
        lines.append("    TQue<QuePosition::VECIN, 1> inQueueX;")
        lines.append("    TQue<QuePosition::VECOUT, 1> outQueueY;")
        lines.append("    GlobalTensor<float> inputGm;")
        lines.append("    GlobalTensor<float> outputGm;")
        lines.append("};\n")
        
        lines.append(f"extern \"C\" __global__ __aicore__ void {program.name}_kernel(GM_ADDR input, GM_ADDR output) {{")
        lines.append(f"    {program.name}Kernel op;")
        lines.append("    op.Init(input, output);")
        lines.append("    op.Process();")
        lines.append("}")
        
        return "\n".join(lines)
    
    def generate_all(self, program, output_prefix: str, output_base_dir: str = ".") -> Dict[str, str]:
        """
        Generate code for all backends and save to files.
        
        Args:
            program: PTOProgram built using PTOFunctionBuilder
            output_prefix: Category name for output subdirectory (e.g., "sinh_taylor", "aten_primitives")
            output_base_dir: Base directory for output
            
        Output structure:
            output_base_dir/
            ├── output_arm64/{output_prefix}/{program.name}.c
            ├── output_cuda/{output_prefix}/{program.name}.cu
            ├── output_ascend910b/{output_prefix}/{program.name}.cpp
            └── output_pto/{output_prefix}/{program.name}.pto
            
        Returns:
            Dict mapping backend name to output file path
        """
        import os
        results = {}
        
        generators = {
            "arm64": self.generate_arm64,
            "cuda": self.generate_cuda,
            "ascend910b": self.generate_ascend,
        }
        
        for backend_key, backend_info in BACKENDS.items():
            # New structure: output_arm64/category/file.c
            output_dir = os.path.join(output_base_dir, f"output{backend_info['suffix']}", output_prefix)
            os.makedirs(output_dir, exist_ok=True)
            
            code = generators[backend_key](program)
            output_file = os.path.join(output_dir, f"{program.name}{backend_info['extension']}")
            
            with open(output_file, "w") as f:
                f.write(code)
            
            results[backend_key] = output_file
            print(f"  [{backend_info['name']}] -> {output_file}")
        
        # Also save PTO-AS assembly
        compiler = PTOCompiler()
        pto_asm = compiler.compile(program)
        pto_dir = os.path.join(output_base_dir, "output_pto", output_prefix)
        os.makedirs(pto_dir, exist_ok=True)
        pto_file = os.path.join(pto_dir, f"{program.name}.pto")
        with open(pto_file, "w") as f:
            f.write(pto_asm)
        results["pto"] = pto_file
        print(f"  [PTO-AS] -> {pto_file}")
        
        return results


# Convenience functions for backward compatibility
def generate_all_backends(program, output_prefix: str, output_base_dir: str = ".", 
                          enable_fusion: bool = True) -> Dict[str, str]:
    """Generate code for all backends (convenience wrapper)."""
    generator = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return generator.generate_all(program, output_prefix, output_base_dir)


def generate_arm64_code(program, enable_fusion: bool = True) -> str:
    """Generate ARM64 NEON code (convenience wrapper)."""
    generator = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return generator.generate_arm64(program)


def generate_cuda_code(program, enable_fusion: bool = True) -> str:
    """Generate CUDA code (convenience wrapper)."""
    generator = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return generator.generate_cuda(program)


def generate_ascend_code(program, enable_fusion: bool = True) -> str:
    """Generate Ascend C code (convenience wrapper)."""
    generator = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return generator.generate_ascend(program)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Simple matrix multiply
    print("=" * 60)
    print("Example 1: Simple Matrix Multiply")
    print("=" * 60)
    
    program1 = (PTOFunctionBuilder("matmul_example")
        # Declare tiles
        .tile("a", 64, 64, ElementType.F16)
        .tile("b", 64, 64, ElementType.F16)
        .tile("c", 64, 64, ElementType.F32)
        # Declare memory
        .memref("mem_a", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F16)
        .memref("mem_c", MemorySpace.GM, ElementType.F32)
        # Load operands
        .load("a", "mem_a", 0, 0)
        .load("b", "mem_b", 0, 0)
        # Compute
        .matmul("c", "a", "b")
        # Store result
        .store("c", "mem_c", 0, 0)
        .build())
    
    compiler = PTOCompiler()
    code1 = compiler.compile(program1)
    print(code1)
    
    # Example 2: Tiled matrix multiply with nested loops
    print("\n" + "=" * 60)
    print("Example 2: Tiled Matrix Multiply with Nested Loops")
    print("=" * 60)
    
    program2 = (PTOFunctionBuilder("tiled_matmul")
        # Declare tiles (smaller tiles for tiling)
        .tile("a_tile", 16, 16, ElementType.F16)
        .tile("b_tile", 16, 16, ElementType.F16)
        .tile("c_tile", 16, 16, ElementType.F32)
        # Larger tile for iteration bounds
        .tile("full_matrix", 64, 64, ElementType.F32)
        # Memory
        .memref("mem_a", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F16)
        .memref("mem_c", MemorySpace.GM, ElementType.F32)
        # Nested loop over tiles (4x4 tiles to cover 64x64)
        .for_loop("i", 0, 4, 1)  # Outer loop
        .for_loop("j", 0, 4, 1)  # Inner loop
        # Load tiles
        .load("a_tile", "mem_a")
        .load("b_tile", "mem_b")
        # Compute
        .matmul("c_tile", "a_tile", "b_tile")
        # Store
        .store("c_tile", "mem_c")
        .end_for()  # End inner
        .end_for()  # End outer
        .build())
    
    code2 = compiler.compile(program2)
    print(code2)
    
    # Example 3: MLP forward pass
    print("\n" + "=" * 60)
    print("Example 3: MLP Forward Pass")
    print("=" * 60)
    
    program3 = (PTOFunctionBuilder("mlp_forward")
        # Input, weights, bias, output tiles
        .tile("input", 64, 128, ElementType.F16)
        .tile("weight", 128, 64, ElementType.F16)
        .tile("bias", 64, 64, ElementType.F32)
        .tile("output", 64, 64, ElementType.F32)
        .tile("activated", 64, 64, ElementType.F32)
        # Memory refs
        .memref("mem_in", MemorySpace.GM, ElementType.F16)
        .memref("mem_w", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F32)
        .memref("mem_out", MemorySpace.GM, ElementType.F32)
        # Load data
        .load("input", "mem_in")
        .load("weight", "mem_w")
        .load("bias", "mem_b")
        # Linear layer: output = input @ weight
        .matmul("output", "input", "weight")
        # Add bias
        .add("output", "output", "bias")
        # ReLU activation
        .relu("activated", "output")
        # Store result
        .store("activated", "mem_out")
        .build())
    
    code3 = compiler.compile(program3)
    print(code3)
    
    print("\n" + "=" * 60)
    print("Compilation completed successfully!")
    print("=" * 60)
