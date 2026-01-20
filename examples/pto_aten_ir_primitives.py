"""
PTO ATen IR Primitives Implementation

This module implements ATen IR primitives using PTO ISA instructions.
Each primitive is implemented as a function that builds a PTO program,
which can then be compiled to ARM64 NEON code.

Based on PyTorch's Core Aten IR and Prims IR:
https://docs.pytorch.org/docs/stable/torch.compiler_ir.html

Categories implemented:
1. Elementwise Unary: abs, neg, exp, log, sqrt, rsqrt, reciprocal, sigmoid, tanh, relu
2. Elementwise Binary: add, sub, mul, div, maximum, minimum, pow
3. Scalar Operations: add_scalar, mul_scalar, div_scalar
4. Reductions: sum, mean, amax, amin
5. Composite Operations: softmax, gelu, silu, sinh, cosh
"""

import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_compile import PTOFunctionBuilder, PTOCompiler
from pto_isa_definition import ElementType, MemorySpace


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TILE_ROWS = 8
DEFAULT_TILE_COLS = 8
DEFAULT_DTYPE = ElementType.F32


# =============================================================================
# Elementwise Unary Primitives
# =============================================================================

def prims_abs(rows=8, cols=8):
    """
    prims.abs: (Tensor self) -> Tensor
    
    Elementwise absolute value.
    PTO Mapping: TABS
    """
    return (PTOFunctionBuilder("prims_abs")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .abs("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def prims_neg(rows=8, cols=8):
    """
    prims.neg: (Tensor self) -> Tensor
    
    Elementwise negation.
    PTO Mapping: TNEG
    """
    return (PTOFunctionBuilder("prims_neg")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .neg("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def prims_exp(rows=8, cols=8):
    """
    prims.exp: (Tensor self) -> Tensor
    
    Elementwise exponential.
    PTO Mapping: TEXP
    """
    return (PTOFunctionBuilder("prims_exp")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .exp("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def prims_log(rows=8, cols=8):
    """
    prims.log: (Tensor self) -> Tensor
    
    Elementwise natural logarithm.
    PTO Mapping: TLOG
    """
    return (PTOFunctionBuilder("prims_log")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .log("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def prims_sqrt(rows=8, cols=8):
    """
    prims.sqrt: (Tensor self) -> Tensor
    
    Elementwise square root.
    PTO Mapping: TSQRT
    """
    return (PTOFunctionBuilder("prims_sqrt")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .sqrt("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def prims_rsqrt(rows=8, cols=8):
    """
    prims.rsqrt: (Tensor self) -> Tensor
    
    Elementwise reciprocal square root: 1/sqrt(x)
    PTO Mapping: TRSQRT
    """
    return (PTOFunctionBuilder("prims_rsqrt")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .rsqrt("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def prims_reciprocal(rows=8, cols=8):
    """
    prims.reciprocal: (Tensor self) -> Tensor
    
    Elementwise reciprocal: 1/x
    PTO Mapping: TRECIP
    """
    return (PTOFunctionBuilder("prims_reciprocal")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .recip("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def aten_relu(rows=8, cols=8):
    """
    aten.relu: (Tensor self) -> Tensor
    
    Elementwise ReLU: max(0, x)
    PTO Mapping: TRELU
    """
    return (PTOFunctionBuilder("aten_relu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .relu("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Composite Unary Operations
# =============================================================================

def aten_sigmoid(rows=8, cols=8):
    """
    aten.sigmoid: (Tensor self) -> Tensor
    
    Elementwise sigmoid: 1 / (1 + exp(-x))
    PTO Mapping: TNEG, TEXP, TADDS, TRECIP (fused)
    """
    return (PTOFunctionBuilder("aten_sigmoid")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("t1", rows, cols, DEFAULT_DTYPE)  # -x
        .tile("t2", rows, cols, DEFAULT_DTYPE)  # exp(-x)
        .tile("t3", rows, cols, DEFAULT_DTYPE)  # 1 + exp(-x)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # sigmoid(x) = 1 / (1 + exp(-x))
        .neg("t1", "x")           # t1 = -x
        .exp("t2", "t1")          # t2 = exp(-x)
        .adds("t3", "t2", 1.0)    # t3 = 1 + exp(-x)
        .recip("result", "t3")    # result = 1 / t3
        
        .store("result", "output", 0, 0)
        
        .build())


def aten_tanh(rows=8, cols=8):
    """
    aten.tanh: (Tensor self) -> Tensor
    
    Elementwise hyperbolic tangent: (exp(2x) - 1) / (exp(2x) + 1)
    Alternative: 2 * sigmoid(2x) - 1
    
    Using: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return (PTOFunctionBuilder("aten_tanh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("numerator", rows, cols, DEFAULT_DTYPE)
        .tile("denominator", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        .exp("exp_x", "x")              # exp(x)
        .neg("neg_x", "x")              # -x
        .exp("exp_neg_x", "neg_x")      # exp(-x)
        .sub("numerator", "exp_x", "exp_neg_x")    # exp(x) - exp(-x)
        .add("denominator", "exp_x", "exp_neg_x") # exp(x) + exp(-x)
        .div("result", "numerator", "denominator")
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Elementwise Binary Primitives
# =============================================================================

def prims_add(rows=8, cols=8):
    """
    prims.add: (Tensor self, Tensor other) -> Tensor
    
    Elementwise addition.
    PTO Mapping: TADD
    """
    return (PTOFunctionBuilder("prims_add")
        .tile("a", rows, cols, DEFAULT_DTYPE)
        .tile("b", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_a", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_b", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .add("result", "a", "b")
        .store("result", "output", 0, 0)
        
        .build())


def prims_sub(rows=8, cols=8):
    """
    prims.sub: (Tensor self, Tensor other) -> Tensor
    
    Elementwise subtraction.
    PTO Mapping: TSUB
    """
    return (PTOFunctionBuilder("prims_sub")
        .tile("a", rows, cols, DEFAULT_DTYPE)
        .tile("b", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_a", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_b", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .sub("result", "a", "b")
        .store("result", "output", 0, 0)
        
        .build())


def prims_mul(rows=8, cols=8):
    """
    prims.mul: (Tensor self, Tensor other) -> Tensor
    
    Elementwise multiplication.
    PTO Mapping: TMUL
    """
    return (PTOFunctionBuilder("prims_mul")
        .tile("a", rows, cols, DEFAULT_DTYPE)
        .tile("b", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_a", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_b", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .mul("result", "a", "b")
        .store("result", "output", 0, 0)
        
        .build())


def prims_div(rows=8, cols=8):
    """
    prims.div: (Tensor self, Tensor other) -> Tensor
    
    Elementwise division.
    PTO Mapping: TDIV
    """
    return (PTOFunctionBuilder("prims_div")
        .tile("a", rows, cols, DEFAULT_DTYPE)
        .tile("b", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_a", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_b", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .div("result", "a", "b")
        .store("result", "output", 0, 0)
        
        .build())


def prims_maximum(rows=8, cols=8):
    """
    prims.maximum: (Tensor self, Tensor other) -> Tensor
    
    Elementwise maximum.
    PTO Mapping: TMAX
    """
    return (PTOFunctionBuilder("prims_maximum")
        .tile("a", rows, cols, DEFAULT_DTYPE)
        .tile("b", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_a", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_b", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .max("result", "a", "b")
        .store("result", "output", 0, 0)
        
        .build())


def prims_minimum(rows=8, cols=8):
    """
    prims.minimum: (Tensor self, Tensor other) -> Tensor
    
    Elementwise minimum.
    PTO Mapping: TMIN
    """
    return (PTOFunctionBuilder("prims_minimum")
        .tile("a", rows, cols, DEFAULT_DTYPE)
        .tile("b", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_a", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_b", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .min("result", "a", "b")
        .store("result", "output", 0, 0)
        
        .build())


def prims_pow(rows=8, cols=8):
    """
    prims.pow: (Tensor base, Tensor exponent) -> Tensor
    
    Elementwise power: base^exponent = exp(exponent * log(base))
    PTO Mapping: TLOG, TMUL, TEXP (composite)
    """
    return (PTOFunctionBuilder("prims_pow")
        .tile("base", rows, cols, DEFAULT_DTYPE)
        .tile("exp", rows, cols, DEFAULT_DTYPE)
        .tile("log_base", rows, cols, DEFAULT_DTYPE)
        .tile("product", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_base", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_exp", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("base", "input_base", 0, 0)
        .load("exp", "input_exp", 0, 0)
        
        # pow(base, exp) = exp(exp * log(base))
        .log("log_base", "base")       # log(base)
        .mul("product", "exp", "log_base")  # exp * log(base)
        .exp("result", "product")      # exp(exp * log(base))
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Scalar Operations
# =============================================================================

def aten_add_scalar(scalar_value, rows=8, cols=8):
    """
    aten.add.Scalar: (Tensor self, Scalar other) -> Tensor
    
    Add scalar to tensor.
    PTO Mapping: TADDS
    """
    return (PTOFunctionBuilder("aten_add_scalar")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .adds("result", "x", scalar_value)
        .store("result", "output", 0, 0)
        
        .build())


def aten_mul_scalar(scalar_value, rows=8, cols=8):
    """
    aten.mul.Scalar: (Tensor self, Scalar other) -> Tensor
    
    Multiply tensor by scalar.
    PTO Mapping: TMULS
    """
    return (PTOFunctionBuilder("aten_mul_scalar")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .muls("result", "x", scalar_value)
        .store("result", "output", 0, 0)
        
        .build())


def aten_div_scalar(scalar_value, rows=8, cols=8):
    """
    aten.div.Scalar: (Tensor self, Scalar other) -> Tensor
    
    Divide tensor by scalar.
    PTO Mapping: TDIVS
    """
    return (PTOFunctionBuilder("aten_div_scalar")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .divs("result", "x", scalar_value)
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Reduction Operations
# =============================================================================

def prims_sum_row(rows=8, cols=8):
    """
    prims.sum: (Tensor inp, int[] dims) -> Tensor
    
    Sum reduction along rows (dim=1).
    PTO Mapping: TROWSUM
    """
    return (PTOFunctionBuilder("prims_sum_row")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, 1, DEFAULT_DTYPE)  # Result has 1 column
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .rowsum("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def prims_sum_col(rows=8, cols=8):
    """
    prims.sum: (Tensor inp, int[] dims) -> Tensor
    
    Sum reduction along columns (dim=0).
    PTO Mapping: TCOLSUM
    """
    return (PTOFunctionBuilder("prims_sum_col")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", 1, cols, DEFAULT_DTYPE)  # Result has 1 row
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .colsum("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def aten_mean_row(rows=8, cols=8):
    """
    aten.mean: (Tensor self, int[] dim) -> Tensor
    
    Mean reduction along rows.
    PTO Mapping: TROWSUM, TDIVS
    """
    return (PTOFunctionBuilder("aten_mean_row")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("sum_result", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .rowsum("sum_result", "x")
        .divs("result", "sum_result", float(cols))  # Divide by number of columns
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Matrix Operations
# =============================================================================

def aten_mm(m=8, k=8, n=8):
    """
    aten.mm: (Tensor self, Tensor mat2) -> Tensor
    
    Matrix multiplication: result = self @ mat2
    PTO Mapping: TMATMUL
    """
    return (PTOFunctionBuilder("aten_mm")
        .tile("a", m, k, DEFAULT_DTYPE)  # [M, K]
        .tile("b", k, n, DEFAULT_DTYPE)  # [K, N]
        .tile("result", m, n, DEFAULT_DTYPE)  # [M, N]
        .memref("input_a", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_b", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .matmul("result", "a", "b")
        .store("result", "output", 0, 0)
        
        .build())


def aten_addmm(m=8, k=8, n=8, beta=1.0, alpha=1.0):
    """
    aten.addmm: (Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor
    
    Matrix multiply with bias: beta * self + alpha * (mat1 @ mat2)
    PTO Mapping: TMULS, TMATMUL, TMULS, TADD
    """
    return (PTOFunctionBuilder("aten_addmm")
        .tile("bias", m, n, DEFAULT_DTYPE)  # [M, N]
        .tile("a", m, k, DEFAULT_DTYPE)     # [M, K]
        .tile("b", k, n, DEFAULT_DTYPE)     # [K, N]
        .tile("mm_result", m, n, DEFAULT_DTYPE)
        .tile("scaled_bias", m, n, DEFAULT_DTYPE)
        .tile("scaled_mm", m, n, DEFAULT_DTYPE)
        .tile("result", m, n, DEFAULT_DTYPE)
        .memref("input_bias", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_a", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_b", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("bias", "input_bias", 0, 0)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        
        # beta * bias
        .muls("scaled_bias", "bias", beta)
        
        # mat1 @ mat2
        .matmul("mm_result", "a", "b")
        
        # alpha * (mat1 @ mat2)
        .muls("scaled_mm", "mm_result", alpha)
        
        # beta * bias + alpha * (mat1 @ mat2)
        .add("result", "scaled_bias", "scaled_mm")
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Activation Functions (Composite)
# =============================================================================

def aten_gelu(rows=8, cols=8):
    """
    aten.gelu: (Tensor self) -> Tensor
    
    Gaussian Error Linear Unit.
    Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Simplified: x * sigmoid(1.702 * x)  (fast approximation)
    """
    return (PTOFunctionBuilder("aten_gelu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("scaled_x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_scaled", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("sigmoid_out", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # GELU ≈ x * sigmoid(1.702 * x)
        .muls("scaled_x", "x", 1.702)        # 1.702 * x
        .neg("neg_scaled", "scaled_x")        # -1.702 * x
        .exp("exp_neg", "neg_scaled")         # exp(-1.702 * x)
        .adds("one_plus", "exp_neg", 1.0)     # 1 + exp(-1.702 * x)
        .recip("sigmoid_out", "one_plus")     # sigmoid(1.702 * x)
        .mul("result", "x", "sigmoid_out")    # x * sigmoid(1.702 * x)
        
        .store("result", "output", 0, 0)
        
        .build())


def aten_silu(rows=8, cols=8):
    """
    aten.silu (Swish): (Tensor self) -> Tensor
    
    Sigmoid Linear Unit: x * sigmoid(x)
    PTO Mapping: TNEG, TEXP, TADDS, TRECIP, TMUL
    """
    return (PTOFunctionBuilder("aten_silu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("sigmoid_out", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # SiLU = x * sigmoid(x)
        .neg("neg_x", "x")                # -x
        .exp("exp_neg", "neg_x")          # exp(-x)
        .adds("one_plus", "exp_neg", 1.0) # 1 + exp(-x)
        .recip("sigmoid_out", "one_plus") # sigmoid(x)
        .mul("result", "x", "sigmoid_out") # x * sigmoid(x)
        
        .store("result", "output", 0, 0)
        
        .build())


def aten_softmax_row(rows=8, cols=8):
    """
    aten.softmax: (Tensor self, int dim) -> Tensor
    
    Softmax along rows (dim=1): exp(x_i) / sum(exp(x))
    
    For numerical stability: exp(x_i - max(x)) / sum(exp(x - max(x)))
    """
    return (PTOFunctionBuilder("aten_softmax_row")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("sum_exp", rows, 1, DEFAULT_DTYPE)  # Row-wise sum
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # Note: For full stability, should subtract max first
        # Simplified version (without max subtraction):
        .exp("exp_x", "x")            # exp(x)
        .rowsum("sum_exp", "exp_x")   # sum(exp(x)) per row
        
        # Division by sum (broadcast sum to all columns)
        # This is simplified - proper implementation needs broadcast
        .div("result", "exp_x", "sum_exp")
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Hyperbolic Functions
# =============================================================================

def aten_sinh(rows=8, cols=8):
    """
    aten.sinh: (Tensor self) -> Tensor
    
    Hyperbolic sine using Taylor expansion:
    sinh(x) = x + x³/6 + x⁵/120 + x⁷/5040 + ...
    
    This is the same as the sinh example in pto_isa_sinh.py
    """
    return (PTOFunctionBuilder("aten_sinh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_squared", rows, cols, DEFAULT_DTYPE)
        .tile("term", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # sinh(x) using Taylor expansion
        .muls("result", "x", 1.0)          # result = x
        .mul("x_squared", "x", "x")        # x²
        .muls("term", "x", 1.0)            # term = x
        
        # Term 2: x³/6
        .mul("term", "term", "x_squared")
        .divs("term", "term", 6.0)
        .add("result", "result", "term")
        
        # Term 3: x⁵/120
        .mul("term", "term", "x_squared")
        .divs("term", "term", 20.0)
        .add("result", "result", "term")
        
        # Term 4: x⁷/5040
        .mul("term", "term", "x_squared")
        .divs("term", "term", 42.0)
        .add("result", "result", "term")
        
        # Term 5: x⁹/362880
        .mul("term", "term", "x_squared")
        .divs("term", "term", 72.0)
        .add("result", "result", "term")
        
        .store("result", "output", 0, 0)
        
        .build())


def aten_cosh(rows=8, cols=8):
    """
    aten.cosh: (Tensor self) -> Tensor
    
    Hyperbolic cosine: cosh(x) = (exp(x) + exp(-x)) / 2
    """
    return (PTOFunctionBuilder("aten_cosh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("sum", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # cosh(x) = (exp(x) + exp(-x)) / 2
        .exp("exp_x", "x")              # exp(x)
        .neg("neg_x", "x")              # -x
        .exp("exp_neg_x", "neg_x")      # exp(-x)
        .add("sum", "exp_x", "exp_neg_x")  # exp(x) + exp(-x)
        .divs("result", "sum", 2.0)     # / 2
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Clamp Operation
# =============================================================================

def aten_clamp(min_val, max_val, rows=8, cols=8):
    """
    aten.clamp: (Tensor self, Scalar? min, Scalar? max) -> Tensor
    
    Clamp values to [min, max] range.
    PTO Mapping: TMAX (with min), TMIN (with max)
    """
    return (PTOFunctionBuilder("aten_clamp")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("min_tile", rows, cols, DEFAULT_DTYPE)
        .tile("max_tile", rows, cols, DEFAULT_DTYPE)
        .tile("clamped_low", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # Create scalar tiles for min/max
        .expands("min_tile", min_val)   # Broadcast min
        .expands("max_tile", max_val)   # Broadcast max
        
        # Clamp: max(min_val, min(max_val, x))
        .max("clamped_low", "x", "min_tile")      # max(x, min_val)
        .min("result", "clamped_low", "max_tile") # min(result, max_val)
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Registry of All Primitives
# =============================================================================

PRIMITIVE_REGISTRY = {
    # Elementwise unary
    "prims.abs": prims_abs,
    "prims.neg": prims_neg,
    "prims.exp": prims_exp,
    "prims.log": prims_log,
    "prims.sqrt": prims_sqrt,
    "prims.rsqrt": prims_rsqrt,
    "prims.reciprocal": prims_reciprocal,
    "aten.relu": aten_relu,
    "aten.sigmoid": aten_sigmoid,
    "aten.tanh": aten_tanh,
    
    # Elementwise binary
    "prims.add": prims_add,
    "prims.sub": prims_sub,
    "prims.mul": prims_mul,
    "prims.div": prims_div,
    "prims.maximum": prims_maximum,
    "prims.minimum": prims_minimum,
    "prims.pow": prims_pow,
    
    # Scalar operations
    "aten.add.Scalar": aten_add_scalar,
    "aten.mul.Scalar": aten_mul_scalar,
    "aten.div.Scalar": aten_div_scalar,
    
    # Reductions
    "prims.sum_row": prims_sum_row,
    "prims.sum_col": prims_sum_col,
    "aten.mean_row": aten_mean_row,
    
    # Matrix operations
    "aten.mm": aten_mm,
    "aten.addmm": aten_addmm,
    
    # Activations
    "aten.gelu": aten_gelu,
    "aten.silu": aten_silu,
    "aten.softmax_row": aten_softmax_row,
    
    # Hyperbolic
    "aten.sinh": aten_sinh,
    "aten.cosh": aten_cosh,
    
    # Special
    "aten.clamp": aten_clamp,
}


# Main: Generate All Primitives for All Backends
# =============================================================================

if __name__ == "__main__":
    import os
    import sys
    
    # Add parent directory for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from pto_compile import generate_all_backends, BACKENDS
    
    print("=" * 70)
    print("PTO ATen IR Primitives - Multi-Backend Code Generation")
    print("=" * 70)
    
    # Base output directory
    OUTPUT_PREFIX = "aten_primitives"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    compiler = PTOCompiler()
    
    # Generate code for all primitives
    primitives_to_generate = [
        # Basic unary
        ("prims.abs", [], {}),
        ("prims.neg", [], {}),
        ("prims.exp", [], {}),
        ("prims.log", [], {}),
        ("prims.sqrt", [], {}),
        ("prims.rsqrt", [], {}),
        ("prims.reciprocal", [], {}),
        ("aten.relu", [], {}),
        
        # Composite unary
        ("aten.sigmoid", [], {}),
        ("aten.tanh", [], {}),
        
        # Binary
        ("prims.add", [], {}),
        ("prims.sub", [], {}),
        ("prims.mul", [], {}),
        ("prims.div", [], {}),
        ("prims.maximum", [], {}),
        ("prims.minimum", [], {}),
        ("prims.pow", [], {}),
        
        # Scalar
        ("aten.add.Scalar", [2.0], {}),
        ("aten.mul.Scalar", [0.5], {}),
        ("aten.div.Scalar", [4.0], {}),
        
        # Activations
        ("aten.gelu", [], {}),
        ("aten.silu", [], {}),
        
        # Hyperbolic
        ("aten.sinh", [], {}),
        ("aten.cosh", [], {}),
        
        # Matrix
        ("aten.mm", [], {'m': 8, 'k': 8, 'n': 8}),
        
        # Reduction
        ("prims.sum_row", [], {}),
        ("aten.mean_row", [], {}),
    ]
    
    print(f"\nGenerating {len(primitives_to_generate)} primitives for {len(BACKENDS)} backends...")
    print(f"Backends: {', '.join(BACKENDS.keys())}")
    print()
    
    total_files = 0
    
    for prim_name, args, kwargs in primitives_to_generate:
        safe_name = prim_name.replace(".", "_").replace("Scalar", "scalar")
        
        try:
            # Get the program builder
            builder_func = PRIMITIVE_REGISTRY[prim_name]
            if prim_name in ("aten.add.Scalar", "aten.mul.Scalar", "aten.div.Scalar"):
                program = builder_func(args[0])
            elif prim_name == "aten.mm":
                program = builder_func(**kwargs)
            else:
                program = builder_func()
            
            print(f"[{prim_name}]")
            
            # Generate for all backends
            results = generate_all_backends(
                program, 
                OUTPUT_PREFIX,
                output_base_dir=SCRIPT_DIR,
                enable_fusion=True
            )
            
            total_files += len(results)
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'=' * 70}")
    print("Generation Complete!")
    print(f"Total files generated: {total_files}")
    print(f"Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{OUTPUT_PREFIX}/")
    print(f"  - output_pto/{OUTPUT_PREFIX}/")
    print("=" * 70)
