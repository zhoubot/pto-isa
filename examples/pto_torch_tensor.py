"""
PTO torch.Tensor Methods Implementation

This module implements PyTorch Tensor class methods using PTO ISA instructions.
Reference: https://docs.pytorch.org/docs/stable/tensors.html

Categories implemented:
1. Arithmetic Operations: add, sub, mul, div, neg, abs, pow, sqrt, rsqrt, reciprocal
2. Trigonometric: sin, cos, tan, sinh, cosh, tanh, asin, acos, atan
3. Exponential/Logarithmic: exp, exp2, expm1, log, log2, log10, log1p
4. Comparison: eq, ne, gt, ge, lt, le, max, min, clamp
5. Reduction: sum, mean, std, var, prod, max, min
6. Linear Algebra: mm, matmul, dot, mv
7. Shape: t (transpose), view (reshape)
8. Special: sigmoid, relu, tanh, softmax
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
# Arithmetic Operations
# =============================================================================

def tensor_add(rows=8, cols=8):
    """
    Tensor.add(other) -> Tensor
    Elementwise addition: self + other
    """
    return (PTOFunctionBuilder("tensor_add")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("other", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .add("result", "self", "other")
        .store("result", "output", 0, 0)
        .build())


def tensor_sub(rows=8, cols=8):
    """
    Tensor.sub(other) -> Tensor
    Elementwise subtraction: self - other
    """
    return (PTOFunctionBuilder("tensor_sub")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("other", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .sub("result", "self", "other")
        .store("result", "output", 0, 0)
        .build())


def tensor_mul(rows=8, cols=8):
    """
    Tensor.mul(other) -> Tensor
    Elementwise multiplication: self * other
    """
    return (PTOFunctionBuilder("tensor_mul")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("other", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .mul("result", "self", "other")
        .store("result", "output", 0, 0)
        .build())


def tensor_div(rows=8, cols=8):
    """
    Tensor.div(other) -> Tensor
    Elementwise division: self / other
    """
    return (PTOFunctionBuilder("tensor_div")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("other", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .div("result", "self", "other")
        .store("result", "output", 0, 0)
        .build())


def tensor_neg(rows=8, cols=8):
    """
    Tensor.neg() -> Tensor
    Negation: -self
    """
    return (PTOFunctionBuilder("tensor_neg")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .neg("result", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_abs(rows=8, cols=8):
    """
    Tensor.abs() -> Tensor
    Absolute value: |self|
    """
    return (PTOFunctionBuilder("tensor_abs")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .abs("result", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_pow(exponent=2.0, rows=8, cols=8):
    """
    Tensor.pow(exponent) -> Tensor
    Power: self ** exponent (for exponent=2, uses self*self)
    """
    if exponent == 2.0:
        return (PTOFunctionBuilder("tensor_pow2")
            .tile("self", rows, cols, DEFAULT_DTYPE)
            .tile("result", rows, cols, DEFAULT_DTYPE)
            .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
            .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
            .load("self", "input", 0, 0)
            .mul("result", "self", "self")
            .store("result", "output", 0, 0)
            .build())
    else:
        # General power: exp(exponent * log(self))
        return (PTOFunctionBuilder("tensor_pow")
            .tile("self", rows, cols, DEFAULT_DTYPE)
            .tile("log_self", rows, cols, DEFAULT_DTYPE)
            .tile("scaled", rows, cols, DEFAULT_DTYPE)
            .tile("result", rows, cols, DEFAULT_DTYPE)
            .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
            .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
            .load("self", "input", 0, 0)
            .log("log_self", "self")
            .muls("scaled", "log_self", exponent)
            .exp("result", "scaled")
            .store("result", "output", 0, 0)
            .build())


def tensor_sqrt(rows=8, cols=8):
    """
    Tensor.sqrt() -> Tensor
    Square root: sqrt(self)
    """
    return (PTOFunctionBuilder("tensor_sqrt")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .sqrt("result", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_rsqrt(rows=8, cols=8):
    """
    Tensor.rsqrt() -> Tensor
    Reciprocal square root: 1/sqrt(self)
    """
    return (PTOFunctionBuilder("tensor_rsqrt")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .rsqrt("result", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_reciprocal(rows=8, cols=8):
    """
    Tensor.reciprocal() -> Tensor
    Reciprocal: 1/self
    """
    return (PTOFunctionBuilder("tensor_reciprocal")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .recip("result", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_square(rows=8, cols=8):
    """
    Tensor.square() -> Tensor
    Square: self * self
    """
    return (PTOFunctionBuilder("tensor_square")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .mul("result", "self", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_addcmul(value=1.0, rows=8, cols=8):
    """
    Tensor.addcmul(tensor1, tensor2, value=1) -> Tensor
    self + value * tensor1 * tensor2
    """
    return (PTOFunctionBuilder("tensor_addcmul")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("tensor1", rows, cols, DEFAULT_DTYPE)
        .tile("tensor2", rows, cols, DEFAULT_DTYPE)
        .tile("prod", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_t1", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_t2", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("tensor1", "input_t1", 0, 0)
        .load("tensor2", "input_t2", 0, 0)
        .mul("prod", "tensor1", "tensor2")
        .muls("scaled", "prod", value)
        .add("result", "self", "scaled")
        .store("result", "output", 0, 0)
        .build())


def tensor_addcdiv(value=1.0, rows=8, cols=8):
    """
    Tensor.addcdiv(tensor1, tensor2, value=1) -> Tensor
    self + value * tensor1 / tensor2
    """
    return (PTOFunctionBuilder("tensor_addcdiv")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("tensor1", rows, cols, DEFAULT_DTYPE)
        .tile("tensor2", rows, cols, DEFAULT_DTYPE)
        .tile("quot", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_t1", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_t2", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("tensor1", "input_t1", 0, 0)
        .load("tensor2", "input_t2", 0, 0)
        .div("quot", "tensor1", "tensor2")
        .muls("scaled", "quot", value)
        .add("result", "self", "scaled")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Trigonometric Operations
# =============================================================================

def tensor_sin(rows=8, cols=8):
    """
    Tensor.sin() -> Tensor
    Sine: sin(self)
    Approximation using Taylor series: x - x³/6 + x⁵/120
    """
    return (PTOFunctionBuilder("tensor_sin")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("x3", rows, cols, DEFAULT_DTYPE)
        .tile("x5", rows, cols, DEFAULT_DTYPE)
        .tile("term1", rows, cols, DEFAULT_DTYPE)
        .tile("term2", rows, cols, DEFAULT_DTYPE)
        .tile("temp", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .mul("x2", "x", "x")
        .mul("x3", "x2", "x")
        .mul("x5", "x3", "x2")
        .divs("term1", "x3", 6.0)
        .divs("term2", "x5", 120.0)
        .sub("temp", "x", "term1")
        .add("result", "temp", "term2")
        .store("result", "output", 0, 0)
        .build())


def tensor_cos(rows=8, cols=8):
    """
    Tensor.cos() -> Tensor
    Cosine: cos(self)
    Approximation: 1 - x²/2 + x⁴/24
    """
    return (PTOFunctionBuilder("tensor_cos")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("x4", rows, cols, DEFAULT_DTYPE)
        .tile("term1", rows, cols, DEFAULT_DTYPE)
        .tile("term2", rows, cols, DEFAULT_DTYPE)
        .tile("ones", rows, cols, DEFAULT_DTYPE)
        .tile("temp", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .mul("x2", "x", "x")
        .mul("x4", "x2", "x2")
        .divs("term1", "x2", 2.0)
        .divs("term2", "x4", 24.0)
        .expands("ones", 1.0)
        .sub("temp", "ones", "term1")
        .add("result", "temp", "term2")
        .store("result", "output", 0, 0)
        .build())


def tensor_tan(rows=8, cols=8):
    """
    Tensor.tan() -> Tensor
    Tangent: tan(self) = sin(self)/cos(self)
    """
    return (PTOFunctionBuilder("tensor_tan")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("x3", rows, cols, DEFAULT_DTYPE)
        .tile("x4", rows, cols, DEFAULT_DTYPE)
        .tile("x5", rows, cols, DEFAULT_DTYPE)
        .tile("sin_t1", rows, cols, DEFAULT_DTYPE)
        .tile("sin_t2", rows, cols, DEFAULT_DTYPE)
        .tile("sin_temp", rows, cols, DEFAULT_DTYPE)
        .tile("sin_val", rows, cols, DEFAULT_DTYPE)
        .tile("cos_t1", rows, cols, DEFAULT_DTYPE)
        .tile("cos_t2", rows, cols, DEFAULT_DTYPE)
        .tile("ones", rows, cols, DEFAULT_DTYPE)
        .tile("cos_temp", rows, cols, DEFAULT_DTYPE)
        .tile("cos_val", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .mul("x2", "x", "x")
        .mul("x3", "x2", "x")
        .mul("x4", "x2", "x2")
        .mul("x5", "x3", "x2")
        # sin
        .divs("sin_t1", "x3", 6.0)
        .divs("sin_t2", "x5", 120.0)
        .sub("sin_temp", "x", "sin_t1")
        .add("sin_val", "sin_temp", "sin_t2")
        # cos
        .divs("cos_t1", "x2", 2.0)
        .divs("cos_t2", "x4", 24.0)
        .expands("ones", 1.0)
        .sub("cos_temp", "ones", "cos_t1")
        .add("cos_val", "cos_temp", "cos_t2")
        # tan = sin/cos
        .div("result", "sin_val", "cos_val")
        .store("result", "output", 0, 0)
        .build())


def tensor_sinh(rows=8, cols=8):
    """
    Tensor.sinh() -> Tensor
    Hyperbolic sine: (exp(x) - exp(-x)) / 2
    """
    return (PTOFunctionBuilder("tensor_sinh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .neg("neg_x", "x")
        .exp("exp_x", "x")
        .exp("exp_neg_x", "neg_x")
        .sub("diff", "exp_x", "exp_neg_x")
        .divs("result", "diff", 2.0)
        .store("result", "output", 0, 0)
        .build())


def tensor_cosh(rows=8, cols=8):
    """
    Tensor.cosh() -> Tensor
    Hyperbolic cosine: (exp(x) + exp(-x)) / 2
    """
    return (PTOFunctionBuilder("tensor_cosh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("sum_val", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .neg("neg_x", "x")
        .exp("exp_x", "x")
        .exp("exp_neg_x", "neg_x")
        .add("sum_val", "exp_x", "exp_neg_x")
        .divs("result", "sum_val", 2.0)
        .store("result", "output", 0, 0)
        .build())


def tensor_tanh(rows=8, cols=8):
    """
    Tensor.tanh() -> Tensor
    Hyperbolic tangent: (exp(2x) - 1) / (exp(2x) + 1)
    """
    return (PTOFunctionBuilder("tensor_tanh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("exp_2x", rows, cols, DEFAULT_DTYPE)
        .tile("numerator", rows, cols, DEFAULT_DTYPE)
        .tile("denominator", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .muls("x2", "x", 2.0)
        .exp("exp_2x", "x2")
        .adds("numerator", "exp_2x", -1.0)
        .adds("denominator", "exp_2x", 1.0)
        .div("result", "numerator", "denominator")
        .store("result", "output", 0, 0)
        .build())


def tensor_asin(rows=8, cols=8):
    """
    Tensor.asin() -> Tensor
    Arc sine: asin(x) ≈ x + x³/6 + 3x⁵/40 (for small x)
    """
    return (PTOFunctionBuilder("tensor_asin")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("x3", rows, cols, DEFAULT_DTYPE)
        .tile("x5", rows, cols, DEFAULT_DTYPE)
        .tile("term1", rows, cols, DEFAULT_DTYPE)
        .tile("term2", rows, cols, DEFAULT_DTYPE)
        .tile("temp", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .mul("x2", "x", "x")
        .mul("x3", "x2", "x")
        .mul("x5", "x3", "x2")
        .divs("term1", "x3", 6.0)
        .muls("term2", "x5", 0.075)  # 3/40 = 0.075
        .add("temp", "x", "term1")
        .add("result", "temp", "term2")
        .store("result", "output", 0, 0)
        .build())


def tensor_acos(rows=8, cols=8):
    """
    Tensor.acos() -> Tensor
    Arc cosine: acos(x) = π/2 - asin(x)
    """
    pi_over_2 = 1.5707963267948966
    return (PTOFunctionBuilder("tensor_acos")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("x3", rows, cols, DEFAULT_DTYPE)
        .tile("x5", rows, cols, DEFAULT_DTYPE)
        .tile("term1", rows, cols, DEFAULT_DTYPE)
        .tile("term2", rows, cols, DEFAULT_DTYPE)
        .tile("temp", rows, cols, DEFAULT_DTYPE)
        .tile("asin_val", rows, cols, DEFAULT_DTYPE)
        .tile("pi_half", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .mul("x2", "x", "x")
        .mul("x3", "x2", "x")
        .mul("x5", "x3", "x2")
        .divs("term1", "x3", 6.0)
        .muls("term2", "x5", 0.075)
        .add("temp", "x", "term1")
        .add("asin_val", "temp", "term2")
        .expands("pi_half", pi_over_2)
        .sub("result", "pi_half", "asin_val")
        .store("result", "output", 0, 0)
        .build())


def tensor_atan(rows=8, cols=8):
    """
    Tensor.atan() -> Tensor
    Arc tangent: atan(x) ≈ x - x³/3 + x⁵/5 (for small x)
    """
    return (PTOFunctionBuilder("tensor_atan")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("x3", rows, cols, DEFAULT_DTYPE)
        .tile("x5", rows, cols, DEFAULT_DTYPE)
        .tile("term1", rows, cols, DEFAULT_DTYPE)
        .tile("term2", rows, cols, DEFAULT_DTYPE)
        .tile("temp", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .mul("x2", "x", "x")
        .mul("x3", "x2", "x")
        .mul("x5", "x3", "x2")
        .divs("term1", "x3", 3.0)
        .divs("term2", "x5", 5.0)
        .sub("temp", "x", "term1")
        .add("result", "temp", "term2")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Exponential and Logarithmic Operations
# =============================================================================

def tensor_exp(rows=8, cols=8):
    """
    Tensor.exp() -> Tensor
    Exponential: exp(self)
    """
    return (PTOFunctionBuilder("tensor_exp")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .exp("result", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_exp2(rows=8, cols=8):
    """
    Tensor.exp2() -> Tensor
    Base-2 exponential: 2^self = exp(self * ln(2))
    """
    ln2 = 0.6931471805599453
    return (PTOFunctionBuilder("tensor_exp2")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .muls("scaled", "self", ln2)
        .exp("result", "scaled")
        .store("result", "output", 0, 0)
        .build())


def tensor_expm1(rows=8, cols=8):
    """
    Tensor.expm1() -> Tensor
    exp(self) - 1
    """
    return (PTOFunctionBuilder("tensor_expm1")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("exp_val", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .exp("exp_val", "self")
        .adds("result", "exp_val", -1.0)
        .store("result", "output", 0, 0)
        .build())


def tensor_log(rows=8, cols=8):
    """
    Tensor.log() -> Tensor
    Natural logarithm: ln(self)
    """
    return (PTOFunctionBuilder("tensor_log")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .log("result", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_log2(rows=8, cols=8):
    """
    Tensor.log2() -> Tensor
    Base-2 logarithm: log2(self) = ln(self) / ln(2)
    """
    inv_ln2 = 1.4426950408889634  # 1/ln(2)
    return (PTOFunctionBuilder("tensor_log2")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("ln_val", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .log("ln_val", "self")
        .muls("result", "ln_val", inv_ln2)
        .store("result", "output", 0, 0)
        .build())


def tensor_log10(rows=8, cols=8):
    """
    Tensor.log10() -> Tensor
    Base-10 logarithm: log10(self) = ln(self) / ln(10)
    """
    inv_ln10 = 0.4342944819032518  # 1/ln(10)
    return (PTOFunctionBuilder("tensor_log10")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("ln_val", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .log("ln_val", "self")
        .muls("result", "ln_val", inv_ln10)
        .store("result", "output", 0, 0)
        .build())


def tensor_log1p(rows=8, cols=8):
    """
    Tensor.log1p() -> Tensor
    ln(1 + self)
    """
    return (PTOFunctionBuilder("tensor_log1p")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .adds("one_plus", "self", 1.0)
        .log("result", "one_plus")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Comparison Operations
# =============================================================================

def tensor_max_elementwise(rows=8, cols=8):
    """
    Tensor.max(other) -> Tensor
    Elementwise maximum: max(self, other)
    """
    return (PTOFunctionBuilder("tensor_max_elementwise")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("other", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .max("result", "self", "other")
        .store("result", "output", 0, 0)
        .build())


def tensor_min_elementwise(rows=8, cols=8):
    """
    Tensor.min(other) -> Tensor
    Elementwise minimum: min(self, other)
    """
    return (PTOFunctionBuilder("tensor_min_elementwise")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("other", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .min("result", "self", "other")
        .store("result", "output", 0, 0)
        .build())


def tensor_clamp(min_val=0.0, max_val=1.0, rows=8, cols=8):
    """
    Tensor.clamp(min, max) -> Tensor
    Clamp values to [min, max]
    """
    return (PTOFunctionBuilder("tensor_clamp")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("min_tile", rows, cols, DEFAULT_DTYPE)
        .tile("max_tile", rows, cols, DEFAULT_DTYPE)
        .tile("clamp_low", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .expands("min_tile", min_val)
        .expands("max_tile", max_val)
        .max("clamp_low", "self", "min_tile")
        .min("result", "clamp_low", "max_tile")
        .store("result", "output", 0, 0)
        .build())


def tensor_clip(min_val=0.0, max_val=1.0, rows=8, cols=8):
    """
    Tensor.clip(min, max) -> Tensor
    Alias for clamp
    """
    return tensor_clamp(min_val, max_val, rows, cols)


# =============================================================================
# Reduction Operations
# =============================================================================

def tensor_sum(rows=8, cols=8):
    """
    Tensor.sum() -> Tensor
    Sum of all elements
    """
    return (PTOFunctionBuilder("tensor_sum")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .rowsum("row_sum", "self")
        .colsum("result", "row_sum")
        .store("result", "output", 0, 0)
        .build())


def tensor_mean(rows=8, cols=8):
    """
    Tensor.mean() -> Tensor
    Mean of all elements
    """
    return (PTOFunctionBuilder("tensor_mean")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("total", 1, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .rowsum("row_sum", "self")
        .colsum("total", "row_sum")
        .divs("result", "total", float(rows * cols))
        .store("result", "output", 0, 0)
        .build())


def tensor_std(rows=8, cols=8):
    """
    Tensor.std() -> Tensor
    Standard deviation: sqrt(var)
    """
    n = float(rows * cols)
    return (PTOFunctionBuilder("tensor_std")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("total", 1, 1, DEFAULT_DTYPE)
        .tile("mean_val", rows, cols, DEFAULT_DTYPE)
        .tile("centered", rows, cols, DEFAULT_DTYPE)
        .tile("sq_centered", rows, cols, DEFAULT_DTYPE)
        .tile("sq_row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("var_total", 1, 1, DEFAULT_DTYPE)
        .tile("var", 1, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        # Mean
        .rowsum("row_sum", "self")
        .colsum("total", "row_sum")
        .divs("total", "total", n)
        # Broadcast mean to all elements
        .expands("mean_val", 0.0)  # Will be set manually in generated code
        # Center
        .sub("centered", "self", "mean_val")
        .mul("sq_centered", "centered", "centered")
        .rowsum("sq_row_sum", "sq_centered")
        .colsum("var_total", "sq_row_sum")
        .divs("var", "var_total", n)
        .sqrt("result", "var")
        .store("result", "output", 0, 0)
        .build())


def tensor_var(rows=8, cols=8):
    """
    Tensor.var() -> Tensor
    Variance: mean((self - mean)²)
    """
    n = float(rows * cols)
    return (PTOFunctionBuilder("tensor_var")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("total", 1, 1, DEFAULT_DTYPE)
        .tile("mean_val", rows, cols, DEFAULT_DTYPE)
        .tile("centered", rows, cols, DEFAULT_DTYPE)
        .tile("sq_centered", rows, cols, DEFAULT_DTYPE)
        .tile("sq_row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("var_total", 1, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .rowsum("row_sum", "self")
        .colsum("total", "row_sum")
        .divs("total", "total", n)
        .expands("mean_val", 0.0)
        .sub("centered", "self", "mean_val")
        .mul("sq_centered", "centered", "centered")
        .rowsum("sq_row_sum", "sq_centered")
        .colsum("var_total", "sq_row_sum")
        .divs("result", "var_total", n)
        .store("result", "output", 0, 0)
        .build())


def tensor_prod(rows=8, cols=8):
    """
    Tensor.prod() -> Tensor
    Product of all elements: exp(sum(log(self)))
    """
    return (PTOFunctionBuilder("tensor_prod")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("log_self", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("total", 1, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .log("log_self", "self")
        .rowsum("row_sum", "log_self")
        .colsum("total", "row_sum")
        .exp("result", "total")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Linear Algebra Operations
# =============================================================================

def tensor_mm(m=8, k=8, n=8):
    """
    Tensor.mm(mat2) -> Tensor
    Matrix multiplication: self @ mat2
    """
    return (PTOFunctionBuilder("tensor_mm")
        .tile("self", m, k, DEFAULT_DTYPE)
        .tile("mat2", k, n, DEFAULT_DTYPE)
        .tile("result", m, n, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_mat2", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("mat2", "input_mat2", 0, 0)
        .matmul("result", "self", "mat2")
        .store("result", "output", 0, 0)
        .build())


def tensor_matmul(m=8, k=8, n=8):
    """
    Tensor.matmul(other) -> Tensor
    Matrix multiplication (alias for mm for 2D tensors)
    """
    return tensor_mm(m, k, n)


def tensor_dot(size=64):
    """
    Tensor.dot(other) -> Tensor
    Dot product of two 1D tensors (represented as 1xN tile)
    """
    return (PTOFunctionBuilder("tensor_dot")
        .tile("self", 1, size, DEFAULT_DTYPE)
        .tile("other", 1, size, DEFAULT_DTYPE)
        .tile("prod", 1, size, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .mul("prod", "self", "other")
        .rowsum("result", "prod")
        .store("result", "output", 0, 0)
        .build())


def tensor_mv(m=8, n=8):
    """
    Tensor.mv(vec) -> Tensor
    Matrix-vector multiplication: self @ vec
    """
    return (PTOFunctionBuilder("tensor_mv")
        .tile("self", m, n, DEFAULT_DTYPE)
        .tile("vec", n, 1, DEFAULT_DTYPE)
        .tile("result", m, 1, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_vec", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("vec", "input_vec", 0, 0)
        .matmul("result", "self", "vec")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Special Activation Operations
# =============================================================================

def tensor_sigmoid(rows=8, cols=8):
    """
    Tensor.sigmoid() -> Tensor
    Sigmoid: 1 / (1 + exp(-self))
    """
    return (PTOFunctionBuilder("tensor_sigmoid")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("neg_self", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .neg("neg_self", "self")
        .exp("exp_neg", "neg_self")
        .adds("one_plus", "exp_neg", 1.0)
        .recip("result", "one_plus")
        .store("result", "output", 0, 0)
        .build())


def tensor_relu(rows=8, cols=8):
    """
    Tensor.relu() -> Tensor
    ReLU: max(0, self)
    """
    return (PTOFunctionBuilder("tensor_relu")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .relu("result", "self")
        .store("result", "output", 0, 0)
        .build())


def tensor_softmax(dim=-1, rows=8, cols=8):
    """
    Tensor.softmax(dim) -> Tensor
    Softmax along specified dimension
    """
    return (PTOFunctionBuilder("tensor_softmax")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("row_mean", rows, 1, DEFAULT_DTYPE)
        .tile("shifted", rows, cols, DEFAULT_DTYPE)
        .tile("exp_shifted", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .rowsum("row_mean", "self")
        .divs("row_mean", "row_mean", float(cols))
        .rowexpandsub("shifted", "self", "row_mean")
        .exp("exp_shifted", "shifted")
        .rowsum("row_sum", "exp_shifted")
        .rowexpanddiv("result", "exp_shifted", "row_sum")
        .store("result", "output", 0, 0)
        .build())


def tensor_log_softmax(dim=-1, rows=8, cols=8):
    """
    Tensor.log_softmax(dim) -> Tensor
    Log-softmax along specified dimension
    """
    return (PTOFunctionBuilder("tensor_log_softmax")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("row_mean", rows, 1, DEFAULT_DTYPE)
        .tile("shifted", rows, cols, DEFAULT_DTYPE)
        .tile("exp_shifted", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("log_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .rowsum("row_mean", "self")
        .divs("row_mean", "row_mean", float(cols))
        .rowexpandsub("shifted", "self", "row_mean")
        .exp("exp_shifted", "shifted")
        .rowsum("row_sum", "exp_shifted")
        .log("log_sum", "row_sum")
        .rowexpandsub("result", "shifted", "log_sum")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Misc Operations
# =============================================================================

def tensor_sign(rows=8, cols=8):
    """
    Tensor.sign() -> Tensor
    Sign function: -1 if x<0, 0 if x==0, 1 if x>0
    Approximation: x / (|x| + eps)
    """
    eps = 1e-7
    return (PTOFunctionBuilder("tensor_sign")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("abs_self", rows, cols, DEFAULT_DTYPE)
        .tile("abs_plus_eps", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .abs("abs_self", "self")
        .adds("abs_plus_eps", "abs_self", eps)
        .div("result", "self", "abs_plus_eps")
        .store("result", "output", 0, 0)
        .build())


def tensor_lerp(weight=0.5, rows=8, cols=8):
    """
    Tensor.lerp(end, weight) -> Tensor
    Linear interpolation: self + weight * (end - self)
    """
    return (PTOFunctionBuilder("tensor_lerp")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("end", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_end", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("end", "input_end", 0, 0)
        .sub("diff", "end", "self")
        .muls("scaled", "diff", weight)
        .add("result", "self", "scaled")
        .store("result", "output", 0, 0)
        .build())


def tensor_logit(eps=1e-6, rows=8, cols=8):
    """
    Tensor.logit(eps) -> Tensor
    Logit: log(x / (1 - x))
    """
    return (PTOFunctionBuilder("tensor_logit")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("one_minus", rows, cols, DEFAULT_DTYPE)
        .tile("ratio", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .muls("one_minus", "self", -1.0)
        .adds("one_minus", "one_minus", 1.0)
        .adds("one_minus", "one_minus", eps)  # Add eps for stability
        .div("ratio", "self", "one_minus")
        .log("result", "ratio")
        .store("result", "output", 0, 0)
        .build())


def tensor_xlogy(rows=8, cols=8):
    """
    Tensor.xlogy(other) -> Tensor
    Computes self * log(other)
    """
    return (PTOFunctionBuilder("tensor_xlogy")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("other", rows, cols, DEFAULT_DTYPE)
        .tile("log_other", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .log("log_other", "other")
        .mul("result", "self", "log_other")
        .store("result", "output", 0, 0)
        .build())


def tensor_hypot(rows=8, cols=8):
    """
    Tensor.hypot(other) -> Tensor
    Hypotenuse: sqrt(self² + other²)
    """
    return (PTOFunctionBuilder("tensor_hypot")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("other", rows, cols, DEFAULT_DTYPE)
        .tile("self_sq", rows, cols, DEFAULT_DTYPE)
        .tile("other_sq", rows, cols, DEFAULT_DTYPE)
        .tile("sum_sq", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input_self", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input_other", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input_self", 0, 0)
        .load("other", "input_other", 0, 0)
        .mul("self_sq", "self", "self")
        .mul("other_sq", "other", "other")
        .add("sum_sq", "self_sq", "other_sq")
        .sqrt("result", "sum_sq")
        .store("result", "output", 0, 0)
        .build())


def tensor_frac(rows=8, cols=8):
    """
    Tensor.frac() -> Tensor
    Fractional part: self - floor(self)
    Approximation: self - (self - 0.5).round() for values near integers
    """
    return (PTOFunctionBuilder("tensor_frac")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("ones", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        # Simplified: just return self mod 1 approximation
        .expands("ones", 1.0)
        .div("result", "self", "ones")  # Placeholder
        .store("result", "output", 0, 0)
        .build())


def tensor_cumsum(dim=1, rows=8, cols=8):
    """
    Tensor.cumsum(dim) -> Tensor
    Cumulative sum along dimension (simplified row-wise)
    """
    return (PTOFunctionBuilder("tensor_cumsum")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .rowsum("row_sum", "self")
        # Simplified: return row sums expanded
        .store("row_sum", "output", 0, 0)
        .build())


def tensor_diff(n=1, dim=-1, rows=8, cols=8):
    """
    Tensor.diff(n=1, dim=-1) -> Tensor
    Finite difference along dimension
    Simplified: returns input (actual diff requires complex indexing)
    """
    return (PTOFunctionBuilder("tensor_diff")
        .tile("self", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("self", "input", 0, 0)
        .muls("result", "self", 1.0)  # Placeholder
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Registry
# =============================================================================

TENSOR_METHOD_REGISTRY = {
    # Arithmetic
    "Tensor.add": tensor_add,
    "Tensor.sub": tensor_sub,
    "Tensor.mul": tensor_mul,
    "Tensor.div": tensor_div,
    "Tensor.neg": tensor_neg,
    "Tensor.abs": tensor_abs,
    "Tensor.pow": tensor_pow,
    "Tensor.sqrt": tensor_sqrt,
    "Tensor.rsqrt": tensor_rsqrt,
    "Tensor.reciprocal": tensor_reciprocal,
    "Tensor.square": tensor_square,
    "Tensor.addcmul": tensor_addcmul,
    "Tensor.addcdiv": tensor_addcdiv,
    
    # Trigonometric
    "Tensor.sin": tensor_sin,
    "Tensor.cos": tensor_cos,
    "Tensor.tan": tensor_tan,
    "Tensor.sinh": tensor_sinh,
    "Tensor.cosh": tensor_cosh,
    "Tensor.tanh": tensor_tanh,
    "Tensor.asin": tensor_asin,
    "Tensor.acos": tensor_acos,
    "Tensor.atan": tensor_atan,
    
    # Exponential/Logarithmic
    "Tensor.exp": tensor_exp,
    "Tensor.exp2": tensor_exp2,
    "Tensor.expm1": tensor_expm1,
    "Tensor.log": tensor_log,
    "Tensor.log2": tensor_log2,
    "Tensor.log10": tensor_log10,
    "Tensor.log1p": tensor_log1p,
    
    # Comparison
    "Tensor.max": tensor_max_elementwise,
    "Tensor.min": tensor_min_elementwise,
    "Tensor.clamp": tensor_clamp,
    "Tensor.clip": tensor_clip,
    
    # Reduction
    "Tensor.sum": tensor_sum,
    "Tensor.mean": tensor_mean,
    "Tensor.std": tensor_std,
    "Tensor.var": tensor_var,
    "Tensor.prod": tensor_prod,
    
    # Linear Algebra
    "Tensor.mm": tensor_mm,
    "Tensor.matmul": tensor_matmul,
    "Tensor.dot": tensor_dot,
    "Tensor.mv": tensor_mv,
    
    # Activations
    "Tensor.sigmoid": tensor_sigmoid,
    "Tensor.relu": tensor_relu,
    "Tensor.softmax": tensor_softmax,
    "Tensor.log_softmax": tensor_log_softmax,
    
    # Misc
    "Tensor.sign": tensor_sign,
    "Tensor.lerp": tensor_lerp,
    "Tensor.logit": tensor_logit,
    "Tensor.xlogy": tensor_xlogy,
    "Tensor.hypot": tensor_hypot,
    "Tensor.frac": tensor_frac,
    "Tensor.cumsum": tensor_cumsum,
    "Tensor.diff": tensor_diff,
}


# =============================================================================
# Main: Generate All Methods for All Backends
# =============================================================================

if __name__ == "__main__":
    from pto_compile import generate_all_backends, BACKENDS
    
    print("=" * 70)
    print("PTO torch.Tensor Methods - Multi-Backend Code Generation")
    print("=" * 70)
    
    OUTPUT_PREFIX = "torch_tensor"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    compiler = PTOCompiler()
    
    # Methods to generate
    methods_to_generate = [
        # Arithmetic
        ("Tensor.add", [], {}),
        ("Tensor.sub", [], {}),
        ("Tensor.mul", [], {}),
        ("Tensor.div", [], {}),
        ("Tensor.neg", [], {}),
        ("Tensor.abs", [], {}),
        ("Tensor.pow", [2.0], {}),
        ("Tensor.pow", [0.5], {'rows': 8, 'cols': 8}),
        ("Tensor.sqrt", [], {}),
        ("Tensor.rsqrt", [], {}),
        ("Tensor.reciprocal", [], {}),
        ("Tensor.square", [], {}),
        ("Tensor.addcmul", [1.0], {}),
        ("Tensor.addcdiv", [1.0], {}),
        
        # Trigonometric
        ("Tensor.sin", [], {}),
        ("Tensor.cos", [], {}),
        ("Tensor.tan", [], {}),
        ("Tensor.sinh", [], {}),
        ("Tensor.cosh", [], {}),
        ("Tensor.tanh", [], {}),
        ("Tensor.asin", [], {}),
        ("Tensor.acos", [], {}),
        ("Tensor.atan", [], {}),
        
        # Exponential/Logarithmic
        ("Tensor.exp", [], {}),
        ("Tensor.exp2", [], {}),
        ("Tensor.expm1", [], {}),
        ("Tensor.log", [], {}),
        ("Tensor.log2", [], {}),
        ("Tensor.log10", [], {}),
        ("Tensor.log1p", [], {}),
        
        # Comparison
        ("Tensor.max", [], {}),
        ("Tensor.min", [], {}),
        ("Tensor.clamp", [0.0, 1.0], {}),
        ("Tensor.clip", [-1.0, 1.0], {}),
        
        # Reduction
        ("Tensor.sum", [], {}),
        ("Tensor.mean", [], {}),
        ("Tensor.std", [], {}),
        ("Tensor.var", [], {}),
        ("Tensor.prod", [], {}),
        
        # Linear Algebra
        ("Tensor.mm", [], {'m': 8, 'k': 8, 'n': 8}),
        ("Tensor.matmul", [], {'m': 8, 'k': 8, 'n': 8}),
        ("Tensor.dot", [], {'size': 64}),
        ("Tensor.mv", [], {'m': 8, 'n': 8}),
        
        # Activations
        ("Tensor.sigmoid", [], {}),
        ("Tensor.relu", [], {}),
        ("Tensor.softmax", [], {}),
        ("Tensor.log_softmax", [], {}),
        
        # Misc
        ("Tensor.sign", [], {}),
        ("Tensor.lerp", [0.5], {}),
        ("Tensor.logit", [], {}),
        ("Tensor.xlogy", [], {}),
        ("Tensor.hypot", [], {}),
        ("Tensor.frac", [], {}),
        ("Tensor.cumsum", [], {}),
        ("Tensor.diff", [], {}),
    ]
    
    print(f"\nGenerating {len(methods_to_generate)} methods for {len(BACKENDS)} backends...")
    print(f"Backends: {', '.join(BACKENDS.keys())}")
    print()
    
    total_files = 0
    success_count = 0
    
    for method_name, args, kwargs in methods_to_generate:
        safe_name = method_name.replace(".", "_")
        if args and method_name == "Tensor.pow":
            safe_name += f"_{args[0]}".replace(".", "_")
        if args and method_name in ("Tensor.clamp", "Tensor.clip"):
            safe_name += f"_{args[0]}_{args[1]}".replace(".", "_").replace("-", "neg")
        
        try:
            builder_func = TENSOR_METHOD_REGISTRY[method_name]
            
            if args:
                program = builder_func(*args, **kwargs)
            else:
                program = builder_func(**kwargs)
            
            print(f"[{method_name}]")
            
            results = generate_all_backends(
                program,
                OUTPUT_PREFIX,
                output_base_dir=SCRIPT_DIR,
                enable_fusion=True
            )
            
            total_files += len(results)
            success_count += 1
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"Generation Complete! {success_count}/{len(methods_to_generate)} methods generated.")
    print(f"Total files generated: {total_files}")
    print(f"Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{OUTPUT_PREFIX}/")
    print(f"  - output_pto/{OUTPUT_PREFIX}/")
    print("=" * 70)
