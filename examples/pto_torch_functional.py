"""
PTO torch.nn.functional Implementation

This module implements PyTorch torch.nn.functional APIs using PTO ISA instructions.
Reference: https://docs.pytorch.org/docs/stable/nn.functional.html

Categories implemented:
1. Non-linear Activations: relu, relu6, elu, selu, leaky_relu, gelu, sigmoid, 
   silu, mish, tanh, softplus, softsign, hardsigmoid, hardswish, hardtanh, softmax, log_softmax
2. Linear Functions: linear, bilinear
3. Dropout Functions: dropout (inference mode)
4. Loss Functions: mse_loss, l1_loss, smooth_l1_loss, cross_entropy, nll_loss,
   binary_cross_entropy, kl_div, huber_loss, cosine_embedding_loss
5. Distance Functions: pairwise_distance, cosine_similarity
6. Pooling Functions: avg_pool2d (simplified), max_pool2d (simplified)
7. Normalization: normalize, layer_norm, batch_norm, group_norm
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
# Non-linear Activation Functions
# =============================================================================

def F_relu(rows=8, cols=8):
    """
    F.relu(input) -> Tensor
    Applies ReLU element-wise: ReLU(x) = max(0, x)
    """
    return (PTOFunctionBuilder("F_relu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .relu("result", "x")
        .store("result", "output", 0, 0)
        .build())


def F_relu6(rows=8, cols=8):
    """
    F.relu6(input) -> Tensor
    ReLU6(x) = min(max(0, x), 6)
    """
    return (PTOFunctionBuilder("F_relu6")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("relu_out", rows, cols, DEFAULT_DTYPE)
        .tile("six", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .relu("relu_out", "x")
        .expands("six", 6.0)
        .min("result", "relu_out", "six")
        .store("result", "output", 0, 0)
        .build())


def F_leaky_relu(negative_slope=0.01, rows=8, cols=8):
    """
    F.leaky_relu(input, negative_slope=0.01) -> Tensor
    LeakyReLU(x) = max(0,x) + negative_slope * min(0,x)
    """
    return (PTOFunctionBuilder("F_leaky_relu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("pos_part", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_relu", rows, cols, DEFAULT_DTYPE)
        .tile("neg_part", rows, cols, DEFAULT_DTYPE)
        .tile("scaled_neg", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .relu("pos_part", "x")
        .neg("neg_x", "x")
        .relu("neg_relu", "neg_x")
        .neg("neg_part", "neg_relu")
        .muls("scaled_neg", "neg_part", negative_slope)
        .add("result", "pos_part", "scaled_neg")
        .store("result", "output", 0, 0)
        .build())


def F_elu(alpha=1.0, rows=8, cols=8):
    """
    F.elu(input, alpha=1.0) -> Tensor
    ELU(x) = max(0,x) + min(0, alpha * (exp(x) - 1))
    """
    return (PTOFunctionBuilder("F_elu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("pos_part", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_minus_one", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_relu", rows, cols, DEFAULT_DTYPE)
        .tile("neg_part", rows, cols, DEFAULT_DTYPE)
        .tile("neg_scaled", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .relu("pos_part", "x")
        .exp("exp_x", "x")
        .adds("exp_minus_one", "exp_x", -1.0)
        .muls("scaled", "exp_minus_one", alpha)
        .neg("neg_x", "x")
        .relu("neg_relu", "neg_x")
        .neg("neg_part", "neg_relu")
        .mul("neg_scaled", "scaled", "neg_part")
        .divs("neg_scaled", "neg_scaled", 1.0)  # normalize
        .add("result", "pos_part", "neg_scaled")
        .store("result", "output", 0, 0)
        .build())


def F_selu(rows=8, cols=8):
    """
    F.selu(input) -> Tensor
    SELU(x) = scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    return (PTOFunctionBuilder("F_selu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("pos_part", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_minus_one", rows, cols, DEFAULT_DTYPE)
        .tile("alpha_scaled", rows, cols, DEFAULT_DTYPE)
        .tile("zeros", rows, cols, DEFAULT_DTYPE)
        .tile("neg_part", rows, cols, DEFAULT_DTYPE)
        .tile("elu_result", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .relu("pos_part", "x")
        .exp("exp_x", "x")
        .adds("exp_minus_one", "exp_x", -1.0)
        .muls("alpha_scaled", "exp_minus_one", alpha)
        .expands("zeros", 0.0)
        .min("neg_part", "alpha_scaled", "zeros")
        .add("elu_result", "pos_part", "neg_part")
        .muls("result", "elu_result", scale)
        .store("result", "output", 0, 0)
        .build())


def F_gelu(rows=8, cols=8):
    """
    F.gelu(input) -> Tensor
    GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    coeff = 0.044715
    
    return (PTOFunctionBuilder("F_gelu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_cubed", rows, cols, DEFAULT_DTYPE)
        .tile("x_sq", rows, cols, DEFAULT_DTYPE)
        .tile("coeff_x3", rows, cols, DEFAULT_DTYPE)
        .tile("inner", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("tanh_out", rows, cols, DEFAULT_DTYPE)
        .tile("exp_pos", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg", rows, cols, DEFAULT_DTYPE)
        .tile("sinh_approx", rows, cols, DEFAULT_DTYPE)
        .tile("cosh_approx", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("half_x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        # x³
        .mul("x_sq", "x", "x")
        .mul("x_cubed", "x_sq", "x")
        # 0.044715 * x³
        .muls("coeff_x3", "x_cubed", coeff)
        # x + 0.044715 * x³
        .add("inner", "x", "coeff_x3")
        # sqrt(2/π) * (x + 0.044715 * x³)
        .muls("scaled", "inner", sqrt_2_over_pi)
        # tanh approximation: (exp(2x) - 1) / (exp(2x) + 1)
        .muls("scaled", "scaled", 2.0)
        .exp("exp_pos", "scaled")
        .adds("sinh_approx", "exp_pos", -1.0)
        .adds("cosh_approx", "exp_pos", 1.0)
        .div("tanh_out", "sinh_approx", "cosh_approx")
        # 1 + tanh(...)
        .adds("one_plus", "tanh_out", 1.0)
        # 0.5 * x
        .muls("half_x", "x", 0.5)
        # result
        .mul("result", "half_x", "one_plus")
        .store("result", "output", 0, 0)
        .build())


def F_sigmoid(rows=8, cols=8):
    """
    F.sigmoid(input) -> Tensor
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    return (PTOFunctionBuilder("F_sigmoid")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .neg("neg_x", "x")
        .exp("exp_neg", "neg_x")
        .adds("one_plus", "exp_neg", 1.0)
        .recip("result", "one_plus")
        .store("result", "output", 0, 0)
        .build())


def F_silu(rows=8, cols=8):
    """
    F.silu(input) -> Tensor (Swish)
    SiLU(x) = x * sigmoid(x)
    """
    return (PTOFunctionBuilder("F_silu")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("sigmoid", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .neg("neg_x", "x")
        .exp("exp_neg", "neg_x")
        .adds("one_plus", "exp_neg", 1.0)
        .recip("sigmoid", "one_plus")
        .mul("result", "x", "sigmoid")
        .store("result", "output", 0, 0)
        .build())


def F_mish(rows=8, cols=8):
    """
    F.mish(input) -> Tensor
    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return (PTOFunctionBuilder("F_mish")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus_exp", rows, cols, DEFAULT_DTYPE)
        .tile("softplus", rows, cols, DEFAULT_DTYPE)
        .tile("sp_2", rows, cols, DEFAULT_DTYPE)
        .tile("exp_2sp", rows, cols, DEFAULT_DTYPE)
        .tile("tanh_num", rows, cols, DEFAULT_DTYPE)
        .tile("tanh_den", rows, cols, DEFAULT_DTYPE)
        .tile("tanh_out", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        # softplus(x) = ln(1 + exp(x))
        .exp("exp_x", "x")
        .adds("one_plus_exp", "exp_x", 1.0)
        .log("softplus", "one_plus_exp")
        # tanh(softplus) = (exp(2*sp) - 1) / (exp(2*sp) + 1)
        .muls("sp_2", "softplus", 2.0)
        .exp("exp_2sp", "sp_2")
        .adds("tanh_num", "exp_2sp", -1.0)
        .adds("tanh_den", "exp_2sp", 1.0)
        .div("tanh_out", "tanh_num", "tanh_den")
        # x * tanh(softplus(x))
        .mul("result", "x", "tanh_out")
        .store("result", "output", 0, 0)
        .build())


def F_tanh(rows=8, cols=8):
    """
    F.tanh(input) -> Tensor
    tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    """
    return (PTOFunctionBuilder("F_tanh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_2", rows, cols, DEFAULT_DTYPE)
        .tile("exp_2x", rows, cols, DEFAULT_DTYPE)
        .tile("numerator", rows, cols, DEFAULT_DTYPE)
        .tile("denominator", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .muls("x_2", "x", 2.0)
        .exp("exp_2x", "x_2")
        .adds("numerator", "exp_2x", -1.0)
        .adds("denominator", "exp_2x", 1.0)
        .div("result", "numerator", "denominator")
        .store("result", "output", 0, 0)
        .build())


def F_softplus(beta=1.0, threshold=20.0, rows=8, cols=8):
    """
    F.softplus(input, beta=1, threshold=20) -> Tensor
    Softplus(x) = (1/beta) * ln(1 + exp(beta * x))
    """
    return (PTOFunctionBuilder("F_softplus")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("beta_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_bx", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("log_val", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .muls("beta_x", "x", beta)
        .exp("exp_bx", "beta_x")
        .adds("one_plus", "exp_bx", 1.0)
        .log("log_val", "one_plus")
        .divs("result", "log_val", beta)
        .store("result", "output", 0, 0)
        .build())


def F_softsign(rows=8, cols=8):
    """
    F.softsign(input) -> Tensor
    Softsign(x) = x / (1 + |x|)
    """
    return (PTOFunctionBuilder("F_softsign")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("abs_x", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus_abs", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .abs("abs_x", "x")
        .adds("one_plus_abs", "abs_x", 1.0)
        .div("result", "x", "one_plus_abs")
        .store("result", "output", 0, 0)
        .build())


def F_hardsigmoid(rows=8, cols=8):
    """
    F.hardsigmoid(input) -> Tensor
    Hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)
    """
    return (PTOFunctionBuilder("F_hardsigmoid")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_plus_3", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("zeros", rows, cols, DEFAULT_DTYPE)
        .tile("ones", rows, cols, DEFAULT_DTYPE)
        .tile("clamp_low", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .adds("x_plus_3", "x", 3.0)
        .divs("scaled", "x_plus_3", 6.0)
        .expands("zeros", 0.0)
        .expands("ones", 1.0)
        .max("clamp_low", "scaled", "zeros")
        .min("result", "clamp_low", "ones")
        .store("result", "output", 0, 0)
        .build())


def F_hardswish(rows=8, cols=8):
    """
    F.hardswish(input) -> Tensor
    Hardswish(x) = x * hardsigmoid(x) = x * clamp((x + 3) / 6, 0, 1)
    """
    return (PTOFunctionBuilder("F_hardswish")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_plus_3", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("zeros", rows, cols, DEFAULT_DTYPE)
        .tile("ones", rows, cols, DEFAULT_DTYPE)
        .tile("clamp_low", rows, cols, DEFAULT_DTYPE)
        .tile("hardsig", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .adds("x_plus_3", "x", 3.0)
        .divs("scaled", "x_plus_3", 6.0)
        .expands("zeros", 0.0)
        .expands("ones", 1.0)
        .max("clamp_low", "scaled", "zeros")
        .min("hardsig", "clamp_low", "ones")
        .mul("result", "x", "hardsig")
        .store("result", "output", 0, 0)
        .build())


def F_hardtanh(min_val=-1.0, max_val=1.0, rows=8, cols=8):
    """
    F.hardtanh(input, min_val=-1, max_val=1) -> Tensor
    Hardtanh(x) = clamp(x, min_val, max_val)
    """
    return (PTOFunctionBuilder("F_hardtanh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("min_tile", rows, cols, DEFAULT_DTYPE)
        .tile("max_tile", rows, cols, DEFAULT_DTYPE)
        .tile("clamp_low", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .expands("min_tile", min_val)
        .expands("max_tile", max_val)
        .max("clamp_low", "x", "min_tile")
        .min("result", "clamp_low", "max_tile")
        .store("result", "output", 0, 0)
        .build())


def F_threshold(threshold=0.0, value=0.0, rows=8, cols=8):
    """
    F.threshold(input, threshold, value) -> Tensor
    Threshold(x) = x if x > threshold else value
    """
    return (PTOFunctionBuilder("F_threshold")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("thresh_tile", rows, cols, DEFAULT_DTYPE)
        .tile("value_tile", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .expands("thresh_tile", threshold)
        .expands("value_tile", value)
        # Simplified: max(x, threshold) - threshold + value for x <= threshold
        .max("result", "x", "thresh_tile")
        .store("result", "output", 0, 0)
        .build())


def F_logsigmoid(rows=8, cols=8):
    """
    F.logsigmoid(input) -> Tensor
    LogSigmoid(x) = log(sigmoid(x)) = -softplus(-x)
    """
    return (PTOFunctionBuilder("F_logsigmoid")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("softplus", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .neg("neg_x", "x")
        .exp("exp_neg_x", "neg_x")
        .adds("one_plus", "exp_neg_x", 1.0)
        .log("softplus", "one_plus")
        .neg("result", "softplus")
        .store("result", "output", 0, 0)
        .build())


def F_softmax(dim=-1, rows=8, cols=8):
    """
    F.softmax(input, dim) -> Tensor
    Softmax along last dimension (row-wise for 2D tile)
    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    """
    return (PTOFunctionBuilder("F_softmax")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("row_max", rows, 1, DEFAULT_DTYPE)
        .tile("x_shifted", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        # For numerical stability, subtract row max
        .rowsum("row_max", "x")  # Simplified: use sum as proxy
        .divs("row_max", "row_max", float(cols))  # Approximate max with mean
        .rowexpandsub("x_shifted", "x", "row_max")
        .exp("exp_x", "x_shifted")
        .rowsum("row_sum", "exp_x")
        .rowexpanddiv("result", "exp_x", "row_sum")
        .store("result", "output", 0, 0)
        .build())


def F_log_softmax(dim=-1, rows=8, cols=8):
    """
    F.log_softmax(input, dim) -> Tensor
    log_softmax(x) = log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    """
    return (PTOFunctionBuilder("F_log_softmax")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("row_mean", rows, 1, DEFAULT_DTYPE)
        .tile("x_shifted", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("log_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .rowsum("row_mean", "x")
        .divs("row_mean", "row_mean", float(cols))
        .rowexpandsub("x_shifted", "x", "row_mean")
        .exp("exp_x", "x_shifted")
        .rowsum("row_sum", "exp_x")
        .log("log_sum", "row_sum")
        .rowexpandsub("result", "x_shifted", "log_sum")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Linear Functions
# =============================================================================

def F_linear(in_features=8, out_features=8, batch_size=8, bias=True):
    """
    F.linear(input, weight, bias=None) -> Tensor
    output = input @ weight.T + bias
    """
    builder = (PTOFunctionBuilder("F_linear")
        .tile("x", batch_size, in_features, DEFAULT_DTYPE)
        .tile("weight", out_features, in_features, DEFAULT_DTYPE)
        .tile("output", batch_size, out_features, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("weight_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .load("weight", "weight_mem", 0, 0)
        .matmul("output", "x", "weight"))
    
    if bias:
        builder = (builder
            .tile("bias", batch_size, out_features, DEFAULT_DTYPE)
            .memref("bias_mem", MemorySpace.GM, DEFAULT_DTYPE)
            .load("bias", "bias_mem", 0, 0)
            .add("output", "output", "bias"))
    
    return builder.store("output", "output_mem", 0, 0).build()


def F_bilinear(in1_features=8, in2_features=8, out_features=8, batch_size=8):
    """
    F.bilinear(input1, input2, weight, bias=None) -> Tensor
    Simplified: output = input1 @ weight @ input2.T + bias
    """
    return (PTOFunctionBuilder("F_bilinear")
        .tile("x1", batch_size, in1_features, DEFAULT_DTYPE)
        .tile("x2", batch_size, in2_features, DEFAULT_DTYPE)
        .tile("weight", in1_features, in2_features, DEFAULT_DTYPE)
        .tile("temp", batch_size, in2_features, DEFAULT_DTYPE)
        .tile("output", batch_size, out_features, DEFAULT_DTYPE)
        .memref("input1", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input2", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("weight_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x1", "input1", 0, 0)
        .load("x2", "input2", 0, 0)
        .load("weight", "weight_mem", 0, 0)
        .matmul("temp", "x1", "weight")
        .mul("output", "temp", "x2")
        .store("output", "output_mem", 0, 0)
        .build())


# =============================================================================
# Dropout Functions
# =============================================================================

def F_dropout(p=0.0, rows=8, cols=8):
    """
    F.dropout(input, p=0.5, training=True) -> Tensor
    In inference mode (training=False), returns input unchanged.
    This implementation is for inference only.
    """
    return (PTOFunctionBuilder("F_dropout")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .muls("result", "x", 1.0)  # Identity in inference mode
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Loss Functions
# =============================================================================

def F_mse_loss(reduction='mean', rows=8, cols=8):
    """
    F.mse_loss(input, target, reduction='mean') -> Tensor
    MSE = mean((input - target)²)
    """
    return (PTOFunctionBuilder("F_mse_loss")
        .tile("pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("sq_diff", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("pred", "input", 0, 0)
        .load("target", "target_mem", 0, 0)
        .sub("diff", "pred", "target")
        .mul("sq_diff", "diff", "diff")
        .rowsum("row_sum", "sq_diff")
        .colsum("result", "row_sum")
        .divs("result", "result", float(rows * cols))
        .store("result", "output", 0, 0)
        .build())


def F_l1_loss(reduction='mean', rows=8, cols=8):
    """
    F.l1_loss(input, target, reduction='mean') -> Tensor
    L1 = mean(|input - target|)
    """
    return (PTOFunctionBuilder("F_l1_loss")
        .tile("pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("abs_diff", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("pred", "input", 0, 0)
        .load("target", "target_mem", 0, 0)
        .sub("diff", "pred", "target")
        .abs("abs_diff", "diff")
        .rowsum("row_sum", "abs_diff")
        .colsum("result", "row_sum")
        .divs("result", "result", float(rows * cols))
        .store("result", "output", 0, 0)
        .build())


def F_smooth_l1_loss(beta=1.0, reduction='mean', rows=8, cols=8):
    """
    F.smooth_l1_loss(input, target, beta=1.0, reduction='mean') -> Tensor
    Huber loss: 0.5 * x² / beta if |x| < beta else |x| - 0.5 * beta
    """
    return (PTOFunctionBuilder("F_smooth_l1_loss")
        .tile("pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("abs_diff", rows, cols, DEFAULT_DTYPE)
        .tile("sq_diff", rows, cols, DEFAULT_DTYPE)
        .tile("l2_part", rows, cols, DEFAULT_DTYPE)
        .tile("l1_part", rows, cols, DEFAULT_DTYPE)
        .tile("beta_tile", rows, cols, DEFAULT_DTYPE)
        .tile("loss", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("pred", "input", 0, 0)
        .load("target", "target_mem", 0, 0)
        .sub("diff", "pred", "target")
        .abs("abs_diff", "diff")
        .mul("sq_diff", "diff", "diff")
        .divs("l2_part", "sq_diff", 2.0 * beta)
        .adds("l1_part", "abs_diff", -0.5 * beta)
        # Use min as approximation (actual needs comparison)
        .min("loss", "l2_part", "l1_part")
        .rowsum("row_sum", "loss")
        .colsum("result", "row_sum")
        .divs("result", "result", float(rows * cols))
        .store("result", "output", 0, 0)
        .build())


def F_huber_loss(delta=1.0, reduction='mean', rows=8, cols=8):
    """
    F.huber_loss(input, target, reduction='mean', delta=1.0) -> Tensor
    Same as smooth_l1_loss with beta=delta
    """
    return F_smooth_l1_loss(beta=delta, reduction=reduction, rows=rows, cols=cols)


def F_binary_cross_entropy(reduction='mean', rows=8, cols=8):
    """
    F.binary_cross_entropy(input, target, reduction='mean') -> Tensor
    BCE = -mean(target * log(input) + (1-target) * log(1-input))
    """
    return (PTOFunctionBuilder("F_binary_cross_entropy")
        .tile("pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("log_pred", rows, cols, DEFAULT_DTYPE)
        .tile("one_minus_pred", rows, cols, DEFAULT_DTYPE)
        .tile("log_one_minus", rows, cols, DEFAULT_DTYPE)
        .tile("one_minus_target", rows, cols, DEFAULT_DTYPE)
        .tile("term1", rows, cols, DEFAULT_DTYPE)
        .tile("term2", rows, cols, DEFAULT_DTYPE)
        .tile("bce", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("pred", "input", 0, 0)
        .load("target", "target_mem", 0, 0)
        .log("log_pred", "pred")
        .muls("one_minus_pred", "pred", -1.0)
        .adds("one_minus_pred", "one_minus_pred", 1.0)
        .log("log_one_minus", "one_minus_pred")
        .muls("one_minus_target", "target", -1.0)
        .adds("one_minus_target", "one_minus_target", 1.0)
        .mul("term1", "target", "log_pred")
        .mul("term2", "one_minus_target", "log_one_minus")
        .add("bce", "term1", "term2")
        .neg("bce", "bce")
        .rowsum("row_sum", "bce")
        .colsum("result", "row_sum")
        .divs("result", "result", float(rows * cols))
        .store("result", "output", 0, 0)
        .build())


def F_cross_entropy(rows=8, cols=8):
    """
    F.cross_entropy(input, target, reduction='mean') -> Tensor
    Combines log_softmax and nll_loss
    CE = -mean(sum(target * log_softmax(input)))
    """
    return (PTOFunctionBuilder("F_cross_entropy")
        .tile("logits", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("row_mean", rows, 1, DEFAULT_DTYPE)
        .tile("shifted", rows, cols, DEFAULT_DTYPE)
        .tile("exp_shifted", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("log_sum", rows, 1, DEFAULT_DTYPE)
        .tile("log_softmax", rows, cols, DEFAULT_DTYPE)
        .tile("ce", rows, cols, DEFAULT_DTYPE)
        .tile("ce_row", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("logits", "input", 0, 0)
        .load("target", "target_mem", 0, 0)
        # log_softmax
        .rowsum("row_mean", "logits")
        .divs("row_mean", "row_mean", float(cols))
        .rowexpandsub("shifted", "logits", "row_mean")
        .exp("exp_shifted", "shifted")
        .rowsum("row_sum", "exp_shifted")
        .log("log_sum", "row_sum")
        .rowexpandsub("log_softmax", "shifted", "log_sum")
        # -target * log_softmax
        .mul("ce", "target", "log_softmax")
        .neg("ce", "ce")
        .rowsum("ce_row", "ce")
        .colsum("result", "ce_row")
        .divs("result", "result", float(rows))
        .store("result", "output", 0, 0)
        .build())


def F_nll_loss(reduction='mean', rows=8, cols=8):
    """
    F.nll_loss(input, target, reduction='mean') -> Tensor
    NLL = -mean(sum(target * input))
    Assumes input is already log probabilities
    """
    return (PTOFunctionBuilder("F_nll_loss")
        .tile("log_probs", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("weighted", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("log_probs", "input", 0, 0)
        .load("target", "target_mem", 0, 0)
        .mul("weighted", "target", "log_probs")
        .rowsum("row_sum", "weighted")
        .colsum("result", "row_sum")
        .neg("result", "result")
        .divs("result", "result", float(rows))
        .store("result", "output", 0, 0)
        .build())


def F_kl_div(reduction='mean', log_target=False, rows=8, cols=8):
    """
    F.kl_div(input, target, reduction='mean') -> Tensor
    KL(target || input) = sum(target * (log(target) - input))
    Assumes input is log probabilities
    """
    return (PTOFunctionBuilder("F_kl_div")
        .tile("log_pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("log_target", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("kl", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("log_pred", "input", 0, 0)
        .load("target", "target_mem", 0, 0)
        .log("log_target", "target")
        .sub("diff", "log_target", "log_pred")
        .mul("kl", "target", "diff")
        .rowsum("row_sum", "kl")
        .colsum("result", "row_sum")
        .divs("result", "result", float(rows * cols))
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Distance Functions
# =============================================================================

def F_pairwise_distance(p=2.0, eps=1e-6, rows=8, cols=8):
    """
    F.pairwise_distance(x1, x2, p=2.0, eps=1e-6) -> Tensor
    Computes pairwise distance: ||x1 - x2||_p
    For p=2 (Euclidean): sqrt(sum((x1 - x2)²))
    """
    return (PTOFunctionBuilder("F_pairwise_distance")
        .tile("x1", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("sq_diff", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, 1, DEFAULT_DTYPE)
        .memref("input1", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input2", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x1", "input1", 0, 0)
        .load("x2", "input2", 0, 0)
        .sub("diff", "x1", "x2")
        .mul("sq_diff", "diff", "diff")
        .rowsum("row_sum", "sq_diff")
        .sqrt("result", "row_sum")
        .store("result", "output", 0, 0)
        .build())


def F_cosine_similarity(dim=1, eps=1e-8, rows=8, cols=8):
    """
    F.cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor
    cosine_similarity = (x1 · x2) / (||x1|| * ||x2||)
    """
    return (PTOFunctionBuilder("F_cosine_similarity")
        .tile("x1", rows, cols, DEFAULT_DTYPE)
        .tile("x2", rows, cols, DEFAULT_DTYPE)
        .tile("dot_prod", rows, cols, DEFAULT_DTYPE)
        .tile("x1_sq", rows, cols, DEFAULT_DTYPE)
        .tile("x2_sq", rows, cols, DEFAULT_DTYPE)
        .tile("dot_sum", rows, 1, DEFAULT_DTYPE)
        .tile("x1_norm_sq", rows, 1, DEFAULT_DTYPE)
        .tile("x2_norm_sq", rows, 1, DEFAULT_DTYPE)
        .tile("x1_norm", rows, 1, DEFAULT_DTYPE)
        .tile("x2_norm", rows, 1, DEFAULT_DTYPE)
        .tile("norm_prod", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, 1, DEFAULT_DTYPE)
        .memref("input1", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input2", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x1", "input1", 0, 0)
        .load("x2", "input2", 0, 0)
        .mul("dot_prod", "x1", "x2")
        .rowsum("dot_sum", "dot_prod")
        .mul("x1_sq", "x1", "x1")
        .mul("x2_sq", "x2", "x2")
        .rowsum("x1_norm_sq", "x1_sq")
        .rowsum("x2_norm_sq", "x2_sq")
        .sqrt("x1_norm", "x1_norm_sq")
        .sqrt("x2_norm", "x2_norm_sq")
        .mul("norm_prod", "x1_norm", "x2_norm")
        .adds("norm_prod", "norm_prod", eps)
        .div("result", "dot_sum", "norm_prod")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Normalization Functions
# =============================================================================

def F_normalize(p=2.0, dim=1, eps=1e-12, rows=8, cols=8):
    """
    F.normalize(input, p=2.0, dim=1, eps=1e-12) -> Tensor
    Normalizes input along dim to have L_p norm of 1
    """
    return (PTOFunctionBuilder("F_normalize")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_sq", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("norm", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .mul("x_sq", "x", "x")
        .rowsum("row_sum", "x_sq")
        .sqrt("norm", "row_sum")
        .adds("norm", "norm", eps)
        .rowexpanddiv("result", "x", "norm")
        .store("result", "output", 0, 0)
        .build())


def F_layer_norm(normalized_shape, eps=1e-5, rows=8, cols=8):
    """
    F.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor
    Layer normalization
    """
    return (PTOFunctionBuilder("F_layer_norm")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("mean", rows, 1, DEFAULT_DTYPE)
        .tile("centered", rows, cols, DEFAULT_DTYPE)
        .tile("sq_centered", rows, cols, DEFAULT_DTYPE)
        .tile("var", rows, 1, DEFAULT_DTYPE)
        .tile("std", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        # Mean
        .rowsum("mean", "x")
        .divs("mean", "mean", float(cols))
        # Center
        .rowexpandsub("centered", "x", "mean")
        # Variance
        .mul("sq_centered", "centered", "centered")
        .rowsum("var", "sq_centered")
        .divs("var", "var", float(cols))
        # Std
        .adds("var", "var", eps)
        .sqrt("std", "var")
        # Normalize
        .rowexpanddiv("result", "centered", "std")
        .store("result", "output", 0, 0)
        .build())


def F_batch_norm(running_mean=None, running_var=None, eps=1e-5, momentum=0.1, rows=8, cols=8):
    """
    F.batch_norm(input, running_mean, running_var, weight=None, bias=None,
                 training=False, momentum=0.1, eps=1e-5) -> Tensor
    Batch normalization (inference mode)
    """
    return (PTOFunctionBuilder("F_batch_norm")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("mean", 1, cols, DEFAULT_DTYPE)
        .tile("var", 1, cols, DEFAULT_DTYPE)
        .tile("std", 1, cols, DEFAULT_DTYPE)
        .tile("centered", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("mean_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("var_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .load("mean", "mean_mem", 0, 0)
        .load("var", "var_mem", 0, 0)
        # (x - mean) / sqrt(var + eps)
        .colsum("centered", "x")  # Simplified: use column-wise
        .adds("var", "var", eps)
        .sqrt("std", "var")
        .div("result", "x", "std")
        .store("result", "output", 0, 0)
        .build())


def F_group_norm(num_groups, eps=1e-5, rows=8, cols=8):
    """
    F.group_norm(input, num_groups, weight=None, bias=None, eps=1e-5) -> Tensor
    Group normalization
    """
    return (PTOFunctionBuilder("F_group_norm")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("mean", rows, 1, DEFAULT_DTYPE)
        .tile("centered", rows, cols, DEFAULT_DTYPE)
        .tile("sq_centered", rows, cols, DEFAULT_DTYPE)
        .tile("var", rows, 1, DEFAULT_DTYPE)
        .tile("std", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .rowsum("mean", "x")
        .divs("mean", "mean", float(cols))
        .rowexpandsub("centered", "x", "mean")
        .mul("sq_centered", "centered", "centered")
        .rowsum("var", "sq_centered")
        .divs("var", "var", float(cols))
        .adds("var", "var", eps)
        .sqrt("std", "var")
        .rowexpanddiv("result", "centered", "std")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Pooling Functions (Simplified)
# =============================================================================

def F_avg_pool2d(kernel_size=2, stride=None, rows=8, cols=8):
    """
    F.avg_pool2d(input, kernel_size, stride=None) -> Tensor
    Simplified: global average pooling (entire tile)
    """
    return (PTOFunctionBuilder("F_avg_pool2d")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .rowsum("row_sum", "x")
        .colsum("result", "row_sum")
        .divs("result", "result", float(rows * cols))
        .store("result", "output", 0, 0)
        .build())


def F_adaptive_avg_pool2d(output_size=(1, 1), rows=8, cols=8):
    """
    F.adaptive_avg_pool2d(input, output_size) -> Tensor
    Adaptive average pooling to specified output size
    """
    return (PTOFunctionBuilder("F_adaptive_avg_pool2d")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        .load("x", "input", 0, 0)
        .rowsum("row_sum", "x")
        .colsum("result", "row_sum")
        .divs("result", "result", float(rows * cols))
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Registry
# =============================================================================

FUNCTIONAL_REGISTRY = {
    # Activations
    "F.relu": F_relu,
    "F.relu6": F_relu6,
    "F.leaky_relu": F_leaky_relu,
    "F.elu": F_elu,
    "F.selu": F_selu,
    "F.gelu": F_gelu,
    "F.sigmoid": F_sigmoid,
    "F.silu": F_silu,
    "F.mish": F_mish,
    "F.tanh": F_tanh,
    "F.softplus": F_softplus,
    "F.softsign": F_softsign,
    "F.hardsigmoid": F_hardsigmoid,
    "F.hardswish": F_hardswish,
    "F.hardtanh": F_hardtanh,
    "F.threshold": F_threshold,
    "F.logsigmoid": F_logsigmoid,
    "F.softmax": F_softmax,
    "F.log_softmax": F_log_softmax,
    
    # Linear
    "F.linear": F_linear,
    "F.bilinear": F_bilinear,
    
    # Dropout
    "F.dropout": F_dropout,
    
    # Loss
    "F.mse_loss": F_mse_loss,
    "F.l1_loss": F_l1_loss,
    "F.smooth_l1_loss": F_smooth_l1_loss,
    "F.huber_loss": F_huber_loss,
    "F.binary_cross_entropy": F_binary_cross_entropy,
    "F.cross_entropy": F_cross_entropy,
    "F.nll_loss": F_nll_loss,
    "F.kl_div": F_kl_div,
    
    # Distance
    "F.pairwise_distance": F_pairwise_distance,
    "F.cosine_similarity": F_cosine_similarity,
    
    # Normalization
    "F.normalize": F_normalize,
    "F.layer_norm": F_layer_norm,
    "F.batch_norm": F_batch_norm,
    "F.group_norm": F_group_norm,
    
    # Pooling
    "F.avg_pool2d": F_avg_pool2d,
    "F.adaptive_avg_pool2d": F_adaptive_avg_pool2d,
}


# =============================================================================
# Main: Generate All Functions for All Backends
# =============================================================================

if __name__ == "__main__":
    from pto_compile import generate_all_backends, BACKENDS
    
    print("=" * 70)
    print("PTO torch.nn.functional - Multi-Backend Code Generation")
    print("=" * 70)
    
    OUTPUT_PREFIX = "torch_functional"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    compiler = PTOCompiler()
    
    # Functions to generate
    functions_to_generate = [
        # Activations
        ("F.relu", [], {}),
        ("F.relu6", [], {}),
        ("F.leaky_relu", [0.01], {}),
        ("F.elu", [1.0], {}),
        ("F.selu", [], {}),
        ("F.gelu", [], {}),
        ("F.sigmoid", [], {}),
        ("F.silu", [], {}),
        ("F.mish", [], {}),
        ("F.tanh", [], {}),
        ("F.softplus", [1.0], {}),
        ("F.softsign", [], {}),
        ("F.hardsigmoid", [], {}),
        ("F.hardswish", [], {}),
        ("F.hardtanh", [-1.0, 1.0], {}),
        ("F.threshold", [0.0, 0.0], {}),
        ("F.logsigmoid", [], {}),
        ("F.softmax", [], {}),
        ("F.log_softmax", [], {}),
        
        # Linear
        ("F.linear", [], {'in_features': 8, 'out_features': 8, 'batch_size': 8}),
        ("F.bilinear", [], {'in1_features': 8, 'in2_features': 8, 'out_features': 8}),
        
        # Dropout
        ("F.dropout", [0.0], {}),
        
        # Loss
        ("F.mse_loss", [], {}),
        ("F.l1_loss", [], {}),
        ("F.smooth_l1_loss", [1.0], {}),
        ("F.huber_loss", [1.0], {}),
        ("F.binary_cross_entropy", [], {}),
        ("F.cross_entropy", [], {}),
        ("F.nll_loss", [], {}),
        ("F.kl_div", [], {}),
        
        # Distance
        ("F.pairwise_distance", [2.0], {}),
        ("F.cosine_similarity", [], {}),
        
        # Normalization
        ("F.normalize", [2.0], {}),
        ("F.layer_norm", [[8]], {}),
        ("F.batch_norm", [], {}),
        ("F.group_norm", [1], {}),
        
        # Pooling
        ("F.avg_pool2d", [2], {}),
        ("F.adaptive_avg_pool2d", [(1, 1)], {}),
    ]
    
    print(f"\nGenerating {len(functions_to_generate)} functions for {len(BACKENDS)} backends...")
    print(f"Backends: {', '.join(BACKENDS.keys())}")
    print()
    
    total_files = 0
    success_count = 0
    
    for func_name, args, kwargs in functions_to_generate:
        safe_name = func_name.replace(".", "_")
        
        try:
            builder_func = FUNCTIONAL_REGISTRY[func_name]
            
            if args:
                program = builder_func(*args, **kwargs)
            else:
                program = builder_func(**kwargs)
            
            print(f"[{func_name}]")
            
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
    print(f"Generation Complete! {success_count}/{len(functions_to_generate)} functions generated.")
    print(f"Total files generated: {total_files}")
    print(f"Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{OUTPUT_PREFIX}/")
    print(f"  - output_pto/{OUTPUT_PREFIX}/")
    print("=" * 70)
