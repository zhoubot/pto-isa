"""
PTO torch.nn Operators Implementation

This module implements PyTorch torch.nn operators using PTO ISA instructions.
Each operator is implemented as a function that builds a PTO program,
which can then be compiled to ARM64 NEON code with loop fusion.

Reference: https://docs.pytorch.org/docs/stable/nn.html

Categories implemented:
1. Non-linear Activations: ReLU, Sigmoid, Tanh, Softmax, GELU, SiLU, LeakyReLU, ELU, Hardswish
2. Normalization Layers: LayerNorm, RMSNorm, BatchNorm
3. Linear Layers: Linear (fully connected)
4. Dropout Layers: Dropout (inference mode)
5. Loss Functions: MSELoss, L1Loss, SmoothL1Loss
6. Pooling Layers: AvgPool (simplified 2D)
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
OUTPUT_DIR = "torch_nn_arm64"


# =============================================================================
# Non-linear Activations (weighted sum, nonlinearity)
# =============================================================================

def nn_ReLU(rows=8, cols=8):
    """
    nn.ReLU: Applies ReLU(x) = max(0, x) element-wise.
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    PTO Mapping: TRELU
    """
    return (PTOFunctionBuilder("nn_ReLU")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .relu("result", "x")
        .store("result", "output", 0, 0)
        
        .build())


def nn_ReLU6(rows=8, cols=8):
    """
    nn.ReLU6: Applies ReLU6(x) = min(max(0, x), 6) element-wise.
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU6.html
    PTO Mapping: TRELU, TEXPANDS, TMIN
    """
    return (PTOFunctionBuilder("nn_ReLU6")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("relu_out", rows, cols, DEFAULT_DTYPE)
        .tile("six", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .relu("relu_out", "x")           # max(0, x)
        .expands("six", 6.0)             # broadcast 6
        .min("result", "relu_out", "six") # min(relu_out, 6)
        .store("result", "output", 0, 0)
        
        .build())


def nn_LeakyReLU(negative_slope=0.01, rows=8, cols=8):
    """
    nn.LeakyReLU: LeakyReLU(x) = max(0,x) + negative_slope * min(0,x)
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    PTO Mapping: TRELU, TNEG, TRELU, TNEG, TMULS, TADD
    """
    return (PTOFunctionBuilder("nn_LeakyReLU")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("pos_part", rows, cols, DEFAULT_DTYPE)    # max(0, x)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)       # -x
        .tile("neg_relu", rows, cols, DEFAULT_DTYPE)    # max(0, -x)
        .tile("neg_part", rows, cols, DEFAULT_DTYPE)    # -max(0, -x) = min(0, x)
        .tile("scaled_neg", rows, cols, DEFAULT_DTYPE)  # negative_slope * min(0, x)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # max(0, x)
        .relu("pos_part", "x")
        
        # min(0, x) = -max(0, -x)
        .neg("neg_x", "x")
        .relu("neg_relu", "neg_x")
        .neg("neg_part", "neg_relu")
        
        # negative_slope * min(0, x)
        .muls("scaled_neg", "neg_part", negative_slope)
        
        # result = max(0,x) + negative_slope * min(0,x)
        .add("result", "pos_part", "scaled_neg")
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_ELU(alpha=1.0, rows=8, cols=8):
    """
    nn.ELU: ELU(x) = max(0,x) + min(0, alpha*(exp(x)-1))
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.ELU.html
    """
    return (PTOFunctionBuilder("nn_ELU")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("pos_part", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_minus_1", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_relu", rows, cols, DEFAULT_DTYPE)
        .tile("neg_part", rows, cols, DEFAULT_DTYPE)
        .tile("neg_contrib", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # max(0, x)
        .relu("pos_part", "x")
        
        # alpha * (exp(x) - 1)
        .exp("exp_x", "x")
        .adds("exp_minus_1", "exp_x", -1.0)
        .muls("scaled", "exp_minus_1", alpha)
        
        # min(0, scaled) = -max(0, -scaled)
        .neg("neg_x", "scaled")
        .relu("neg_relu", "neg_x")
        .neg("neg_contrib", "neg_relu")
        
        # result
        .add("result", "pos_part", "neg_contrib")
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_Sigmoid(rows=8, cols=8):
    """
    nn.Sigmoid: Applies sigmoid(x) = 1 / (1 + exp(-x)) element-wise.
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
    PTO Mapping: TNEG, TEXP, TADDS, TRECIP
    """
    return (PTOFunctionBuilder("nn_Sigmoid")
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


def nn_Tanh(rows=8, cols=8):
    """
    nn.Tanh: Applies tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) element-wise.
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html
    """
    return (PTOFunctionBuilder("nn_Tanh")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("numerator", rows, cols, DEFAULT_DTYPE)
        .tile("denominator", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        .exp("exp_x", "x")
        .neg("neg_x", "x")
        .exp("exp_neg_x", "neg_x")
        .sub("numerator", "exp_x", "exp_neg_x")
        .add("denominator", "exp_x", "exp_neg_x")
        .div("result", "numerator", "denominator")
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_Softmax(rows=8, cols=8):
    """
    nn.Softmax: Applies softmax(x_i) = exp(x_i) / sum(exp(x_j)) along dim=1 (rows).
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    Note: Simplified version without max subtraction for numerical stability.
    PTO Mapping: TEXP, TROWSUM, TDIV
    """
    return (PTOFunctionBuilder("nn_Softmax")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("sum_exp", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        .exp("exp_x", "x")
        .rowsum("sum_exp", "exp_x")
        .div("result", "exp_x", "sum_exp")  # Broadcasting division
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_LogSoftmax(rows=8, cols=8):
    """
    nn.LogSoftmax: Applies log(softmax(x)) = x - log(sum(exp(x))) along dim=1.
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
    """
    return (PTOFunctionBuilder("nn_LogSoftmax")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("sum_exp", rows, 1, DEFAULT_DTYPE)
        .tile("log_sum", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        .exp("exp_x", "x")
        .rowsum("sum_exp", "exp_x")
        .log("log_sum", "sum_exp")
        .rowexpandsub("result", "x", "log_sum")  # Row-wise broadcast subtraction
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_GELU(rows=8, cols=8):
    """
    nn.GELU: Gaussian Error Linear Unit.
    Approximation: x * sigmoid(1.702 * x)
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    """
    return (PTOFunctionBuilder("nn_GELU")
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
        .muls("scaled_x", "x", 1.702)
        .neg("neg_scaled", "scaled_x")
        .exp("exp_neg", "neg_scaled")
        .adds("one_plus", "exp_neg", 1.0)
        .recip("sigmoid_out", "one_plus")
        .mul("result", "x", "sigmoid_out")
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_SiLU(rows=8, cols=8):
    """
    nn.SiLU (Swish): Applies x * sigmoid(x) element-wise.
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html
    """
    return (PTOFunctionBuilder("nn_SiLU")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("neg_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("sigmoid_out", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        .neg("neg_x", "x")
        .exp("exp_neg", "neg_x")
        .adds("one_plus", "exp_neg", 1.0)
        .recip("sigmoid_out", "one_plus")
        .mul("result", "x", "sigmoid_out")
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_Mish(rows=8, cols=8):
    """
    nn.Mish: Applies x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Mish.html
    """
    return (PTOFunctionBuilder("nn_Mish")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus_exp", rows, cols, DEFAULT_DTYPE)
        .tile("softplus", rows, cols, DEFAULT_DTYPE)  # ln(1 + exp(x))
        .tile("exp_sp", rows, cols, DEFAULT_DTYPE)
        .tile("neg_sp", rows, cols, DEFAULT_DTYPE)
        .tile("exp_neg_sp", rows, cols, DEFAULT_DTYPE)
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
        
        # tanh(softplus)
        .exp("exp_sp", "softplus")
        .neg("neg_sp", "softplus")
        .exp("exp_neg_sp", "neg_sp")
        .sub("tanh_num", "exp_sp", "exp_neg_sp")
        .add("tanh_den", "exp_sp", "exp_neg_sp")
        .div("tanh_out", "tanh_num", "tanh_den")
        
        # x * tanh(softplus(x))
        .mul("result", "x", "tanh_out")
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_Hardswish(rows=8, cols=8):
    """
    nn.Hardswish: x * ReLU6(x + 3) / 6
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardswish.html
    """
    return (PTOFunctionBuilder("nn_Hardswish")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_plus_3", rows, cols, DEFAULT_DTYPE)
        .tile("relu_out", rows, cols, DEFAULT_DTYPE)
        .tile("six", rows, cols, DEFAULT_DTYPE)
        .tile("relu6_out", rows, cols, DEFAULT_DTYPE)
        .tile("scaled", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # x + 3
        .adds("x_plus_3", "x", 3.0)
        
        # ReLU6(x + 3) = min(max(0, x+3), 6)
        .relu("relu_out", "x_plus_3")
        .expands("six", 6.0)
        .min("relu6_out", "relu_out", "six")
        
        # x * ReLU6(x + 3) / 6
        .mul("scaled", "x", "relu6_out")
        .divs("result", "scaled", 6.0)
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_Hardsigmoid(rows=8, cols=8):
    """
    nn.Hardsigmoid: ReLU6(x + 3) / 6
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    """
    return (PTOFunctionBuilder("nn_Hardsigmoid")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_plus_3", rows, cols, DEFAULT_DTYPE)
        .tile("relu_out", rows, cols, DEFAULT_DTYPE)
        .tile("six", rows, cols, DEFAULT_DTYPE)
        .tile("relu6_out", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        .adds("x_plus_3", "x", 3.0)
        .relu("relu_out", "x_plus_3")
        .expands("six", 6.0)
        .min("relu6_out", "relu_out", "six")
        .divs("result", "relu6_out", 6.0)
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_Softplus(beta=1.0, rows=8, cols=8):
    """
    nn.Softplus: softplus(x) = (1/beta) * log(1 + exp(beta * x))
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    """
    return (PTOFunctionBuilder("nn_Softplus")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("scaled_x", rows, cols, DEFAULT_DTYPE)
        .tile("exp_x", rows, cols, DEFAULT_DTYPE)
        .tile("one_plus", rows, cols, DEFAULT_DTYPE)
        .tile("log_out", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        .muls("scaled_x", "x", beta)
        .exp("exp_x", "scaled_x")
        .adds("one_plus", "exp_x", 1.0)
        .log("log_out", "one_plus")
        .divs("result", "log_out", beta)
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Normalization Layers
# =============================================================================

def nn_LayerNorm(rows=8, cols=8, eps=1e-5):
    """
    nn.LayerNorm: Applies Layer Normalization.
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    Simplified: without learnable gamma/beta parameters.
    """
    return (PTOFunctionBuilder("nn_LayerNorm")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("mean", rows, 1, DEFAULT_DTYPE)
        .tile("x_minus_mean", rows, cols, DEFAULT_DTYPE)
        .tile("squared", rows, cols, DEFAULT_DTYPE)
        .tile("var_sum", rows, 1, DEFAULT_DTYPE)
        .tile("variance", rows, 1, DEFAULT_DTYPE)
        .tile("var_eps", rows, 1, DEFAULT_DTYPE)
        .tile("std", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # Compute mean
        .rowsum("row_sum", "x")
        .divs("mean", "row_sum", float(cols))
        
        # x - mean (row-wise broadcast)
        .rowexpandsub("x_minus_mean", "x", "mean")
        
        # Compute variance
        .mul("squared", "x_minus_mean", "x_minus_mean")
        .rowsum("var_sum", "squared")
        .divs("variance", "var_sum", float(cols))
        
        # sqrt(var + eps)
        .adds("var_eps", "variance", eps)
        .sqrt("std", "var_eps")
        
        # Normalize (row-wise broadcast divide)
        .rowexpanddiv("result", "x_minus_mean", "std")
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_RMSNorm(rows=8, cols=8, eps=1e-5):
    """
    nn.RMSNorm: Root Mean Square Layer Normalization.
    y = x / sqrt(mean(x^2) + eps)
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
    """
    return (PTOFunctionBuilder("nn_RMSNorm")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("x_squared", rows, cols, DEFAULT_DTYPE)
        .tile("mean_sq_sum", rows, 1, DEFAULT_DTYPE)
        .tile("mean_sq", rows, 1, DEFAULT_DTYPE)
        .tile("mean_sq_eps", rows, 1, DEFAULT_DTYPE)
        .tile("rms", rows, 1, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        
        # mean(x^2)
        .mul("x_squared", "x", "x")
        .rowsum("mean_sq_sum", "x_squared")
        .divs("mean_sq", "mean_sq_sum", float(cols))
        
        # sqrt(mean(x^2) + eps)
        .adds("mean_sq_eps", "mean_sq", eps)
        .sqrt("rms", "mean_sq_eps")
        
        # x / rms
        .div("result", "x", "rms")
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Linear Layers
# =============================================================================

def nn_Linear(in_features=8, out_features=8, batch_size=8):
    """
    nn.Linear: Applies linear transformation y = xW^T + b.
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Input: [batch_size, in_features]
    Weight: [out_features, in_features] -> transposed to [in_features, out_features]
    Output: [batch_size, out_features]
    
    Note: bias is pre-expanded to [batch_size, out_features] for compatibility.
    """
    return (PTOFunctionBuilder("nn_Linear")
        .tile("x", batch_size, in_features, DEFAULT_DTYPE)
        .tile("weight", in_features, out_features, DEFAULT_DTYPE)
        .tile("bias", batch_size, out_features, DEFAULT_DTYPE)  # Pre-expanded bias
        .tile("mm_result", batch_size, out_features, DEFAULT_DTYPE)
        .tile("result", batch_size, out_features, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("weight_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("bias_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .load("weight", "weight_mem", 0, 0)
        .load("bias", "bias_mem", 0, 0)
        
        .matmul("mm_result", "x", "weight")
        .add("result", "mm_result", "bias")
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_Bilinear(in1_features=8, in2_features=8, out_features=8, batch_size=8):
    """
    nn.Bilinear: Applies bilinear transformation.
    Simplified: y = x1 * x2 (element-wise) then linear
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Bilinear.html
    """
    return (PTOFunctionBuilder("nn_Bilinear")
        .tile("x1", batch_size, in1_features, DEFAULT_DTYPE)
        .tile("x2", batch_size, in2_features, DEFAULT_DTYPE)
        .tile("product", batch_size, in1_features, DEFAULT_DTYPE)
        .tile("weight", in1_features, out_features, DEFAULT_DTYPE)
        .tile("result", batch_size, out_features, DEFAULT_DTYPE)
        .memref("input1", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("input2", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("weight_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x1", "input1", 0, 0)
        .load("x2", "input2", 0, 0)
        .load("weight", "weight_mem", 0, 0)
        
        .mul("product", "x1", "x2")
        .matmul("result", "product", "weight")
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Loss Functions
# =============================================================================

def nn_MSELoss(rows=8, cols=8):
    """
    nn.MSELoss: Mean Squared Error loss = mean((pred - target)^2)
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    """
    return (PTOFunctionBuilder("nn_MSELoss")
        .tile("pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("squared", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("total_sum", 1, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("pred_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("pred", "pred_mem", 0, 0)
        .load("target", "target_mem", 0, 0)
        
        .sub("diff", "pred", "target")
        .mul("squared", "diff", "diff")
        .rowsum("row_sum", "squared")
        .colsum("total_sum", "row_sum")
        .divs("result", "total_sum", float(rows * cols))
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_L1Loss(rows=8, cols=8):
    """
    nn.L1Loss: Mean Absolute Error loss = mean(|pred - target|)
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
    """
    return (PTOFunctionBuilder("nn_L1Loss")
        .tile("pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("abs_diff", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("total_sum", 1, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("pred_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("pred", "pred_mem", 0, 0)
        .load("target", "target_mem", 0, 0)
        
        .sub("diff", "pred", "target")
        .abs("abs_diff", "diff")
        .rowsum("row_sum", "abs_diff")
        .colsum("total_sum", "row_sum")
        .divs("result", "total_sum", float(rows * cols))
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_SmoothL1Loss(beta=1.0, rows=8, cols=8):
    """
    nn.SmoothL1Loss: Huber loss.
    loss = 0.5 * x^2 / beta  if |x| < beta
           |x| - 0.5 * beta  otherwise
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    Simplified approximation using smooth components.
    """
    return (PTOFunctionBuilder("nn_SmoothL1Loss")
        .tile("pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("diff", rows, cols, DEFAULT_DTYPE)
        .tile("abs_diff", rows, cols, DEFAULT_DTYPE)
        .tile("squared", rows, cols, DEFAULT_DTYPE)
        .tile("l2_term", rows, cols, DEFAULT_DTYPE)
        .tile("l1_term", rows, cols, DEFAULT_DTYPE)
        .tile("smooth", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("total_sum", 1, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("pred_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("pred", "pred_mem", 0, 0)
        .load("target", "target_mem", 0, 0)
        
        .sub("diff", "pred", "target")
        .abs("abs_diff", "diff")
        
        # L2 component: 0.5 * x^2 / beta
        .mul("squared", "diff", "diff")
        .divs("l2_term", "squared", 2.0 * beta)
        
        # L1 component: |x| - 0.5 * beta
        .adds("l1_term", "abs_diff", -0.5 * beta)
        
        # Use L2 for small values (simplified approximation)
        .min("smooth", "l2_term", "l1_term")
        
        .rowsum("row_sum", "smooth")
        .colsum("total_sum", "row_sum")
        .divs("result", "total_sum", float(rows * cols))
        
        .store("result", "output", 0, 0)
        
        .build())


def nn_CrossEntropyLoss(rows=8, cols=8):
    """
    nn.CrossEntropyLoss: Cross entropy = -sum(target * log(softmax(pred)))
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    Simplified: assumes target is one-hot encoded.
    """
    return (PTOFunctionBuilder("nn_CrossEntropyLoss")
        .tile("pred", rows, cols, DEFAULT_DTYPE)
        .tile("target", rows, cols, DEFAULT_DTYPE)
        .tile("exp_pred", rows, cols, DEFAULT_DTYPE)
        .tile("sum_exp", rows, 1, DEFAULT_DTYPE)
        .tile("log_sum", rows, 1, DEFAULT_DTYPE)
        .tile("log_softmax", rows, cols, DEFAULT_DTYPE)
        .tile("weighted", rows, cols, DEFAULT_DTYPE)
        .tile("neg_weighted", rows, cols, DEFAULT_DTYPE)
        .tile("row_sum", rows, 1, DEFAULT_DTYPE)
        .tile("total_sum", 1, 1, DEFAULT_DTYPE)
        .tile("result", 1, 1, DEFAULT_DTYPE)
        .memref("pred_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("target_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("pred", "pred_mem", 0, 0)
        .load("target", "target_mem", 0, 0)
        
        # Log-softmax with row broadcast
        .exp("exp_pred", "pred")
        .rowsum("sum_exp", "exp_pred")
        .log("log_sum", "sum_exp")
        .rowexpandsub("log_softmax", "pred", "log_sum")
        
        # -target * log_softmax
        .mul("weighted", "target", "log_softmax")
        .neg("neg_weighted", "weighted")
        
        # Mean reduction
        .rowsum("row_sum", "neg_weighted")
        .colsum("total_sum", "row_sum")
        .divs("result", "total_sum", float(rows))
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Dropout (Inference Mode)
# =============================================================================

def nn_Dropout(p=0.0, rows=8, cols=8):
    """
    nn.Dropout: During inference, dropout is identity.
    During training, would scale by 1/(1-p).
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    This implements inference mode (identity).
    """
    return (PTOFunctionBuilder("nn_Dropout")
        .tile("x", rows, cols, DEFAULT_DTYPE)
        .tile("result", rows, cols, DEFAULT_DTYPE)
        .memref("input", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("x", "input", 0, 0)
        .muls("result", "x", 1.0)  # Identity operation
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Embedding (Simplified)
# =============================================================================

def nn_Embedding(num_embeddings=64, embedding_dim=8):
    """
    nn.Embedding: Simplified embedding lookup.
    
    Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    Note: This is a simplified version - actual embedding requires indexing.
    Here we show matrix multiply with one-hot encoded indices.
    """
    return (PTOFunctionBuilder("nn_Embedding")
        .tile("indices_onehot", 8, num_embeddings, DEFAULT_DTYPE)  # One-hot indices
        .tile("weight", num_embeddings, embedding_dim, DEFAULT_DTYPE)
        .tile("result", 8, embedding_dim, DEFAULT_DTYPE)
        .memref("indices_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("weight_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("indices_onehot", "indices_mem", 0, 0)
        .load("weight", "weight_mem", 0, 0)
        
        .matmul("result", "indices_onehot", "weight")
        
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Registry of All Operators
# =============================================================================

NN_OPERATOR_REGISTRY = {
    # Activations
    "nn.ReLU": nn_ReLU,
    "nn.ReLU6": nn_ReLU6,
    "nn.LeakyReLU": nn_LeakyReLU,
    "nn.ELU": nn_ELU,
    "nn.Sigmoid": nn_Sigmoid,
    "nn.Tanh": nn_Tanh,
    "nn.Softmax": nn_Softmax,
    "nn.LogSoftmax": nn_LogSoftmax,
    "nn.GELU": nn_GELU,
    "nn.SiLU": nn_SiLU,
    "nn.Mish": nn_Mish,
    "nn.Hardswish": nn_Hardswish,
    "nn.Hardsigmoid": nn_Hardsigmoid,
    "nn.Softplus": nn_Softplus,
    
    # Normalization
    "nn.LayerNorm": nn_LayerNorm,
    "nn.RMSNorm": nn_RMSNorm,
    
    # Linear
    "nn.Linear": nn_Linear,
    "nn.Bilinear": nn_Bilinear,
    
    # Loss
    "nn.MSELoss": nn_MSELoss,
    "nn.L1Loss": nn_L1Loss,
    "nn.SmoothL1Loss": nn_SmoothL1Loss,
    "nn.CrossEntropyLoss": nn_CrossEntropyLoss,
    
    # Dropout
    "nn.Dropout": nn_Dropout,
    
    # Embedding
    "nn.Embedding": nn_Embedding,
}



# =============================================================================
# Main: Generate All Operators for All Backends
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Add parent directory for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from pto_compile import generate_all_backends, BACKENDS
    
    print("=" * 70)
    print("PTO torch.nn Operators - Multi-Backend Code Generation")
    print("=" * 70)
    
    # Base output directory
    OUTPUT_PREFIX = "torch_nn"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    compiler = PTOCompiler()
    
    operators_to_generate = [
        # Activations
        ("nn.ReLU", [], {}),
        ("nn.ReLU6", [], {}),
        ("nn.LeakyReLU", [0.01], {}),
        ("nn.ELU", [1.0], {}),
        ("nn.Sigmoid", [], {}),
        ("nn.Tanh", [], {}),
        ("nn.Softmax", [], {}),
        ("nn.LogSoftmax", [], {}),
        ("nn.GELU", [], {}),
        ("nn.SiLU", [], {}),
        ("nn.Mish", [], {}),
        ("nn.Hardswish", [], {}),
        ("nn.Hardsigmoid", [], {}),
        ("nn.Softplus", [1.0], {}),
        
        # Normalization
        ("nn.LayerNorm", [], {'eps': 1e-5}),
        ("nn.RMSNorm", [], {'eps': 1e-5}),
        
        # Linear
        ("nn.Linear", [], {'in_features': 8, 'out_features': 8, 'batch_size': 8}),
        ("nn.Bilinear", [], {'in1_features': 8, 'in2_features': 8, 'out_features': 8, 'batch_size': 8}),
        
        # Loss
        ("nn.MSELoss", [], {}),
        ("nn.L1Loss", [], {}),
        ("nn.SmoothL1Loss", [1.0], {}),
        ("nn.CrossEntropyLoss", [], {}),
        
        # Others
        ("nn.Dropout", [0.0], {}),
        ("nn.Embedding", [], {'num_embeddings': 64, 'embedding_dim': 8}),
    ]
    
    print(f"\nGenerating {len(operators_to_generate)} operators for {len(BACKENDS)} backends...")
    print(f"Backends: {', '.join(BACKENDS.keys())}")
    print()
    
    total_files = 0
    success_count = 0
    
    for op_name, args, kwargs in operators_to_generate:
        safe_name = op_name.replace(".", "_")
        
        try:
            builder_func = NN_OPERATOR_REGISTRY[op_name]
            
            # Build program
            if args:
                program = builder_func(*args, **kwargs)
            else:
                program = builder_func(**kwargs)
            
            print(f"[{op_name}]")
            
            # Generate for all backends
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
    print(f"Generation Complete! {success_count}/{len(operators_to_generate)} operators generated.")
    print(f"Total files generated: {total_files}")
    print(f"Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{OUTPUT_PREFIX}/")
    print(f"  - output_pto/{OUTPUT_PREFIX}/")
    print("=" * 70)
