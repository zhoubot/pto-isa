"""
Auto-generated Python code from PTO Assembly
Module: llama7b_flash
Entry: llama_layer_dynamic

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


def create_tile_add(module=None):
    """
    Create the tile_add function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_add", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_a", MemorySpace.GM, ElementType.F32)
        .memref("input_b", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("a", 32, 128, ElementType.F32)
        .tile("b", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .add("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())

def create_tile_mul(module=None):
    """
    Create the tile_mul function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_mul", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_a", MemorySpace.GM, ElementType.F32)
        .memref("input_b", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("a", 32, 128, ElementType.F32)
        .tile("b", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .mul("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())

def create_tile_muls(module=None):
    """
    Create the tile_muls function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_muls", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("a", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Scalar declarations
        .scalar("scale", ElementType.F32)
        
        # Instructions
        .load("a", "input", 0, 0)
        .muls("result", "a", "scale")
        .store("result", "output", 0, 0)
        .build())

def create_tile_exp(module=None):
    """
    Create the tile_exp function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_exp", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .exp("result", "x")
        .store("result", "output", 0, 0)
        .build())

def create_tile_silu(module=None):
    """
    Create the tile_silu function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_silu", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("neg_x", 32, 128, ElementType.F32)
        .tile("exp_neg_x", 32, 128, ElementType.F32)
        .tile("one_plus_exp", 32, 128, ElementType.F32)
        .tile("sigmoid", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .neg("neg_x", "x")
        .exp("exp_neg_x", "neg_x")
        .adds("one_plus_exp", "exp_neg_x", 1.0)
        .recip("sigmoid", "one_plus_exp")
        .mul("result", "x", "sigmoid")
        .store("result", "output", 0, 0)
        .build())

def create_tile_rsqrt(module=None):
    """
    Create the tile_rsqrt function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_rsqrt", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .rsqrt("result", "x")
        .store("result", "output", 0, 0)
        .build())

def create_tile_matmul(module=None):
    """
    Create the tile_matmul function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_matmul", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_a", MemorySpace.GM, ElementType.F32)
        .memref("input_b", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("a", 32, 128, ElementType.F32)
        .tile("b", 128, 128, ElementType.F32)
        .tile("c", 32, 128, ElementType.F32)
        
        # Instructions
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .matmul("c", "a", "b")
        .store("c", "output", 0, 0)
        .build())

def create_tile_rowmax(module=None):
    """
    Create the tile_rowmax function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_rowmax", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("result", 32, 1, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .rowmax("result", "x")
        .store("result", "output", 0, 0)
        .build())

def create_tile_rowsum(module=None):
    """
    Create the tile_rowsum function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_rowsum", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("result", 32, 1, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .rowsum("result", "x")
        .store("result", "output", 0, 0)
        .build())

def create_tile_rowexpandsub(module=None):
    """
    Create the tile_rowexpandsub function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_rowexpandsub", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_x", MemorySpace.GM, ElementType.F32)
        .memref("input_row", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("row_vals", 32, 1, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandsub("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())

def create_tile_rowexpanddiv(module=None):
    """
    Create the tile_rowexpanddiv function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_rowexpanddiv", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_x", MemorySpace.GM, ElementType.F32)
        .memref("input_row", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("row_vals", 32, 1, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpanddiv("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())

def create_tile_rowexpandmul(module=None):
    """
    Create the tile_rowexpandmul function.
    Type: InCore
    """
    return (PTOFunctionBuilder("tile_rowexpandmul", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_x", MemorySpace.GM, ElementType.F32)
        .memref("input_row", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("row_vals", 32, 1, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandmul("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())

def create_rmsnorm_tile(module=None):
    """
    Create the rmsnorm_tile function.
    Type: InCore
    """
    return (PTOFunctionBuilder("rmsnorm_tile", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("weights", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("x_sq", 32, 128, ElementType.F32)
        .tile("row_sum", 32, 1, ElementType.F32)
        .tile("row_mean", 32, 1, ElementType.F32)
        .tile("row_rsqrt", 32, 1, ElementType.F32)
        .tile("x_norm", 32, 128, ElementType.F32)
        .tile("gamma", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Scalar declarations
        .scalar("eps", ElementType.F32)
        .scalar("inv_cols", ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .load("gamma", "weights", 0, 0)
        .mul("x_sq", "x", "x")
        .rowsum("row_sum", "x_sq")
        .scalar_li("inv_cols", 0.0078125)
        .muls("row_mean", "row_sum", "inv_cols")
        .scalar_li("eps", 1e-05)
        .adds("row_mean", "row_mean", "eps")
        .rsqrt("row_rsqrt", "row_mean")
        .rowexpandmul("x_norm", "x", "row_rsqrt")
        .mul("result", "x_norm", "gamma")
        .store("result", "output", 0, 0)
        .build())

def create_softmax_tile(module=None):
    """
    Create the softmax_tile function.
    Type: InCore
    """
    return (PTOFunctionBuilder("softmax_tile", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("row_max", 32, 1, ElementType.F32)
        .tile("x_shifted", 32, 128, ElementType.F32)
        .tile("exp_x", 32, 128, ElementType.F32)
        .tile("row_sum", 32, 1, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .rowmax("row_max", "x")
        .rowexpandsub("x_shifted", "x", "row_max")
        .exp("exp_x", "x_shifted")
        .rowsum("row_sum", "exp_x")
        .rowexpanddiv("result", "exp_x", "row_sum")
        .store("result", "output", 0, 0)
        .build())

def create_swiglu_tile(module=None):
    """
    Create the swiglu_tile function.
    Type: InCore
    """
    return (PTOFunctionBuilder("swiglu_tile", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_gate", MemorySpace.GM, ElementType.F32)
        .memref("input_up", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("gate", 32, 128, ElementType.F32)
        .tile("up", 32, 128, ElementType.F32)
        .tile("neg_gate", 32, 128, ElementType.F32)
        .tile("exp_neg_gate", 32, 128, ElementType.F32)
        .tile("one_plus_exp", 32, 128, ElementType.F32)
        .tile("sigmoid_gate", 32, 128, ElementType.F32)
        .tile("gate_silu", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("gate", "input_gate", 0, 0)
        .load("up", "input_up", 0, 0)
        .neg("neg_gate", "gate")
        .exp("exp_neg_gate", "neg_gate")
        .adds("one_plus_exp", "exp_neg_gate", 1.0)
        .recip("sigmoid_gate", "one_plus_exp")
        .mul("gate_silu", "gate", "sigmoid_gate")
        .mul("result", "gate_silu", "up")
        .store("result", "output", 0, 0)
        .build())

def create_linear_tile(module=None):
    """
    Create the linear_tile function.
    Type: InCore
    """
    return (PTOFunctionBuilder("linear_tile", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("weight", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("w", 128, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .load("w", "weight", 0, 0)
        .matmul("result", "x", "w")
        .store("result", "output", 0, 0)
        .build())

def create_rope_tile(module=None):
    """
    Create the rope_tile function.
    Type: InCore
    """
    return (PTOFunctionBuilder("rope_tile", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("cos_cache", MemorySpace.GM, ElementType.F32)
        .memref("sin_cache", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("cos_pos", 32, 128, ElementType.F32)
        .tile("sin_pos", 32, 128, ElementType.F32)
        .tile("x_cos", 32, 128, ElementType.F32)
        .tile("x_sin", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .load("cos_pos", "cos_cache", 0, 0)
        .load("sin_pos", "sin_cache", 0, 0)
        .mul("x_cos", "x", "cos_pos")
        .mul("x_sin", "x", "sin_pos")
        .add("result", "x_cos", "x_sin")
        .store("result", "output", 0, 0)
        .build())

def create_attention_score_tile(module=None):
    """
    Create the attention_score_tile function.
    Type: InCore
    """
    return (PTOFunctionBuilder("attention_score_tile", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_q", MemorySpace.GM, ElementType.F32)
        .memref("input_kt", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("q", 32, 128, ElementType.F32)
        .tile("k_t", 128, 128, ElementType.F32)
        .tile("scores", 32, 128, ElementType.F32)
        .tile("scaled_scores", 32, 128, ElementType.F32)
        
        # Scalar declarations
        .scalar("scale", ElementType.F32)
        
        # Instructions
        .load("q", "input_q", 0, 0)
        .load("k_t", "input_kt", 0, 0)
        .matmul("scores", "q", "k_t")
        .scalar_li("scale", 0.08838834764831843)
        .muls("scaled_scores", "scores", "scale")
        .store("scaled_scores", "output", 0, 0)
        .build())

def create_attention_output_tile(module=None):
    """
    Create the attention_output_tile function.
    Type: InCore
    """
    return (PTOFunctionBuilder("attention_output_tile", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_weights", MemorySpace.GM, ElementType.F32)
        .memref("input_v", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("weights", 32, 128, ElementType.F32)
        .tile("v", 128, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("weights", "input_weights", 0, 0)
        .load("v", "input_v", 0, 0)
        .matmul("result", "weights", "v")
        .store("result", "output", 0, 0)
        .build())

def create_residual_add_tile(module=None):
    """
    Create the residual_add_tile function.
    Type: InCore
    """
    return (PTOFunctionBuilder("residual_add_tile", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("input_residual", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("x", 32, 128, ElementType.F32)
        .tile("residual", 32, 128, ElementType.F32)
        .tile("result", 32, 128, ElementType.F32)
        
        # Instructions
        .load("x", "input", 0, 0)
        .load("residual", "input_residual", 0, 0)
        .add("result", "x", "residual")
        .store("result", "output", 0, 0)
        .build())

def create_flash_attn_score_block(module=None):
    """
    Create the flash_attn_score_block function.
    Type: InCore
    """
    return (PTOFunctionBuilder("flash_attn_score_block", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_q", MemorySpace.GM, ElementType.F32)
        .memref("input_k", MemorySpace.GM, ElementType.F32)
        .memref("output_s", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("q_block", 64, 128, ElementType.F32)
        .tile("k_block", 64, 128, ElementType.F32)
        .tile("s_block", 64, 64, ElementType.F32)
        .tile("s_scaled", 64, 64, ElementType.F32)
        
        # Scalar declarations
        .scalar("scale", ElementType.F32)
        
        # Instructions
        .load("q_block", "input_q", 0, 0)
        .load("k_block", "input_k", 0, 0)
        .matmul("s_block", "q_block", "k_block")
        .scalar_li("scale", 0.08838834764831843)
        .muls("s_scaled", "s_block", "scale")
        .store("s_scaled", "output_s", 0, 0)
        .build())

def create_flash_attn_softmax_update(module=None):
    """
    Create the flash_attn_softmax_update function.
    Type: InCore
    """
    return (PTOFunctionBuilder("flash_attn_softmax_update", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_s", MemorySpace.GM, ElementType.F32)
        .memref("input_m_prev", MemorySpace.GM, ElementType.F32)
        .memref("input_l_prev", MemorySpace.GM, ElementType.F32)
        .memref("output_m_new", MemorySpace.GM, ElementType.F32)
        .memref("output_l_new", MemorySpace.GM, ElementType.F32)
        .memref("output_p", MemorySpace.GM, ElementType.F32)
        .memref("output_scale_old", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("s_block", 64, 64, ElementType.F32)
        .tile("m_prev", 64, 1, ElementType.F32)
        .tile("l_prev", 64, 1, ElementType.F32)
        .tile("m_new", 64, 1, ElementType.F32)
        .tile("m_cur", 64, 1, ElementType.F32)
        .tile("l_new", 64, 1, ElementType.F32)
        .tile("p_block", 64, 64, ElementType.F32)
        .tile("s_shifted", 64, 64, ElementType.F32)
        .tile("scale_old", 64, 1, ElementType.F32)
        .tile("m_diff", 64, 1, ElementType.F32)
        .tile("l_scaled", 64, 1, ElementType.F32)
        .tile("p_rowsum", 64, 1, ElementType.F32)
        
        # Instructions
        .load("s_block", "input_s", 0, 0)
        .load("m_prev", "input_m_prev", 0, 0)
        .load("l_prev", "input_l_prev", 0, 0)
        .rowmax("m_cur", "s_block")
        # Unknown: TMAX m_new ['m_prev', 'm_cur']
        .rowexpandsub("s_shifted", "s_block", "m_new")
        .exp("p_block", "s_shifted")
        .sub("m_diff", "m_prev", "m_new")
        .exp("scale_old", "m_diff")
        .mul("l_scaled", "scale_old", "l_prev")
        .rowsum("p_rowsum", "p_block")
        .add("l_new", "l_scaled", "p_rowsum")
        .store("m_new", "output_m_new", 0, 0)
        .store("l_new", "output_l_new", 0, 0)
        .store("p_block", "output_p", 0, 0)
        .store("scale_old", "output_scale_old", 0, 0)
        .build())

def create_flash_attn_output_update(module=None):
    """
    Create the flash_attn_output_update function.
    Type: InCore
    """
    return (PTOFunctionBuilder("flash_attn_output_update", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_o_prev", MemorySpace.GM, ElementType.F32)
        .memref("input_p", MemorySpace.GM, ElementType.F32)
        .memref("input_v", MemorySpace.GM, ElementType.F32)
        .memref("input_scale", MemorySpace.GM, ElementType.F32)
        .memref("output_o", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("o_prev", 64, 128, ElementType.F32)
        .tile("p_block", 64, 64, ElementType.F32)
        .tile("v_block", 64, 128, ElementType.F32)
        .tile("scale_old", 64, 1, ElementType.F32)
        .tile("o_scaled", 64, 128, ElementType.F32)
        .tile("pv", 64, 128, ElementType.F32)
        .tile("o_new", 64, 128, ElementType.F32)
        
        # Instructions
        .load("o_prev", "input_o_prev", 0, 0)
        .load("p_block", "input_p", 0, 0)
        .load("v_block", "input_v", 0, 0)
        .load("scale_old", "input_scale", 0, 0)
        .rowexpandmul("o_scaled", "o_prev", "scale_old")
        .matmul("pv", "p_block", "v_block")
        .add("o_new", "o_scaled", "pv")
        .store("o_new", "output_o", 0, 0)
        .build())

def create_flash_attn_normalize(module=None):
    """
    Create the flash_attn_normalize function.
    Type: InCore
    """
    return (PTOFunctionBuilder("flash_attn_normalize", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_o", MemorySpace.GM, ElementType.F32)
        .memref("input_l", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("o_block", 64, 128, ElementType.F32)
        .tile("l_vec", 64, 1, ElementType.F32)
        .tile("o_final", 64, 128, ElementType.F32)
        
        # Instructions
        .load("o_block", "input_o", 0, 0)
        .load("l_vec", "input_l", 0, 0)
        .rowexpanddiv("o_final", "o_block", "l_vec")
        .store("o_final", "output", 0, 0)
        .build())

def create_flash_attn_init_state(module=None):
    """
    Create the flash_attn_init_state function.
    Type: InCore
    """
    return (PTOFunctionBuilder("flash_attn_init_state", module=module)
        .in_core()
        
        # Memref declarations (function parameters)
        .memref("input_zeros_large", MemorySpace.GM, ElementType.F32)
        .memref("input_zeros_small", MemorySpace.GM, ElementType.F32)
        .memref("input_neg_inf", MemorySpace.GM, ElementType.F32)
        .memref("output_o", MemorySpace.GM, ElementType.F32)
        .memref("output_l", MemorySpace.GM, ElementType.F32)
        .memref("output_m", MemorySpace.GM, ElementType.F32)
        
        # Tile declarations
        .tile("o_init", 64, 128, ElementType.F32)
        .tile("l_init", 64, 1, ElementType.F32)
        .tile("m_init", 64, 1, ElementType.F32)
        
        # Instructions
        .load("o_init", "input_zeros_large", 0, 0)
        .load("l_init", "input_zeros_small", 0, 0)
        .load("m_init", "input_neg_inf", 0, 0)
        .store("o_init", "output_o", 0, 0)
        .store("l_init", "output_l", 0, 0)
        .store("m_init", "output_m", 0, 0)
        .build())

def create_llama_layer_dynamic(module=None):
    """
    Create the llama_layer_dynamic function.
    Type: Orchestration
    """
    return (PTOFunctionBuilder("llama_layer_dynamic", module=module)
        .not_in_core()
        
        # Memref declarations (function parameters)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        .memref("attn_norm_weights", MemorySpace.GM, ElementType.F32)
        .memref("wq", MemorySpace.GM, ElementType.F32)
        .memref("wk", MemorySpace.GM, ElementType.F32)
        .memref("wv", MemorySpace.GM, ElementType.F32)
        .memref("wo", MemorySpace.GM, ElementType.F32)
        .memref("cos_cache", MemorySpace.GM, ElementType.F32)
        .memref("sin_cache", MemorySpace.GM, ElementType.F32)
        .memref("mlp_norm_weights", MemorySpace.GM, ElementType.F32)
        .memref("w_gate", MemorySpace.GM, ElementType.F32)
        .memref("w_up", MemorySpace.GM, ElementType.F32)
        .memref("w_down", MemorySpace.GM, ElementType.F32)
        .memref("all_q_tiles", MemorySpace.GM, ElementType.F32)
        .memref("all_k_tiles", MemorySpace.GM, ElementType.F32)
        .memref("all_v_tiles", MemorySpace.GM, ElementType.F32)
        .memref("all_q_rope", MemorySpace.GM, ElementType.F32)
        .memref("all_k_rope", MemorySpace.GM, ElementType.F32)
        .memref("all_attn_out", MemorySpace.GM, ElementType.F32)
        .memref("all_m_vec", MemorySpace.GM, ElementType.F32)
        .memref("all_l_vec", MemorySpace.GM, ElementType.F32)
        .memref("all_hidden", MemorySpace.GM, ElementType.F32)
        .memref("temp_norm", MemorySpace.GM, ElementType.F32)
        .memref("temp_scores", MemorySpace.GM, ElementType.F32)
        .memref("temp_attn_weights", MemorySpace.GM, ElementType.F32)
        .memref("temp_scale", MemorySpace.GM, ElementType.F32)
        .memref("temp_gate", MemorySpace.GM, ElementType.F32)
        .memref("temp_up", MemorySpace.GM, ElementType.F32)
        .memref("temp_swiglu", MemorySpace.GM, ElementType.F32)
        .memref("temp_mlp_out", MemorySpace.GM, ElementType.F32)
        .memref("const_zeros_large", MemorySpace.GM, ElementType.F32)
        .memref("const_zeros_small", MemorySpace.GM, ElementType.F32)
        .memref("const_neg_inf", MemorySpace.GM, ElementType.F32)
        
        # Scalar declarations
        .scalar("seq_len", ElementType.I32)
        .scalar("tile_rows", ElementType.I32)
        .scalar("num_tiles", ElementType.I32)
        .scalar("zero", ElementType.I32)
        
        # Instructions
        .scalar_li("tile_rows", 32)
        .scalar_li("zero", 0)
        .for_loop("tile_i", 0, "num_tiles", 1)
        .call("rmsnorm_tile", {"input": ("input", "tile_i", 0), "weights": "attn_norm_weights", "output": ("temp_norm", "tile_i", 0)})
        .call("tile_matmul", {"input_a": ("temp_norm", "tile_i", 0), "input_b": "wq", "output": ("all_q_tiles", "tile_i", 0)})
        .call("tile_matmul", {"input_a": ("temp_norm", "tile_i", 0), "input_b": "wk", "output": ("all_k_tiles", "tile_i", 0)})
        .call("tile_matmul", {"input_a": ("temp_norm", "tile_i", 0), "input_b": "wv", "output": ("all_v_tiles", "tile_i", 0)})
        .call("rope_tile", {"input": ("all_q_tiles", "tile_i", 0), "cos_cache": "cos_cache", "sin_cache": "sin_cache", "output": ("all_q_rope", "tile_i", 0)})
        .call("rope_tile", {"input": ("all_k_tiles", "tile_i", 0), "cos_cache": "cos_cache", "sin_cache": "sin_cache", "output": ("all_k_rope", "tile_i", 0)})
        .end_for()
        .for_loop("q_tile", 0, "num_tiles", 1)
        .call("flash_attn_init_state", {"input_zeros_large": "const_zeros_large", "input_zeros_small": "const_zeros_small", "input_neg_inf": "const_neg_inf", "output_o": ("all_attn_out", "q_tile", 0), "output_l": ("all_l_vec", "q_tile", 0), "output_m": ("all_m_vec", "q_tile", 0)})
        .for_loop("kv_tile", 0, "num_tiles", 1)
        .call("flash_attn_score_block", {"input_q": ("all_q_rope", "q_tile", 0), "input_k": ("all_k_rope", "kv_tile", 0), "output_s": ("temp_scores", "q_tile", 0)})
        .call("flash_attn_softmax_update", {"input_s": ("temp_scores", "q_tile", 0), "input_m_prev": ("all_m_vec", "q_tile", 0), "input_l_prev": ("all_l_vec", "q_tile", 0), "output_m_new": ("all_m_vec", "q_tile", 0), "output_l_new": ("all_l_vec", "q_tile", 0), "output_p": ("temp_attn_weights", "q_tile", 0), "output_scale_old": ("temp_scale", "q_tile", 0)})
        .call("flash_attn_output_update", {"input_o_prev": ("all_attn_out", "q_tile", 0), "input_p": ("temp_attn_weights", "q_tile", 0), "input_v": ("all_v_tiles", "kv_tile", 0), "input_scale": ("temp_scale", "q_tile", 0), "output_o": ("all_attn_out", "q_tile", 0)})
        .end_for()
        .call("flash_attn_normalize", {"input_o": ("all_attn_out", "q_tile", 0), "input_l": ("all_l_vec", "q_tile", 0), "output": ("all_attn_out", "q_tile", 0)})
        .end_for()
        .for_loop("tile_i", 0, "num_tiles", 1)
        .call("tile_matmul", {"input_a": ("all_attn_out", "tile_i", 0), "input_b": "wo", "output": ("temp_norm", "tile_i", 0)})
        .call("residual_add_tile", {"input": ("temp_norm", "tile_i", 0), "input_residual": ("input", "tile_i", 0), "output": ("all_hidden", "tile_i", 0)})
        .call("rmsnorm_tile", {"input": ("all_hidden", "tile_i", 0), "weights": "mlp_norm_weights", "output": ("temp_norm", "tile_i", 0)})
        .call("tile_matmul", {"input_a": ("temp_norm", "tile_i", 0), "input_b": "w_gate", "output": ("temp_gate", "tile_i", 0)})
        .call("tile_matmul", {"input_a": ("temp_norm", "tile_i", 0), "input_b": "w_up", "output": ("temp_up", "tile_i", 0)})
        .call("swiglu_tile", {"input_gate": ("temp_gate", "tile_i", 0), "input_up": ("temp_up", "tile_i", 0), "output": ("temp_swiglu", "tile_i", 0)})
        .call("tile_matmul", {"input_a": ("temp_swiglu", "tile_i", 0), "input_b": "w_down", "output": ("temp_mlp_out", "tile_i", 0)})
        .call("residual_add_tile", {"input": ("temp_mlp_out", "tile_i", 0), "input_residual": ("all_hidden", "tile_i", 0), "output": ("output", "tile_i", 0)})
        .end_for()
        .build())


def create_llama7b_flash_module():
    """Create the llama7b_flash module."""
    module = PTOModule("llama7b_flash")

    # Add InCore functions
    module.add_function(create_tile_add(module))
    module.add_function(create_tile_mul(module))
    module.add_function(create_tile_muls(module))
    module.add_function(create_tile_exp(module))
    module.add_function(create_tile_silu(module))
    module.add_function(create_tile_rsqrt(module))
    module.add_function(create_tile_matmul(module))
    module.add_function(create_tile_rowmax(module))
    module.add_function(create_tile_rowsum(module))
    module.add_function(create_tile_rowexpandsub(module))
    module.add_function(create_tile_rowexpanddiv(module))
    module.add_function(create_tile_rowexpandmul(module))
    module.add_function(create_rmsnorm_tile(module))
    module.add_function(create_softmax_tile(module))
    module.add_function(create_swiglu_tile(module))
    module.add_function(create_linear_tile(module))
    module.add_function(create_rope_tile(module))
    module.add_function(create_attention_score_tile(module))
    module.add_function(create_attention_output_tile(module))
    module.add_function(create_residual_add_tile(module))
    module.add_function(create_flash_attn_score_block(module))
    module.add_function(create_flash_attn_softmax_update(module))
    module.add_function(create_flash_attn_output_update(module))
    module.add_function(create_flash_attn_normalize(module))
    module.add_function(create_flash_attn_init_state(module))

    # Add Orchestration functions
    module.add_function(create_llama_layer_dynamic(module))

    module.set_entry("llama_layer_dynamic")

    return module


def main():
    """Create and compile the llama7b_flash module."""
    module = create_llama7b_flash_module()

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
