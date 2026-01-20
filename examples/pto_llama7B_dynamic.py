"""
PTO LLaMA 7B Layer Decode with Dynamic Sequence Length

This module implements a single LLaMA decoder layer with:
- Fixed batch size (B=1 for inference)
- Dynamic sequence length (handled via orchestration)
- Tile-based computation using InCore functions

LLaMA 7B Architecture:
- Hidden dim: 4096
- Num heads: 32
- Head dim: 128 (4096/32)
- Intermediate dim: 11008 (MLP)
- Vocabulary: 32000
- Max sequence: 2048

Layer Structure:
1. RMSNorm (pre-attention)
2. Self-Attention with RoPE
   - Q, K, V projections
   - Rotary Position Embedding
   - Scaled Dot-Product Attention
   - Output projection
3. Residual connection
4. RMSNorm (pre-MLP)
5. MLP (SwiGLU)
   - Gate projection
   - Up projection  
   - SiLU activation
   - Down projection
6. Residual connection

Function Hierarchy:
- Level 1 (InCore): Basic tile ops (add, mul, exp, rsqrt, matmul, etc.)
- Level 2 (InCore): Composed ops (rmsnorm_tile, rope_tile, softmax_tile, swiglu_tile)
- Level 3 (Orchestration): Dynamic loops for sequence processing
"""

import os
import sys
import shutil
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_compile import (
    PTOFunctionBuilder, PTOModule, PTOModuleCompiler,
    MultiBackendCodeGenerator, OrchestrationCodeGenerator
)
from pto_isa_definition import ElementType, MemorySpace, CompareMode
from pto_dynamic_tiling import compute_tile_shape, get_tile_info, MAX_TILE_BYTES

# =============================================================================
# LLaMA 7B Configuration
# =============================================================================

BATCH_SIZE = 1          # Fixed for inference
HIDDEN_DIM = 4096       # Model dimension
NUM_HEADS = 32          # Number of attention heads
HEAD_DIM = 128          # HIDDEN_DIM / NUM_HEADS
INTERMEDIATE_DIM = 11008  # MLP intermediate dimension
MAX_SEQ_LEN = 131072    # Maximum sequence length (128K)

# =============================================================================
# Hardware Constraints for Ascend 910B
# =============================================================================
DTYPE = ElementType.F32
TARGET_ISA = "ascend910b"
ELEMENT_BYTES = 4       # f32 = 4 bytes

# SRAM constraint: 256KB total for InCore function
SRAM_SIZE_KB = 256
SRAM_SIZE_BYTES = SRAM_SIZE_KB * 1024  # 262144 bytes

# Vector and row constraints for Ascend 910B
VECTOR_LANES = 8        # 256-bit vector / 32-bit = 8 for f32
PHYSICAL_ROW_SIZE = 32  # Optimal repeat count for pipeline

# =============================================================================
# Flash Attention Tile Configuration
# =============================================================================
# For Flash Attention, we need to fit Q_block, K_block, V_block, O_block, 
# plus intermediate tiles (S, P, m, l) all within SRAM.
#
# Memory layout for one Flash Attention block:
# - Q_block: Br x d  (query block)
# - K_block: Bc x d  (key block)  
# - V_block: Bc x d  (value block)
# - S_block: Br x Bc (attention scores)
# - P_block: Br x Bc (attention weights after softmax)
# - O_block: Br x d  (output accumulator)
# - m_vec:   Br x 1  (row max for online softmax)
# - l_vec:   Br x 1  (row sum for online softmax)
#
# Total memory = Br*d + Bc*d + Bc*d + Br*Bc + Br*Bc + Br*d + Br + Br
#              = 3*Br*d + 2*Bc*d + 2*Br*Bc + 2*Br
#
# For d=128 (HEAD_DIM), we solve for Br and Bc to fit in 256KB:
# Let Br = Bc = B for simplicity:
# Memory = 3*B*128 + 2*B*128 + 2*B*B + 2*B
#        = 384*B + 256*B + 2*B^2 + 2*B
#        = 2*B^2 + 642*B
#
# For 256KB = 65536 f32 elements:
# 2*B^2 + 642*B <= 65536
# B <= 96 (approx)
#
# Round down to multiple of PHYSICAL_ROW_SIZE (32):
FLASH_BLOCK_SIZE = 64   # Br = Bc = 64 (conservative for safety margin)

# Standard tile for non-attention ops (fits in 16KB each)
TILE_ROWS, TILE_COLS = compute_tile_shape(DTYPE, TARGET_ISA)
TILE_INFO = get_tile_info(DTYPE, TARGET_ISA)

# =============================================================================
# Power-of-2 Tile Size Configuration for Binary Expansion
# =============================================================================
# Different binary expansion levels use different tile sizes:
# - Larger power-of-2 blocks → larger tiles (better throughput)
# - Smaller power-of-2 blocks → smaller tiles (better flexibility)
#
# This allows optimal tile size selection based on workload size.
# Each level N means "when processing 2^N iterations at once"

# Tile rows for each power-of-2 level (must be multiples of PHYSICAL_ROW_SIZE=32)
# Key: power of 2 block size in terms of base tile count
# Value: tile_rows to use for that block
#
# STRATEGY: Use 64-row tiles for ALL quantized blocks to HALVE iteration count!
# - 16K sequence: 512 tiles → 256 iterations (50% reduction)
# - 32K sequence: 1024 tiles → 512 iterations (50% reduction)
# - 128K sequence: 4096 tiles → 2048 iterations (50% reduction)
#
# Only the residual loop (< min_range=256) uses 32-row tiles for fine-grained handling.
TILE_ROWS_BY_LEVEL = {
    4096: 64,   # 128K seq: 4096 → 2048 iterations
    2048: 64,   # 64K seq: 2048 → 1024 iterations
    1024: 64,   # 32K seq: 1024 → 512 iterations  ← NOW USES 64-ROW!
    512:  64,   # 16K seq: 512 → 256 iterations   ← NOW USES 64-ROW!
    256:  64,   # 8K seq: 256 → 128 iterations    ← NOW USES 64-ROW!
}
TILE_ROWS_RESIDUAL = 32  # Residual iterations (< 256 tiles): smaller tiles for flexibility

# All tile variants we need to generate InCore functions for
# Include both TILE_ROWS_BY_LEVEL values (64) and TILE_ROWS_RESIDUAL (32)
TILE_SIZE_VARIANTS = sorted(set(TILE_ROWS_BY_LEVEL.values()) | {TILE_ROWS_RESIDUAL}, reverse=True)  # [64, 32]

# Maximum number of tiles for binary-expanded loops
# Use smallest tile size for counting (most tiles)
MIN_TILE_ROWS = min(TILE_ROWS_BY_LEVEL.values())
MAX_NUM_TILES = MAX_SEQ_LEN // MIN_TILE_ROWS  # 131072 / 32 = 4096

# Minimum block size for binary expansion (256 tiles = 8K sequence elements with 32-row tiles)
# Residual iterations below this threshold handled by single loop
MIN_NUM_TILES = 256

def get_tile_rows_for_level(block_size: int) -> int:
    """Get tile rows for a specific power-of-2 block size."""
    return TILE_ROWS_BY_LEVEL.get(block_size, TILE_ROWS_RESIDUAL)

def get_func_suffix_for_level(block_size: int) -> str:
    """Get function name suffix for a specific power-of-2 block size."""
    rows = get_tile_rows_for_level(block_size)
    return f"_{rows}" if rows != TILE_ROWS else ""

# Verify Flash Attention memory usage
def verify_flash_attention_memory():
    """Verify Flash Attention tiles fit in SRAM."""
    B = FLASH_BLOCK_SIZE
    d = HEAD_DIM
    
    # All tiles needed for one Flash Attention block
    q_block = B * d * ELEMENT_BYTES        # Br x d
    k_block = B * d * ELEMENT_BYTES        # Bc x d
    v_block = B * d * ELEMENT_BYTES        # Bc x d
    s_block = B * B * ELEMENT_BYTES        # Br x Bc
    p_block = B * B * ELEMENT_BYTES        # Br x Bc (can reuse s_block)
    o_block = B * d * ELEMENT_BYTES        # Br x d
    m_vec = B * 1 * ELEMENT_BYTES          # Br x 1
    l_vec = B * 1 * ELEMENT_BYTES          # Br x 1
    
    # With reuse: S and P can share memory
    total_with_reuse = q_block + k_block + v_block + s_block + o_block + m_vec + l_vec
    total_no_reuse = q_block + k_block + v_block + s_block + p_block + o_block + m_vec + l_vec
    
    return {
        'block_size': B,
        'head_dim': d,
        'q_block_kb': q_block / 1024,
        'k_block_kb': k_block / 1024,
        'v_block_kb': v_block / 1024,
        's_block_kb': s_block / 1024,
        'o_block_kb': o_block / 1024,
        'total_with_reuse_kb': total_with_reuse / 1024,
        'total_no_reuse_kb': total_no_reuse / 1024,
        'sram_kb': SRAM_SIZE_KB,
        'fits': total_with_reuse <= SRAM_SIZE_BYTES
    }


# =============================================================================
# Level 1: Basic InCore Functions (Tile Operations)
# =============================================================================

def get_func_name_with_size(base_name: str, rows: int) -> str:
    """Get function name with tile size suffix if not standard."""
    if rows != TILE_ROWS:
        return f"{base_name}_{rows}"
    return base_name

def create_tile_add(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Element-wise addition of two tiles."""
    func_name = get_func_name_with_size("tile_add", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("a", rows, cols, dtype)
        .tile("b", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .add("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())


def create_tile_mul(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Element-wise multiplication of two tiles."""
    func_name = get_func_name_with_size("tile_mul", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("a", rows, cols, dtype)
        .tile("b", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .mul("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())


def create_tile_muls(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Multiply tile by scalar."""
    func_name = get_func_name_with_size("tile_muls", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("a", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .scalar("scale", dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input", 0, 0)
        .muls("result", "a", "scale")
        .store("result", "output", 0, 0)
        .build())


def create_tile_exp(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Element-wise exponential."""
    func_name = get_func_name_with_size("tile_exp", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .exp("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_tile_silu(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: SiLU activation (x * sigmoid(x)).
    
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Implementation:
    1. neg_x = -x
    2. exp_neg_x = exp(-x)
    3. one_plus_exp = 1 + exp(-x)  (using adds with scalar 1)
    4. sigmoid = recip(one_plus_exp) = 1 / (1 + exp(-x))
    5. result = x * sigmoid
    """
    func_name = get_func_name_with_size("tile_silu", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("neg_x", rows, cols, dtype)
        .tile("exp_neg_x", rows, cols, dtype)
        .tile("one_plus_exp", rows, cols, dtype)
        .tile("sigmoid", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("x", "input", 0, 0)
        
        # neg_x = -x
        .neg("neg_x", "x")
        
        # exp(-x)
        .exp("exp_neg_x", "neg_x")
        
        # 1 + exp(-x)
        .adds("one_plus_exp", "exp_neg_x", 1.0)
        
        # sigmoid = 1 / (1 + exp(-x))
        .recip("sigmoid", "one_plus_exp")
        
        # result = x * sigmoid
        .mul("result", "x", "sigmoid")
        
        .store("result", "output", 0, 0)
        .build())


def create_tile_rsqrt(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Element-wise reciprocal square root."""
    func_name = get_func_name_with_size("tile_rsqrt", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rsqrt("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_tile_matmul(m=TILE_ROWS, k=TILE_COLS, n=TILE_COLS, dtype=DTYPE):
    """InCore: Matrix multiplication C = A @ B."""
    func_name = get_func_name_with_size("tile_matmul", m)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("a", m, k, dtype)
        .tile("b", k, n, dtype)
        .tile("c", m, n, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .matmul("c", "a", "b")
        .store("c", "output", 0, 0)
        .build())


def create_tile_rowmax(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Row-wise maximum."""
    func_name = get_func_name_with_size("tile_rowmax", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rowmax("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_tile_rowsum(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Row-wise sum."""
    func_name = get_func_name_with_size("tile_rowsum", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rowsum("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_tile_rowexpandsub(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Broadcast subtraction x - row_vals."""
    func_name = get_func_name_with_size("tile_rowexpandsub", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandsub("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


def create_tile_rowexpanddiv(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Broadcast division x / row_vals."""
    func_name = get_func_name_with_size("tile_rowexpanddiv", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpanddiv("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


def create_tile_rowexpandmul(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """InCore: Broadcast multiplication x * row_vals."""
    func_name = get_func_name_with_size("tile_rowexpandmul", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandmul("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Level 2: Composed InCore Functions (LLaMA Building Blocks)
# =============================================================================

def create_rmsnorm_tile(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: RMSNorm on a single tile.
    
    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
    
    For a tile, we compute row-wise RMSNorm:
    - Square each element
    - Sum each row and divide by cols (mean)
    - Add epsilon, take rsqrt
    - Multiply original x by the scaling factor
    - Multiply by learned weights (gamma)
    """
    func_name = get_func_name_with_size("rmsnorm_tile", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("x_sq", rows, cols, dtype)
        .tile("row_sum", rows, 1, dtype)
        .tile("row_mean", rows, 1, dtype)
        .tile("row_rsqrt", rows, 1, dtype)
        .tile("x_norm", rows, cols, dtype)
        .tile("gamma", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .scalar("eps", dtype)
        .scalar("inv_cols", dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("weights", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        # Load input and weights
        .load("x", "input", 0, 0)
        .load("gamma", "weights", 0, 0)
        
        # x^2
        .mul("x_sq", "x", "x")
        
        # Row sum of x^2
        .rowsum("row_sum", "x_sq")
        
        # Mean = sum / cols (multiply by 1/cols)
        .scalar_li("inv_cols", 1.0 / cols)
        .muls("row_mean", "row_sum", "inv_cols")
        
        # Add epsilon
        .scalar_li("eps", 1e-5)
        .adds("row_mean", "row_mean", "eps")
        
        # rsqrt(mean + eps)
        .rsqrt("row_rsqrt", "row_mean")
        
        # x * rsqrt(...)
        .rowexpandmul("x_norm", "x", "row_rsqrt")
        
        # Multiply by gamma
        .mul("result", "x_norm", "gamma")
        
        .store("result", "output", 0, 0)
        .build())


def create_softmax_tile(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: Softmax on a single tile (row-wise).
    
    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_i - max(x)))
    """
    func_name = get_func_name_with_size("softmax_tile", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_max", rows, 1, dtype)
        .tile("x_shifted", rows, cols, dtype)
        .tile("exp_x", rows, cols, dtype)
        .tile("row_sum", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("x", "input", 0, 0)
        
        # Row max for numerical stability
        .rowmax("row_max", "x")
        
        # x - max
        .rowexpandsub("x_shifted", "x", "row_max")
        
        # exp(x - max)
        .exp("exp_x", "x_shifted")
        
        # Row sum of exp
        .rowsum("row_sum", "exp_x")
        
        # Normalize
        .rowexpanddiv("result", "exp_x", "row_sum")
        
        .store("result", "output", 0, 0)
        .build())


def create_swiglu_tile(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: SwiGLU activation on a tile.
    
    SwiGLU(x, gate) = SiLU(gate) * x
    where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    In LLaMA MLP:
    - gate = x @ W_gate
    - up = x @ W_up
    - output = SiLU(gate) * up
    """
    func_name = get_func_name_with_size("swiglu_tile", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("gate", rows, cols, dtype)
        .tile("up", rows, cols, dtype)
        .tile("neg_gate", rows, cols, dtype)
        .tile("exp_neg_gate", rows, cols, dtype)
        .tile("one_plus_exp", rows, cols, dtype)
        .tile("sigmoid_gate", rows, cols, dtype)
        .tile("gate_silu", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_gate", MemorySpace.GM, dtype)
        .memref("input_up", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("gate", "input_gate", 0, 0)
        .load("up", "input_up", 0, 0)
        
        # SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
        # neg_gate = -gate
        .neg("neg_gate", "gate")
        
        # exp(-gate)
        .exp("exp_neg_gate", "neg_gate")
        
        # 1 + exp(-gate)
        .adds("one_plus_exp", "exp_neg_gate", 1.0)
        
        # sigmoid = 1 / (1 + exp(-gate))
        .recip("sigmoid_gate", "one_plus_exp")
        
        # gate_silu = gate * sigmoid
        .mul("gate_silu", "gate", "sigmoid_gate")
        
        # result = SiLU(gate) * up
        .mul("result", "gate_silu", "up")
        
        .store("result", "output", 0, 0)
        .build())


def create_linear_tile(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: Linear projection (matrix multiply).
    
    output = input @ weight
    """
    func_name = get_func_name_with_size("linear_tile", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("w", cols, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("weight", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("x", "input", 0, 0)
        .load("w", "weight", 0, 0)
        .matmul("result", "x", "w")
        .store("result", "output", 0, 0)
        .build())


def create_rope_tile(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: Apply Rotary Position Embedding (RoPE) to a tile.
    
    RoPE applies rotation based on position:
    x_rot[2i] = x[2i] * cos(θ) - x[2i+1] * sin(θ)
    x_rot[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
    
    For simplicity, we assume cos/sin values are precomputed and passed in.
    """
    func_name = get_func_name_with_size("rope_tile", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("cos_pos", rows, cols, dtype)
        .tile("sin_pos", rows, cols, dtype)
        .tile("x_cos", rows, cols, dtype)
        .tile("x_sin", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("cos_cache", MemorySpace.GM, dtype)
        .memref("sin_cache", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("x", "input", 0, 0)
        .load("cos_pos", "cos_cache", 0, 0)
        .load("sin_pos", "sin_cache", 0, 0)
        
        # x * cos
        .mul("x_cos", "x", "cos_pos")
        
        # x * sin (simplified - actual RoPE needs rotation pairing)
        .mul("x_sin", "x", "sin_pos")
        
        # For simplified version: result = x * cos + rotated_x * sin
        # Here we just do element-wise as approximation
        .add("result", "x_cos", "x_sin")
        
        .store("result", "output", 0, 0)
        .build())


def create_attention_score_tile(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: Compute attention scores for a tile.
    
    scores = Q @ K^T / sqrt(d_k)
    
    Note: In practice, K is transposed before this operation.
    """
    func_name = get_func_name_with_size("attention_score_tile", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("q", rows, cols, dtype)
        .tile("k_t", cols, cols, dtype)  # K transposed
        .tile("scores", rows, cols, dtype)
        .tile("scaled_scores", rows, cols, dtype)
        .scalar("scale", dtype)
        .memref("input_q", MemorySpace.GM, dtype)
        .memref("input_kt", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("q", "input_q", 0, 0)
        .load("k_t", "input_kt", 0, 0)
        
        # Q @ K^T
        .matmul("scores", "q", "k_t")
        
        # Scale by 1/sqrt(d_k)
        .scalar_li("scale", 1.0 / (HEAD_DIM ** 0.5))
        .muls("scaled_scores", "scores", "scale")
        
        .store("scaled_scores", "output", 0, 0)
        .build())


def create_attention_output_tile(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: Compute attention output for a tile.
    
    output = attention_weights @ V
    """
    func_name = get_func_name_with_size("attention_output_tile", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("weights", rows, cols, dtype)
        .tile("v", cols, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_weights", MemorySpace.GM, dtype)
        .memref("input_v", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("weights", "input_weights", 0, 0)
        .load("v", "input_v", 0, 0)
        .matmul("result", "weights", "v")
        .store("result", "output", 0, 0)
        .build())


def create_residual_add_tile(rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    InCore: Residual connection (add two tiles).
    """
    func_name = get_func_name_with_size("residual_add_tile", rows)
    return (PTOFunctionBuilder(func_name)
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("residual", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("input_residual", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("x", "input", 0, 0)
        .load("residual", "input_residual", 0, 0)
        .add("result", "x", "residual")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Flash Attention InCore Functions
# =============================================================================
# Flash Attention processes attention in blocks to fit in SRAM.
# Block size: FLASH_BLOCK_SIZE x FLASH_BLOCK_SIZE (e.g., 64x64)
# 
# Algorithm (from FlashAttention paper):
# For each Q block (Br rows):
#   Initialize O = 0, l = 0, m = -inf
#   For each K,V block (Bc rows):
#     S = Q_block @ K_block^T
#     m_new = max(m, rowmax(S))
#     P = exp(S - m_new)
#     l_new = exp(m - m_new) * l + rowsum(P)
#     O = exp(m - m_new) * O + P @ V_block
#     m = m_new, l = l_new
#   O = O / l
# =============================================================================

def create_flash_attn_score_block(B=FLASH_BLOCK_SIZE, d=HEAD_DIM, dtype=DTYPE):
    """
    InCore: Compute attention scores for one Q-K block pair.
    
    S_block = Q_block @ K_block^T / sqrt(d)
    
    Tiles:
    - Q_block: B x d (loaded)
    - K_block: B x d (loaded, then used as K^T in matmul)
    - S_block: B x B (output)
    
    Memory: B*d + B*d + B*B = 2*B*d + B^2 = 2*64*128 + 64*64 = 16384 + 4096 = 20480 bytes = 20 KB
    """
    scale = 1.0 / (d ** 0.5)
    
    return (PTOFunctionBuilder("flash_attn_score_block")
        .in_core()
        # Q block: B x d
        .tile("q_block", B, d, dtype)
        # K block: B x d (will be used transposed in matmul)
        .tile("k_block", B, d, dtype)
        # Score block: B x B
        .tile("s_block", B, B, dtype)
        .tile("s_scaled", B, B, dtype)
        .scalar("scale", dtype)
        
        .memref("input_q", MemorySpace.GM, dtype)
        .memref("input_k", MemorySpace.GM, dtype)
        .memref("output_s", MemorySpace.GM, dtype)
        
        .load("q_block", "input_q", 0, 0)
        .load("k_block", "input_k", 0, 0)
        
        # S = Q @ K^T (matmul handles transpose internally)
        .matmul("s_block", "q_block", "k_block")
        
        # Scale by 1/sqrt(d)
        .scalar_li("scale", scale)
        .muls("s_scaled", "s_block", "scale")
        
        .store("s_scaled", "output_s", 0, 0)
        .build())


def create_flash_attn_softmax_update(B=FLASH_BLOCK_SIZE, dtype=DTYPE):
    """
    InCore: Online softmax update for Flash Attention.
    
    Given:
    - S_block: B x B (attention scores for current K block)
    - m_prev: B x 1 (previous row max)
    - l_prev: B x 1 (previous row sum)
    - O_prev: B x d (previous output accumulator)
    
    Compute:
    - m_new = max(m_prev, rowmax(S_block))
    - P = exp(S_block - expand(m_new))  # B x B
    - l_new = exp(m_prev - m_new) * l_prev + rowsum(P)
    - scale_old = exp(m_prev - m_new)
    - O_scaled = scale_old * O_prev  # Will be added to P @ V later
    
    Memory: S(B*B) + P(B*B) + m(B) + l(B) + temps = ~16KB for B=64
    """
    return (PTOFunctionBuilder("flash_attn_softmax_update")
        .in_core()
        # Input tiles
        .tile("s_block", B, B, dtype)          # Attention scores
        .tile("m_prev", B, 1, dtype)           # Previous row max
        .tile("l_prev", B, 1, dtype)           # Previous row sum
        
        # Output/intermediate tiles
        .tile("m_new", B, 1, dtype)            # New row max
        .tile("m_cur", B, 1, dtype)            # Current block row max
        .tile("l_new", B, 1, dtype)            # New row sum
        .tile("p_block", B, B, dtype)          # Softmax weights
        .tile("s_shifted", B, B, dtype)        # S - m_new (broadcast)
        .tile("scale_old", B, 1, dtype)        # exp(m_prev - m_new)
        .tile("m_diff", B, 1, dtype)           # m_prev - m_new
        .tile("l_scaled", B, 1, dtype)         # scaled l_prev
        .tile("p_rowsum", B, 1, dtype)         # rowsum(P)
        
        .memref("input_s", MemorySpace.GM, dtype)
        .memref("input_m_prev", MemorySpace.GM, dtype)
        .memref("input_l_prev", MemorySpace.GM, dtype)
        .memref("output_m_new", MemorySpace.GM, dtype)
        .memref("output_l_new", MemorySpace.GM, dtype)
        .memref("output_p", MemorySpace.GM, dtype)
        .memref("output_scale_old", MemorySpace.GM, dtype)
        
        .load("s_block", "input_s", 0, 0)
        .load("m_prev", "input_m_prev", 0, 0)
        .load("l_prev", "input_l_prev", 0, 0)
        
        # m_cur = rowmax(S_block)
        .rowmax("m_cur", "s_block")
        
        # m_new = max(m_prev, m_cur) - element-wise max
        .max("m_new", "m_prev", "m_cur")
        
        # s_shifted = S_block - expand(m_new)
        .rowexpandsub("s_shifted", "s_block", "m_new")
        
        # p_block = exp(s_shifted)
        .exp("p_block", "s_shifted")
        
        # m_diff = m_prev - m_new
        .sub("m_diff", "m_prev", "m_new")
        
        # scale_old = exp(m_diff)
        .exp("scale_old", "m_diff")
        
        # l_scaled = scale_old * l_prev
        .mul("l_scaled", "scale_old", "l_prev")
        
        # p_rowsum = rowsum(P)
        .rowsum("p_rowsum", "p_block")
        
        # l_new = l_scaled + p_rowsum
        .add("l_new", "l_scaled", "p_rowsum")
        
        .store("m_new", "output_m_new", 0, 0)
        .store("l_new", "output_l_new", 0, 0)
        .store("p_block", "output_p", 0, 0)
        .store("scale_old", "output_scale_old", 0, 0)
        .build())


def create_flash_attn_output_update(B=FLASH_BLOCK_SIZE, d=HEAD_DIM, dtype=DTYPE):
    """
    InCore: Update output accumulator for Flash Attention.
    
    O_new = scale_old * O_prev + P @ V_block
    
    Memory: O(B*d) + P(B*B) + V(B*d) + PV(B*d) + scale(B) = 3*B*d + B*B + B
            = 3*64*128 + 64*64 + 64 = 24576 + 4096 + 64 = 28736 bytes ≈ 28KB
    """
    return (PTOFunctionBuilder("flash_attn_output_update")
        .in_core()
        # Input tiles
        .tile("o_prev", B, d, dtype)           # Previous output: B x d
        .tile("p_block", B, B, dtype)          # Softmax weights: B x B
        .tile("v_block", B, d, dtype)          # Value block: B x d
        .tile("scale_old", B, 1, dtype)        # Scaling factor: B x 1
        
        # Intermediate/output tiles
        .tile("o_scaled", B, d, dtype)         # Scaled previous output
        .tile("pv", B, d, dtype)               # P @ V
        .tile("o_new", B, d, dtype)            # New output
        
        .memref("input_o_prev", MemorySpace.GM, dtype)
        .memref("input_p", MemorySpace.GM, dtype)
        .memref("input_v", MemorySpace.GM, dtype)
        .memref("input_scale", MemorySpace.GM, dtype)
        .memref("output_o", MemorySpace.GM, dtype)
        
        .load("o_prev", "input_o_prev", 0, 0)
        .load("p_block", "input_p", 0, 0)
        .load("v_block", "input_v", 0, 0)
        .load("scale_old", "input_scale", 0, 0)
        
        # o_scaled = scale_old * O_prev (broadcast multiply)
        .rowexpandmul("o_scaled", "o_prev", "scale_old")
        
        # pv = P @ V
        .matmul("pv", "p_block", "v_block")
        
        # o_new = o_scaled + pv
        .add("o_new", "o_scaled", "pv")
        
        .store("o_new", "output_o", 0, 0)
        .build())


def create_flash_attn_normalize(B=FLASH_BLOCK_SIZE, d=HEAD_DIM, dtype=DTYPE):
    """
    InCore: Final normalization for Flash Attention output.
    
    O_final = O / l  (row-wise division)
    
    Memory: O(B*d) + l(B) + O_final(B*d) = 2*B*d + B = 2*64*128 + 64 = 16448 bytes ≈ 16KB
    """
    return (PTOFunctionBuilder("flash_attn_normalize")
        .in_core()
        .tile("o_block", B, d, dtype)          # Output accumulator: B x d
        .tile("l_vec", B, 1, dtype)            # Row sums: B x 1
        .tile("o_final", B, d, dtype)          # Normalized output: B x d
        
        .memref("input_o", MemorySpace.GM, dtype)
        .memref("input_l", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("o_block", "input_o", 0, 0)
        .load("l_vec", "input_l", 0, 0)
        
        # O_final = O / l (broadcast division)
        .rowexpanddiv("o_final", "o_block", "l_vec")
        
        .store("o_final", "output", 0, 0)
        .build())


def create_flash_attn_init_state(B=FLASH_BLOCK_SIZE, d=HEAD_DIM, dtype=DTYPE):
    """
    InCore: Initialize Flash Attention state (O=0, l=0, m=-inf).
    
    Note: We initialize by loading from pre-allocated zero/neg_inf buffers.
    In practice, the host would prepare these constant buffers.
    
    Memory: O(B*d) + l(B) + m(B) = B*d + 2*B = 64*128 + 128 = 8320 bytes ≈ 8KB
    """
    return (PTOFunctionBuilder("flash_attn_init_state")
        .in_core()
        .tile("o_init", B, d, dtype)           # O = 0
        .tile("l_init", B, 1, dtype)           # l = 0
        .tile("m_init", B, 1, dtype)           # m = -inf (use very negative number)
        
        # Input: pre-allocated constant buffers (zeros and neg_inf)
        .memref("input_zeros_large", MemorySpace.GM, dtype)  # Pre-filled with 0
        .memref("input_zeros_small", MemorySpace.GM, dtype)  # Pre-filled with 0
        .memref("input_neg_inf", MemorySpace.GM, dtype)      # Pre-filled with -1e9
        
        .memref("output_o", MemorySpace.GM, dtype)
        .memref("output_l", MemorySpace.GM, dtype)
        .memref("output_m", MemorySpace.GM, dtype)
        
        # Load constants from pre-allocated buffers
        .load("o_init", "input_zeros_large", 0, 0)
        .load("l_init", "input_zeros_small", 0, 0)
        .load("m_init", "input_neg_inf", 0, 0)
        
        .store("o_init", "output_o", 0, 0)
        .store("l_init", "output_l", 0, 0)
        .store("m_init", "output_m", 0, 0)
        .build())


# =============================================================================
# Level 3: Orchestration Functions (Dynamic Sequence Processing)
# =============================================================================

def create_llama_attention_dynamic(module, rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    Orchestration: Self-attention with dynamic sequence length.
    
    For each position in the sequence:
    1. Apply RMSNorm
    2. Compute Q, K, V projections  
    3. Apply RoPE to Q, K
    4. Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
    5. Project output
    
    The sequence dimension is handled by the FOR loop.
    """
    return (PTOFunctionBuilder("llama_attention_dynamic", module=module)
        .not_in_core()  # Orchestration function
        
        # Memory references
        .memref("hidden_states", MemorySpace.GM, dtype)      # Input [B, seq, hidden]
        .memref("attention_output", MemorySpace.GM, dtype)   # Output
        .memref("wq", MemorySpace.GM, dtype)                 # Q projection weights
        .memref("wk", MemorySpace.GM, dtype)                 # K projection weights
        .memref("wv", MemorySpace.GM, dtype)                 # V projection weights
        .memref("wo", MemorySpace.GM, dtype)                 # Output projection weights
        .memref("norm_weights", MemorySpace.GM, dtype)       # RMSNorm weights
        .memref("cos_cache", MemorySpace.GM, dtype)          # RoPE cos cache
        .memref("sin_cache", MemorySpace.GM, dtype)          # RoPE sin cache
        
        # Temporary buffers
        .memref("temp_norm", MemorySpace.GM, dtype)
        .memref("temp_q", MemorySpace.GM, dtype)
        .memref("temp_k", MemorySpace.GM, dtype)
        .memref("temp_v", MemorySpace.GM, dtype)
        .memref("temp_q_rope", MemorySpace.GM, dtype)
        .memref("temp_k_rope", MemorySpace.GM, dtype)
        .memref("temp_scores", MemorySpace.GM, dtype)
        .memref("temp_attn_weights", MemorySpace.GM, dtype)
        .memref("temp_attn_out", MemorySpace.GM, dtype)
        
        # Scalars for loop control
        .scalar("seq_len", ElementType.I32)
        .scalar("num_tiles", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar("zero", ElementType.I32)
        
        .scalar_li("tile_size", rows)
        .scalar_li("zero", 0)
        
        # Process sequence in tiles (binary expansion for dynamic seq_len up to 128K)
        .for_loop("seq_idx", 0, "num_tiles", 1, max_range=MAX_NUM_TILES, min_range=MIN_NUM_TILES)
            # Step 1: RMSNorm
            .call("rmsnorm_tile", {
                "input": "hidden_states",
                "weights": "norm_weights",
                "output": "temp_norm"
            })
            
            # Step 2: Q, K, V projections (using tile_matmul for visibility)
            # Q = norm @ Wq
            .call("tile_matmul", {
                "input_a": "temp_norm",
                "input_b": "wq",
                "output": "temp_q"
            })
            # K = norm @ Wk
            .call("tile_matmul", {
                "input_a": "temp_norm",
                "input_b": "wk",
                "output": "temp_k"
            })
            # V = norm @ Wv
            .call("tile_matmul", {
                "input_a": "temp_norm",
                "input_b": "wv",
                "output": "temp_v"
            })
            
            # Step 3: Apply RoPE to Q and K
            .call("rope_tile", {
                "input": "temp_q",
                "cos_cache": "cos_cache",
                "sin_cache": "sin_cache",
                "output": "temp_q_rope"
            })
            .call("rope_tile", {
                "input": "temp_k",
                "cos_cache": "cos_cache",
                "sin_cache": "sin_cache",
                "output": "temp_k_rope"
            })
            
            # ============================================================
            # Flash Attention Block (replaces standard attention)
            # ============================================================
            # Flash Attention processes attention in blocks to fit SRAM
            # For each Q block:
            #   For each K,V block:
            #     1. Compute S = Q @ K^T / sqrt(d)
            #     2. Online softmax update (m, l, P)
            #     3. Accumulate O = scale*O + P @ V
            #   Normalize: O = O / l
            
            # Step 4a: Initialize Flash Attention state (O=0, l=0, m=-inf)
            .call("flash_attn_init_state", {
                "input_zeros_large": "temp_zeros_large",
                "input_zeros_small": "temp_zeros_small",
                "input_neg_inf": "temp_neg_inf",
                "output_o": "temp_attn_out",
                "output_l": "temp_l_vec",
                "output_m": "temp_m_vec"
            })
            
            # Step 4b: Compute attention scores (Q @ K^T / sqrt(d))
            .call("flash_attn_score_block", {
                "input_q": "temp_q_rope",
                "input_k": "temp_k_rope",
                "output_s": "temp_scores"
            })
            
            # Step 4c: Online softmax update
            .call("flash_attn_softmax_update", {
                "input_s": "temp_scores",
                "input_m_prev": "temp_m_vec",
                "input_l_prev": "temp_l_vec",
                "output_m_new": "temp_m_vec",
                "output_l_new": "temp_l_vec",
                "output_p": "temp_attn_weights",
                "output_scale_old": "temp_scale"
            })
            
            # Step 4d: Update output accumulator (O = scale*O + P @ V)
            .call("flash_attn_output_update", {
                "input_o_prev": "temp_attn_out",
                "input_p": "temp_attn_weights",
                "input_v": "temp_v",
                "input_scale": "temp_scale",
                "output_o": "temp_attn_out"
            })
            
            # Step 4e: Final normalization (O = O / l)
            .call("flash_attn_normalize", {
                "input_o": "temp_attn_out",
                "input_l": "temp_l_vec",
                "output": "temp_attn_out"
            })
            
            # Step 7: Output projection (matmul: attn_out @ Wo)
            # This is the aggregation of multi-head outputs
            .call("tile_matmul", {
                "input_a": "temp_attn_out",
                "input_b": "wo",
                "output": "attention_output"
            })
        .end_for()
        
        .build())


def create_llama_mlp_dynamic(module, rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    Orchestration: MLP (Feed-Forward) with dynamic sequence length.
    
    LLaMA uses SwiGLU activation:
    1. gate = input @ W_gate
    2. up = input @ W_up
    3. hidden = SiLU(gate) * up
    4. output = hidden @ W_down
    """
    return (PTOFunctionBuilder("llama_mlp_dynamic", module=module)
        .not_in_core()
        
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .memref("w_gate", MemorySpace.GM, dtype)
        .memref("w_up", MemorySpace.GM, dtype)
        .memref("w_down", MemorySpace.GM, dtype)
        .memref("norm_weights", MemorySpace.GM, dtype)
        
        .memref("temp_norm", MemorySpace.GM, dtype)
        .memref("temp_gate", MemorySpace.GM, dtype)
        .memref("temp_up", MemorySpace.GM, dtype)
        .memref("temp_swiglu", MemorySpace.GM, dtype)
        
        .scalar("seq_len", ElementType.I32)
        .scalar("num_tiles", ElementType.I32)
        .scalar("zero", ElementType.I32)
        
        .scalar_li("zero", 0)
        
        # Binary expansion for dynamic seq_len up to 128K
        .for_loop("seq_idx", 0, "num_tiles", 1, max_range=MAX_NUM_TILES, min_range=MIN_NUM_TILES)
            # RMSNorm
            .call("rmsnorm_tile", {
                "input": "input",
                "weights": "norm_weights",
                "output": "temp_norm"
            })
            
            # Gate and up projections (parallel matmuls)
            .call("tile_matmul", {
                "input_a": "temp_norm",
                "input_b": "w_gate",
                "output": "temp_gate"
            })
            .call("tile_matmul", {
                "input_a": "temp_norm",
                "input_b": "w_up",
                "output": "temp_up"
            })
            
            # SwiGLU
            .call("swiglu_tile", {
                "input_gate": "temp_gate",
                "input_up": "temp_up",
                "output": "temp_swiglu"
            })
            
            # Down projection (aggregation matmul)
            .call("tile_matmul", {
                "input_a": "temp_swiglu",
                "input_b": "w_down",
                "output": "output"
            })
        .end_for()
        
        .build())


def create_llama_layer_dynamic(module, rows=TILE_ROWS, cols=TILE_COLS, dtype=DTYPE):
    """
    Orchestration: Complete LLaMA decoder layer with CORRECT PHASED EXECUTION.
    
    CORRECT Architecture with Cross-Tile Dependencies:
    ===================================================
    The computation is divided into 3 phases with proper dependencies:
    
    Phase 1: Pre-Attention (ALL tiles run in PARALLEL)
    --------------------------------------------------
    For each tile i in [0, num_tiles):
      - RMSNorm[i] on input[i]
      - Q[i] = MatMul(norm[i], Wq)
      - K[i] = MatMul(norm[i], Wk)  
      - V[i] = MatMul(norm[i], Wv)
      - Q_rope[i] = RoPE(Q[i])
      - K_rope[i] = RoPE(K[i])
    
    Phase 2: Flash Attention (CROSS-TILE dependencies)
    --------------------------------------------------
    For each Q tile i:
      Initialize O[i]=0, L[i]=0, M[i]=-inf
      For each KV tile j in [0, num_tiles):  ← Q[i] attends to ALL K,V tiles!
        S[i,j] = Q_rope[i] @ K_rope[j].T / sqrt(d)
        Update O[i], L[i], M[i] with P[i,j] and V[j]
      Normalize O[i] = O[i] / L[i]
    
    >>> This creates the "fan-in" dependency pattern <<<
    >>> Each Q[i]'s attention depends on ALL K[j], V[j] <<<
    
    Phase 3: Post-Attention (depends on Phase 2 completion)
    -------------------------------------------------------
    For each tile i:
      - O_proj[i] = MatMul(attn_out[i], Wo)  ← depends on attention[i] complete
      - Residual: hidden[i] = O_proj[i] + input[i]
      - RMSNorm[i] on hidden[i]
      - Gate[i] = MatMul(norm[i], W_gate)
      - Up[i] = MatMul(norm[i], W_up)
      - SwiGLU: swiglu[i] = SiLU(Gate[i]) * Up[i]
      - Down[i] = MatMul(swiglu[i], W_down)
      - Residual: output[i] = Down[i] + hidden[i]
    
    This creates a proper dependency graph where:
    - Phase 1: All tiles are independent (massive parallelism)
    - Phase 2: Each Q tile depends on ALL K/V tiles (fan-in pattern)
    - Phase 3: Each tile depends only on its own attention (parallelism resumes)
    """
    return (PTOFunctionBuilder("llama_layer_dynamic", module=module)
        .not_in_core()
        
        # Main I/O
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        # Attention weights
        .memref("attn_norm_weights", MemorySpace.GM, dtype)
        .memref("wq", MemorySpace.GM, dtype)
        .memref("wk", MemorySpace.GM, dtype)
        .memref("wv", MemorySpace.GM, dtype)
        .memref("wo", MemorySpace.GM, dtype)
        .memref("cos_cache", MemorySpace.GM, dtype)
        .memref("sin_cache", MemorySpace.GM, dtype)
        
        # MLP weights
        .memref("mlp_norm_weights", MemorySpace.GM, dtype)
        .memref("w_gate", MemorySpace.GM, dtype)
        .memref("w_up", MemorySpace.GM, dtype)
        .memref("w_down", MemorySpace.GM, dtype)
        
        # ================================================================
        # Per-Tile Arrays (indexed by tile number)
        # These buffers hold intermediate results for ALL tiles
        # ================================================================
        # Phase 1 outputs (all tiles)
        .memref("all_q_tiles", MemorySpace.GM, dtype)       # [num_tiles, rows, hidden]
        .memref("all_k_tiles", MemorySpace.GM, dtype)       # [num_tiles, rows, hidden]
        .memref("all_v_tiles", MemorySpace.GM, dtype)       # [num_tiles, rows, hidden]
        .memref("all_q_rope", MemorySpace.GM, dtype)        # [num_tiles, rows, hidden]
        .memref("all_k_rope", MemorySpace.GM, dtype)        # [num_tiles, rows, hidden]
        
        # Phase 2 outputs (attention output per Q tile)
        .memref("all_attn_out", MemorySpace.GM, dtype)      # [num_tiles, rows, hidden]
        .memref("all_m_vec", MemorySpace.GM, dtype)         # [num_tiles, rows, 1]
        .memref("all_l_vec", MemorySpace.GM, dtype)         # [num_tiles, rows, 1]
        
        # Phase 3 intermediates (reusable per-tile buffers)
        .memref("all_hidden", MemorySpace.GM, dtype)        # [num_tiles, rows, hidden]
        
        # Temporary working buffers (reusable)
        .memref("temp_norm", MemorySpace.GM, dtype)
        .memref("temp_scores", MemorySpace.GM, dtype)
        .memref("temp_attn_weights", MemorySpace.GM, dtype)
        .memref("temp_scale", MemorySpace.GM, dtype)
        .memref("temp_gate", MemorySpace.GM, dtype)
        .memref("temp_up", MemorySpace.GM, dtype)
        .memref("temp_swiglu", MemorySpace.GM, dtype)
        .memref("temp_mlp_out", MemorySpace.GM, dtype)
        
        # Flash Attention initialization constants
        .memref("const_zeros_large", MemorySpace.GM, dtype)  # Pre-filled with 0
        .memref("const_zeros_small", MemorySpace.GM, dtype)  # Pre-filled with 0
        .memref("const_neg_inf", MemorySpace.GM, dtype)      # Pre-filled with -inf
        
        # ================================================================
        # Dynamic Tiling Control Variables
        # ================================================================
        .scalar("seq_len", ElementType.I32)
        .scalar("tile_rows", ElementType.I32)
        .scalar("num_tiles", ElementType.I32)
        .scalar("zero", ElementType.I32)
        
        .scalar_li("tile_rows", rows)
        .scalar_li("zero", 0)
        
        # ================================================================
        # PHASE 1: Pre-Attention (ALL tiles run in PARALLEL)
        # ================================================================
        # Each tile computes Q/K/V independently - no cross-tile dependencies
        # OFFSET: ("tensor", "tile_i", 0) - each tile processes different rows
        # Binary expansion for dynamic seq_len up to 128K
        # tile_levels: larger blocks use larger tiles for better throughput
        .for_loop("tile_i", 0, "num_tiles", 1, 
                  max_range=MAX_NUM_TILES, min_range=MIN_NUM_TILES,
                  tile_levels=TILE_ROWS_BY_LEVEL)
            # ============================================================
            # Phase 1: Pre-Attention for tile_i (independent of other tiles)
            # ============================================================
            
            # RMSNorm on input[tile_i]
            .call("rmsnorm_tile", {
                "input": ("input", "tile_i", 0),           # input[tile_i]
                "weights": "attn_norm_weights",            # Shared weights
                "output": ("temp_norm", "tile_i", 0)       # temp_norm[tile_i]
            })
            
            # Q, K, V projections (3 matmuls, all independent)
            .call("tile_matmul", {
                "input_a": ("temp_norm", "tile_i", 0),
                "input_b": "wq",                           # Shared weights
                "output": ("all_q_tiles", "tile_i", 0)     # Q[tile_i]
            })
            .call("tile_matmul", {
                "input_a": ("temp_norm", "tile_i", 0),
                "input_b": "wk",
                "output": ("all_k_tiles", "tile_i", 0)     # K[tile_i]
            })
            .call("tile_matmul", {
                "input_a": ("temp_norm", "tile_i", 0),
                "input_b": "wv",
                "output": ("all_v_tiles", "tile_i", 0)     # V[tile_i]
            })
            
            # RoPE on Q and K (independent)
            .call("rope_tile", {
                "input": ("all_q_tiles", "tile_i", 0),     # Q[tile_i]
                "cos_cache": "cos_cache",
                "sin_cache": "sin_cache",
                "output": ("all_q_rope", "tile_i", 0)      # Q_rope[tile_i]
            })
            .call("rope_tile", {
                "input": ("all_k_tiles", "tile_i", 0),     # K[tile_i]
                "cos_cache": "cos_cache",
                "sin_cache": "sin_cache",
                "output": ("all_k_rope", "tile_i", 0)      # K_rope[tile_i]
            })
        .end_for()  # End Phase 1
        
        # ================================================================
        # PHASE 2: Flash Attention (CROSS-TILE dependencies)
        # ================================================================
        # For each Q tile, we must attend to ALL K,V tiles
        # This creates the "fan-in" dependency: Q[i] depends on all K[j], V[j]
        # OFFSET: Q uses "q_tile", K/V uses "kv_tile" - creates N*N cross-tile deps
        # Binary expansion for dynamic seq_len up to 128K
        .for_loop("q_tile", 0, "num_tiles", 1, 
                  max_range=MAX_NUM_TILES, min_range=MIN_NUM_TILES,
                  tile_levels=TILE_ROWS_BY_LEVEL)
            # Initialize attention state for this Q tile
            .call("flash_attn_init_state", {
                "input_zeros_large": "const_zeros_large",
                "input_zeros_small": "const_zeros_small",
                "input_neg_inf": "const_neg_inf",
                "output_o": ("all_attn_out", "q_tile", 0),   # O[q_tile]
                "output_l": ("all_l_vec", "q_tile", 0),      # L[q_tile]
                "output_m": ("all_m_vec", "q_tile", 0)       # M[q_tile]
            })
            
            # ============================================================
            # Inner Loop: Q[q_tile] attends to ALL K,V tiles
            # This creates cross-tile dependencies!
            # Binary expansion for dynamic seq_len up to 128K
            # ============================================================
            .for_loop("kv_tile", 0, "num_tiles", 1, 
                      max_range=MAX_NUM_TILES, min_range=MIN_NUM_TILES,
                      tile_levels=TILE_ROWS_BY_LEVEL)
                # Compute S[q_tile, kv_tile] = Q_rope[q_tile] @ K_rope[kv_tile].T
                .call("flash_attn_score_block", {
                    "input_q": ("all_q_rope", "q_tile", 0),  # Q_rope[q_tile]
                    "input_k": ("all_k_rope", "kv_tile", 0), # K_rope[kv_tile] ← CROSS-TILE!
                    "output_s": ("temp_scores", "q_tile", 0)
                })
                
                # Online softmax update
                .call("flash_attn_softmax_update", {
                    "input_s": ("temp_scores", "q_tile", 0),
                    "input_m_prev": ("all_m_vec", "q_tile", 0),
                    "input_l_prev": ("all_l_vec", "q_tile", 0),
                    "output_m_new": ("all_m_vec", "q_tile", 0),
                    "output_l_new": ("all_l_vec", "q_tile", 0),
                    "output_p": ("temp_attn_weights", "q_tile", 0),
                    "output_scale_old": ("temp_scale", "q_tile", 0)
                })
                
                # Accumulate: O[q_tile] += scale * P @ V[kv_tile]
                .call("flash_attn_output_update", {
                    "input_o_prev": ("all_attn_out", "q_tile", 0),
                    "input_p": ("temp_attn_weights", "q_tile", 0),
                    "input_v": ("all_v_tiles", "kv_tile", 0),  # V[kv_tile] ← CROSS-TILE!
                    "input_scale": ("temp_scale", "q_tile", 0),
                    "output_o": ("all_attn_out", "q_tile", 0)
                })
            .end_for()  # End kv_tile loop
            
            # Normalize: O[q_tile] = O[q_tile] / L[q_tile]
            .call("flash_attn_normalize", {
                "input_o": ("all_attn_out", "q_tile", 0),
                "input_l": ("all_l_vec", "q_tile", 0),
                "output": ("all_attn_out", "q_tile", 0)
            })
        .end_for()  # End q_tile loop (Phase 2)
        
        # ================================================================
        # PHASE 3: Post-Attention (depends on Phase 2 completion)
        # ================================================================
        # Each tile can now process independently, but only after
        # attention for that tile is complete
        # OFFSET: ("tensor", "tile_i", 0) - each tile processes different rows
        # Binary expansion for dynamic seq_len up to 128K
        .for_loop("tile_i", 0, "num_tiles", 1, 
                  max_range=MAX_NUM_TILES, min_range=MIN_NUM_TILES,
                  tile_levels=TILE_ROWS_BY_LEVEL)
            # Output projection (aggregates multi-head outputs)
            .call("tile_matmul", {
                "input_a": ("all_attn_out", "tile_i", 0),   # attn_out[tile_i]
                "input_b": "wo",                            # Shared weights
                "output": ("temp_norm", "tile_i", 0)        # temp for O_proj
            })
            
            # Residual connection: hidden = O_proj + input
            .call("residual_add_tile", {
                "input": ("temp_norm", "tile_i", 0),
                "input_residual": ("input", "tile_i", 0),
                "output": ("all_hidden", "tile_i", 0)       # hidden[tile_i]
            })
            
            # ============================================================
            # MLP Block for tile_i
            # ============================================================
            
            # Pre-MLP RMSNorm
            .call("rmsnorm_tile", {
                "input": ("all_hidden", "tile_i", 0),       # hidden[tile_i]
                "weights": "mlp_norm_weights",              # Shared weights
                "output": ("temp_norm", "tile_i", 0)
            })
            
            # Gate and Up projections (independent matmuls)
            .call("tile_matmul", {
                "input_a": ("temp_norm", "tile_i", 0),
                "input_b": "w_gate",
                "output": ("temp_gate", "tile_i", 0)
            })
            .call("tile_matmul", {
                "input_a": ("temp_norm", "tile_i", 0),
                "input_b": "w_up",
                "output": ("temp_up", "tile_i", 0)
            })
            
            # SwiGLU activation
            .call("swiglu_tile", {
                "input_gate": ("temp_gate", "tile_i", 0),
                "input_up": ("temp_up", "tile_i", 0),
                "output": ("temp_swiglu", "tile_i", 0)
            })
            
            # Down projection
            .call("tile_matmul", {
                "input_a": ("temp_swiglu", "tile_i", 0),
                "input_b": "w_down",
                "output": ("temp_mlp_out", "tile_i", 0)
            })
            
            # Final residual: output = MLP_out + hidden
            .call("residual_add_tile", {
                "input": ("temp_mlp_out", "tile_i", 0),
                "input_residual": ("all_hidden", "tile_i", 0),
                "output": ("output", "tile_i", 0)           # output[tile_i]
            })
        .end_for()  # End Phase 3
        
        .build())


# =============================================================================
# Module Creation
# =============================================================================

def create_llama7b_module():
    """
    Create the complete LLaMA 7B layer module with Flash Attention.
    
    For each power-of-2 binary expansion level, we create InCore function variants
    with different tile sizes:
    - Large blocks (4096, 2048): 64-row tiles for max throughput
    - Medium blocks (1024, 512, 256): 32-row tiles (standard)
    
    Returns:
        PTOModule with all InCore and Orchestration functions
    """
    module = PTOModule("llama7b_flash")
    
    # Level 1: Basic tile operations (InCore) - Create variants for each tile size
    print("Adding Level 1 InCore functions (basic ops)...")
    print(f"  Creating variants for tile sizes: {TILE_SIZE_VARIANTS}")
    
    for tile_rows in TILE_SIZE_VARIANTS:
        suffix = f" ({tile_rows} rows)" if tile_rows != TILE_ROWS else " (standard)"
        print(f"    - Tile size: {tile_rows}x{TILE_COLS}{suffix}")
        
        module.add_function(create_tile_add(rows=tile_rows))
        module.add_function(create_tile_mul(rows=tile_rows))
        module.add_function(create_tile_muls(rows=tile_rows))
        module.add_function(create_tile_exp(rows=tile_rows))
        module.add_function(create_tile_silu(rows=tile_rows))
        module.add_function(create_tile_rsqrt(rows=tile_rows))
        module.add_function(create_tile_matmul(m=tile_rows))
        module.add_function(create_tile_rowmax(rows=tile_rows))
        module.add_function(create_tile_rowsum(rows=tile_rows))
        module.add_function(create_tile_rowexpandsub(rows=tile_rows))
        module.add_function(create_tile_rowexpanddiv(rows=tile_rows))
        module.add_function(create_tile_rowexpandmul(rows=tile_rows))
    
    # Level 2: Composed operations (InCore) - Create variants for each tile size
    print("Adding Level 2 InCore functions (composed ops)...")
    for tile_rows in TILE_SIZE_VARIANTS:
        module.add_function(create_rmsnorm_tile(rows=tile_rows))
        module.add_function(create_softmax_tile(rows=tile_rows))
        module.add_function(create_swiglu_tile(rows=tile_rows))
        module.add_function(create_linear_tile(rows=tile_rows))
        module.add_function(create_rope_tile(rows=tile_rows))
        module.add_function(create_attention_score_tile(rows=tile_rows))
        module.add_function(create_attention_output_tile(rows=tile_rows))
        module.add_function(create_residual_add_tile(rows=tile_rows))
    
    # Flash Attention InCore functions (fits in 256KB SRAM)
    print("Adding Flash Attention InCore functions...")
    module.add_function(create_flash_attn_score_block())
    module.add_function(create_flash_attn_softmax_update())
    module.add_function(create_flash_attn_output_update())
    module.add_function(create_flash_attn_normalize())
    module.add_function(create_flash_attn_init_state())
    
    # Level 3: Orchestration functions (dynamic sequence)
    print("Adding Level 3 Orchestration functions (dynamic loops)...")
    module.add_function(create_llama_layer_dynamic(module))
    
    # Set entry point
    module.set_entry("llama_layer_dynamic")
    
    return module


# =============================================================================
# Code Generation with Specific Sequence Length
# =============================================================================

def generate_for_seq_len(module, seq_len: int, output_base: str, incore_funcs: list, max_tiles_for_demo: int = None):
    """
    Generate task graph for a specific sequence length.
    
    Args:
        module: PTOModule with all functions
        seq_len: Sequence length to process
        output_base: Base output directory
        incore_funcs: List of InCore function names
        max_tiles_for_demo: Maximum number of tiles to generate for demo (None = no limit)
    """
    # Calculate tiling parameters
    actual_num_full_tiles = seq_len // TILE_ROWS
    actual_tail_rows = seq_len % TILE_ROWS
    actual_has_tail = actual_tail_rows > 0
    
    # For demo purposes, optionally limit the number of tiles to avoid memory issues
    if max_tiles_for_demo is not None:
        num_full_tiles = min(actual_num_full_tiles, max_tiles_for_demo)
    else:
        num_full_tiles = actual_num_full_tiles
    tail_rows = actual_tail_rows if num_full_tiles == actual_num_full_tiles else 0
    has_tail = tail_rows > 0
    
    # Calculate expected task count
    # Each tile iteration: 17 tasks (for LLaMA layer)
    # Main loop: num_full_tiles * 17 tasks
    # Tail (if exists): 17 tasks
    tasks_per_tile = 17  # Attention(11) + MLP(6)
    actual_total_tasks = actual_num_full_tiles * tasks_per_tile + (tasks_per_tile if actual_has_tail else 0)
    demo_tasks = num_full_tiles * tasks_per_tile + (tasks_per_tile if has_tail else 0)
    
    # Use simple folder name
    folder_name = "llama7b"
    
    print(f"\n{'='*70}")
    print(f"Generating for seq_len={seq_len}")
    print(f"{'='*70}")
    print(f"  Actual: {actual_num_full_tiles} full tiles + {actual_tail_rows} tail = {actual_total_tasks} tasks")
    print(f"  Demo:   {num_full_tiles} full tiles + {tail_rows} tail = {demo_tasks} tasks")
    if num_full_tiles < actual_num_full_tiles:
        print(f"  (Limited to {max_tiles_for_demo} tiles for demo)")
    print(f"  output_folder: {folder_name}")
    
    # Create output directories for all backends
    arm64_dir = os.path.join(output_base, "output_arm64", folder_name)
    cuda_dir = os.path.join(output_base, "output_cuda", folder_name)
    ascend_dir = os.path.join(output_base, "output_ascend910b", folder_name)
    pto_dir = os.path.join(output_base, "output_pto", folder_name)
    
    os.makedirs(arm64_dir, exist_ok=True)
    os.makedirs(cuda_dir, exist_ok=True)
    os.makedirs(ascend_dir, exist_ok=True)
    os.makedirs(pto_dir, exist_ok=True)
    
    # Compile to PTO Assembly
    compiler = PTOModuleCompiler(
        inline_in_core=False,
        eliminate_redundant_mem=False
    )
    pto_code = compiler.compile(module)
    
    # Add seq_len info to PTO header
    pto_header = f"""// PTO LLaMA 7B Layer - Dynamic Tiling
// Sequence Length: {seq_len}
// Tile Size: {TILE_ROWS}x{TILE_COLS}
// num_full_tiles: {num_full_tiles}
// tail_rows: {tail_rows}

"""
    pto_code = pto_header + pto_code
    
    pto_file = os.path.join(pto_dir, "llama7b_layer.pto")
    with open(pto_file, "w") as f:
        f.write(pto_code)
    print(f"  [PTO] -> {pto_file}")
    
    # Generate code for all backends
    # Pass module so buffer analysis results are stored for orchestration to use
    gen = MultiBackendCodeGenerator(enable_fusion=True, module=module)
    
    # =========================================================================
    # PRIORITY: Ascend 910B - Generate InCore functions + Orchestration FIRST
    # =========================================================================
    print(f"\n  --- Ascend 910B (Priority) ---")
    for name in incore_funcs:
        func = module.get_function(name)
        ascend_code = gen.generate_ascend(func)
        func_file = os.path.join(ascend_dir, f"{name}.cpp")
        with open(func_file, "w") as f:
            f.write(ascend_code)
    print(f"  [Ascend] {len(incore_funcs)} InCore functions generated")
    
    # =========================================================================
    # Generate UNIFIED Orchestration Code (Platform Independent!)
    # =========================================================================
    # Orchestration function 只调用 PTO runtime 的 task 接口:
    #   - pto_task_alloc, pto_task_add_input, pto_task_add_output, pto_task_submit
    # 这些接口是平台无关的纯 C 代码
    # ARM64, CUDA, Ascend 后端共享同一份 orchestration 代码
    #
    # 使用 CALL 指令自动生成 task scheduling 代码
    # 偏移信息已经在 .call() 中传递: ("tensor", "tile_i", 0)
    #
    # IMPORTANT: Pass actual runtime parameters so binary expansion works correctly!
    # num_tiles controls how many tiles the orchestration processes
    # The binary expansion will automatically apply adaptive tile sizes
    runtime_args = {
        "seq_len": seq_len,
        "num_tiles": num_full_tiles  # This is limited by max_tiles_for_demo
    }
    dump_file = gen.compile_and_run_orchestration(
        module.get_function(module.entry_function),
        ascend_dir,
        extra_args=runtime_args
    )
    
    # Generate visualization for Ascend (PRIMARY)
    # Skip PDF generation if SKIP_PDF=1 environment variable is set (saves ~60+ seconds)
    skip_pdf = os.environ.get('SKIP_PDF', '0') == '1'
    
    if dump_file and os.path.exists(dump_file):
        print(f"  [Ascend TaskGraph] -> {dump_file}")
        
        if skip_pdf:
            print(f"  [Ascend PDF] Skipped (SKIP_PDF=1)")
        else:
            try:
                from visualize_taskgraph import TaskGraphParser, TaskGraphVisualizer
                parser = TaskGraphParser(dump_file)
                parser.parse()
                visualizer = TaskGraphVisualizer(parser)
                pdf_file = dump_file.replace('.txt', '.pdf')
                visualizer.render(pdf_file, format='pdf')
                print(f"  [Ascend PDF] -> {pdf_file}")
            except Exception as e:
                print(f"  [Ascend PDF] Visualization failed: {e}")
    
    # =========================================================================
    # CUDA: InCore functions (kernels) - Orchestration is platform-independent
    # =========================================================================
    print(f"\n  --- CUDA ---")
    for name in incore_funcs:
        func = module.get_function(name)
        cuda_code = gen.generate_cuda(func)
        func_file = os.path.join(cuda_dir, f"{name}.cu")
        with open(func_file, "w") as f:
            f.write(cuda_code)
    print(f"  [CUDA] {len(incore_funcs)} InCore kernels generated")
    
    # Copy platform-independent orchestration code to CUDA directory
    # (Same .c file - orchestration calls PTO runtime, not CUDA APIs directly)
    orch_src = os.path.join(ascend_dir, f"{module.entry_function}_orchestration.c")
    if os.path.exists(orch_src):
        cuda_orch_file = os.path.join(cuda_dir, f"{module.entry_function}_orchestration.c")
        shutil.copy(orch_src, cuda_orch_file)
        print(f"  [CUDA] Orchestration (platform-independent) -> {cuda_orch_file}")
    
    # Copy task graph to CUDA directory (same task graph for all backends)
    if dump_file and os.path.exists(dump_file):
        cuda_dump = os.path.join(cuda_dir, os.path.basename(dump_file))
        shutil.copy(dump_file, cuda_dump)
        pdf_file = dump_file.replace('.txt', '.pdf')
        if os.path.exists(pdf_file):
            cuda_pdf = os.path.join(cuda_dir, os.path.basename(pdf_file))
            shutil.copy(pdf_file, cuda_pdf)
    
    # =========================================================================
    # ARM64: InCore functions - Orchestration is platform-independent
    # =========================================================================
    print(f"\n  --- ARM64 ---")
    for name in incore_funcs:
        func = module.get_function(name)
        arm64_code = gen.generate_arm64(func)
        func_file = os.path.join(arm64_dir, f"{name}.c")
        with open(func_file, "w") as f:
            f.write(arm64_code)
    print(f"  [ARM64] {len(incore_funcs)} InCore functions generated")
    
    # Copy platform-independent orchestration code to ARM64 directory
    # (Same .c file as Ascend/CUDA - orchestration is backend-agnostic)
    orch_src = os.path.join(ascend_dir, f"{module.entry_function}_orchestration.c")
    if os.path.exists(orch_src):
        arm64_orch_file = os.path.join(arm64_dir, f"{module.entry_function}_orchestration.c")
        shutil.copy(orch_src, arm64_orch_file)
        print(f"  [ARM64] Orchestration (platform-independent) -> {arm64_orch_file}")
    
    # Copy task graph to ARM64 directory (same task graph for all backends)
    if dump_file and os.path.exists(dump_file):
        arm64_dump = os.path.join(arm64_dir, os.path.basename(dump_file))
        shutil.copy(dump_file, arm64_dump)
        pdf_file = dump_file.replace('.txt', '.pdf')
        if os.path.exists(pdf_file):
            arm64_pdf = os.path.join(arm64_dir, os.path.basename(pdf_file))
            shutil.copy(pdf_file, arm64_pdf)
    
    return dump_file


# =============================================================================
# NOTE: Platform-Specific Orchestration Functions are REMOVED
# =============================================================================
# 
# Previously, we had _generate_cuda_orchestration() and _generate_ascend_orchestration()
# which generated CUDA/Ascend-specific host code. This was WRONG because:
#
# 1. Orchestration functions should be PLATFORM-INDEPENDENT
# 2. They only call PTO runtime task interfaces:
#    - pto_task_alloc(), pto_task_add_input(), pto_task_add_output(), pto_task_submit()
# 3. These interfaces are pure C code, not CUDA/Ascend APIs
# 4. The PTO runtime handles dispatching tasks to the appropriate backend
#
# CORRECT ARCHITECTURE:
#
#   Orchestration Function (Pure C, Platform-Independent)
#              │
#              ▼
#   PTO Runtime (pto_runtime.c)
#              │
#   ┌──────────┼──────────┐
#   ▼          ▼          ▼
# ARM64     CUDA       Ascend
# InCore    Kernels    Kernels
#
# The same orchestration .c file is used for ALL backends.
# Only the InCore functions differ per backend.
# =============================================================================


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate code for LLaMA 7B layer with Flash Attention."""
    output_base = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("PTO LLaMA 7B Layer - Flash Attention (seq_len=1024)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Batch Size: {BATCH_SIZE} (fixed)")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Num Heads: {NUM_HEADS}")
    print(f"  Head Dim: {HEAD_DIM}")
    print(f"  Intermediate Dim: {INTERMEDIATE_DIM}")
    print(f"  Target ISA: {TARGET_ISA}")
    print(f"  Standard Tile: {TILE_ROWS}x{TILE_COLS} = {TILE_INFO['elements']} elements ({TILE_INFO['bytes']/1024:.1f} KB)")
    
    # Flash Attention configuration
    print(f"\nFlash Attention Configuration:")
    print(f"  SRAM Size: {SRAM_SIZE_KB} KB")
    print(f"  Flash Block Size (Br=Bc): {FLASH_BLOCK_SIZE}")
    
    # Verify Flash Attention fits in SRAM
    flash_info = verify_flash_attention_memory()
    print(f"\nFlash Attention Memory Analysis:")
    print(f"  Q block: {flash_info['block_size']}x{flash_info['head_dim']} = {flash_info['q_block_kb']:.1f} KB")
    print(f"  K block: {flash_info['block_size']}x{flash_info['head_dim']} = {flash_info['k_block_kb']:.1f} KB")
    print(f"  V block: {flash_info['block_size']}x{flash_info['head_dim']} = {flash_info['v_block_kb']:.1f} KB")
    print(f"  S block: {flash_info['block_size']}x{flash_info['block_size']} = {flash_info['s_block_kb']:.1f} KB")
    print(f"  O block: {flash_info['block_size']}x{flash_info['head_dim']} = {flash_info['o_block_kb']:.1f} KB")
    print(f"  Total (with reuse): {flash_info['total_with_reuse_kb']:.1f} KB / {flash_info['sram_kb']} KB")
    print(f"  Fits in SRAM: {'✓ YES' if flash_info['fits'] else '✗ NO'}")
    
    # Create module
    print("\n" + "=" * 70)
    print("Creating LLaMA 7B Module")
    print("=" * 70)
    
    module = create_llama7b_module()
    
    print(f"\nModule: {module.name}")
    print(f"Total functions: {len(module.get_function_names())}")
    
    # Categorize functions
    incore_funcs = []
    orch_funcs = []
    for name in module.get_function_names():
        func = module.get_function(name)
        if func.is_in_core:
            incore_funcs.append(name)
        else:
            orch_funcs.append(name)
    
    print(f"\nInCore functions ({len(incore_funcs)}):")
    for name in incore_funcs:
        print(f"  - {name}")
    
    print(f"\nOrchestration functions ({len(orch_funcs)}):")
    for name in orch_funcs:
        print(f"  - {name}")
    
    print(f"\nEntry: {module.entry_function}")
    
    # =========================================================================
    # Generate for seq_len=16384 (16K) to demonstrate adaptive tile optimization
    # =========================================================================
    # With 16K sequence: num_tiles = 16384/32 = 512 base-tiles
    # Binary: 512 = 0b1000000000 → hits 512-block
    # tile_levels[512] = 64 → scale=2x → actual_iters = 512/2 = 256
    # This gives 50% reduction in iterations!
    #
    # Task count analysis (with adaptive tiles):
    # - Phase 1 (Pre-Attention): 6 tasks × 256 iters = 1,536 tasks
    # - Phase 2 (Flash Attention): ~256² = 65,536 score blocks → with adaptive: 128² = 16,384
    # - Phase 3 (FFN): 6 tasks × 256 iters = 1,536 tasks
    # Total: ~19,456 tasks (vs ~78,848 without optimization = 75% reduction!)
    seq_len = 16384  # 16K sequence length
    
    # Use demo limit of 256 tiles to verify optimization
    # 256 >= min_range, so adaptive tiles kick in:
    # - Binary: 256 = 0b100000000 → hits 256-block
    # - tile_levels[256] = 64 → scale=2x → actual_iters = 256/2 = 128
    # Without optimization: 256 tiles → 256 iterations
    # With optimization: 256 tiles → 128 iterations (50% reduction!)
    generate_for_seq_len(module, seq_len, output_base, incore_funcs, max_tiles_for_demo=256)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Code Generation Complete!")
    print("=" * 70)
    
    num_full_tiles = seq_len // TILE_ROWS
    tail_rows = seq_len % TILE_ROWS
    tasks_per_tile = 17
    total_tasks = num_full_tiles * tasks_per_tile + (tasks_per_tile if tail_rows > 0 else 0)
    print(f"\nGenerated for seq_len={seq_len}:")
    print(f"  {num_full_tiles} full tiles + {tail_rows} tail rows = ~{total_tasks} tasks")
    print(f"\nParallelism within each tile:")
    print(f"  - Q/K/V projections (3 tasks) can run in parallel")
    print(f"  - RoPE(Q) and RoPE(K) (2 tasks) can run in parallel")
    print(f"  - Gate and Up projections (2 tasks) can run in parallel")
    print(f"\nDifferent tiles are INDEPENDENT (process different data regions)")
    
    return module


if __name__ == "__main__":
    main()
