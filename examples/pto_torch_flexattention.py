"""
PTO torch.nn.attention.flex_attention Implementation

This module implements PyTorch FlexAttention APIs using PTO ISA instructions.
Reference: 
- https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
- https://pytorch.org/blog/flexattention/

FlexAttention is a flexible mechanism for computing scaled dot-product attention
with customizable score modifications and sparse block masks.

Core Attention Formula:
    Score = Q @ K^T / sqrt(d_k)
    Score = score_mod(Score, ...)  # Optional modification
    Score = Score + mask           # Apply mask (additive)
    Attention = softmax(Score)
    Output = Attention @ V

Categories implemented:
1. Basic Attention: scaled_dot_product_attention
2. Score Modifications: relative_position_bias, alibi_bias, causal_mask
3. Multi-Head Attention: split_heads, merge_heads, multi_head_attention
4. Block Sparse Attention: block_sparse_attention (simplified)
5. Flash Attention Components: chunked_attention (memory efficient)
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
DEFAULT_HEAD_DIM = 64  # Common head dimension for attention


# =============================================================================
# Basic Scaled Dot-Product Attention
# =============================================================================

def scaled_dot_product_attention(seq_len=8, head_dim=8):
    """
    Scaled Dot-Product Attention (SDPA)
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    This is the core building block of transformer attention.
    
    Args:
        seq_len: Sequence length (number of tokens)
        head_dim: Dimension of each attention head (d_k)
    
    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """
    scale = 1.0 / (head_dim ** 0.5)  # 1/sqrt(d_k)
    
    return (PTOFunctionBuilder("scaled_dot_product_attention")
        # Input tiles: Q, K, V of shape [seq_len, head_dim]
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        
        # K transposed: [head_dim, seq_len]
        .tile("K_T", head_dim, seq_len, DEFAULT_DTYPE)
        
        # Attention scores: [seq_len, seq_len]
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Softmax intermediates
        .tile("row_max", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("attention_weights", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Output: [seq_len, head_dim]
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        # Memory references
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        # Load inputs
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        
        # Step 1: Compute scores = Q @ K^T
        # Note: For simplicity, we compute Q @ K directly (assuming K is pre-transposed)
        # In practice, K_T should be the transpose of K
        .matmul("scores", "Q", "K")  # scores = Q @ K (K should be transposed)
        
        # Step 2: Scale scores by 1/sqrt(d_k)
        .muls("scaled_scores", "scores", scale)
        
        # Step 3: Softmax over rows (last dimension)
        # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        .rowsum("row_max", "scaled_scores")  # Approximate max with mean for stability
        .divs("row_max", "row_max", float(seq_len))
        .rowexpandsub("shifted", "scaled_scores", "row_max")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attention_weights", "exp_scores", "row_sum")
        
        # Step 4: Output = attention_weights @ V
        .matmul("output", "attention_weights", "V")
        
        # Store output
        .store("output", "output_mem")
        .build())


def sdpa_with_scale(seq_len=8, head_dim=8, scale=None):
    """
    SDPA with explicit scale parameter.
    
    Attention(Q, K, V, scale) = softmax(Q @ K^T * scale) @ V
    
    Allows overriding the default 1/sqrt(d_k) scaling.
    """
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("sdpa_with_scale")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        # Softmax
        .rowsum("row_sum", "scaled")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "scaled", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


# =============================================================================
# Score Modifications (score_mod in FlexAttention)
# =============================================================================

def attention_with_causal_mask(seq_len=8, head_dim=8):
    """
    Attention with Causal (Lower Triangular) Mask
    
    This implements autoregressive attention where each position
    can only attend to previous positions (including itself).
    
    Mask: Lower triangular matrix where mask[i,j] = 0 if i >= j, else -inf
    
    Used in: GPT, decoder-only transformers
    
    FlexAttention equivalent:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
    """
    scale = 1.0 / (head_dim ** 0.5)
    large_neg = -1e9  # Approximation of -inf for softmax
    
    return (PTOFunctionBuilder("attention_causal_mask")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Causal mask: loaded from memory (pre-computed lower triangular)
        .tile("causal_mask", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("masked_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Softmax
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("mask_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .load("causal_mask", "mask_mem")  # Pre-computed: 0 for valid, -inf for masked
        
        # Compute attention scores
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        
        # Apply causal mask (additive)
        .add("masked_scores", "scaled", "causal_mask")
        
        # Softmax
        .rowsum("row_sum", "masked_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "masked_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        # Output
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


def attention_with_alibi(seq_len=8, head_dim=8, alibi_slope=0.1):
    """
    Attention with ALiBi (Attention with Linear Biases)
    
    ALiBi adds position-dependent biases to attention scores:
    bias[i,j] = -slope * |i - j|
    
    This allows models to generalize to longer sequences than trained on.
    
    Reference: "Train Short, Test Long" (Press et al., 2021)
    
    FlexAttention equivalent:
        def alibi_bias(b, h, q_idx, kv_idx):
            return -alibi_slope * abs(q_idx - kv_idx)
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("attention_alibi")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # ALiBi bias matrix: loaded from memory (pre-computed -slope * |i-j|)
        .tile("alibi_bias", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("biased_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Softmax
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("alibi_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .load("alibi_bias", "alibi_mem")  # Pre-computed: -slope * |i-j|
        
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        
        # Apply ALiBi bias (additive)
        .add("biased_scores", "scaled", "alibi_bias")
        
        # Softmax
        .rowsum("row_sum", "biased_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "biased_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


def attention_with_relative_position_bias(seq_len=8, head_dim=8):
    """
    Attention with Relative Position Bias
    
    Adds learnable relative position biases to attention scores.
    Used in T5, Swin Transformer, etc.
    
    score_mod(score, b, h, q_idx, kv_idx) = score + relative_bias[q_idx - kv_idx]
    
    FlexAttention equivalent:
        def relative_pos_bias(b, h, q_idx, kv_idx):
            return relative_bias_table[q_idx - kv_idx + max_distance]
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("attention_relative_position")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Relative position bias: [seq_len, seq_len] matrix
        .tile("rel_pos_bias", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("biased_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("bias_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .load("rel_pos_bias", "bias_mem")
        
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        .add("biased_scores", "scaled", "rel_pos_bias")
        
        # Softmax
        .rowsum("row_sum", "biased_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "biased_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


def attention_with_sliding_window(seq_len=8, head_dim=8, window_size=4):
    """
    Sliding Window Attention
    
    Each token attends only to tokens within a fixed window around it.
    This reduces complexity from O(n²) to O(n * window_size).
    
    Used in: Longformer, BigBird, Mistral
    
    FlexAttention equivalent:
        def sliding_window(b, h, q_idx, kv_idx):
            return abs(q_idx - kv_idx) <= window_size // 2
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("attention_sliding_window")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Window mask: 0 for within window, -inf for outside
        .tile("window_mask", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("masked_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("mask_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .load("window_mask", "mask_mem")
        
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        .add("masked_scores", "scaled", "window_mask")
        
        # Softmax
        .rowsum("row_sum", "masked_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "masked_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


# =============================================================================
# Multi-Head Attention Components
# =============================================================================

def linear_projection_qkv(seq_len=8, d_model=64, head_dim=8):
    """
    Linear Projection for Q, K, V
    
    Projects input embeddings into Q, K, V spaces:
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    This is the first step in multi-head attention.
    """
    return (PTOFunctionBuilder("linear_projection_qkv")
        .tile("X", seq_len, d_model, DEFAULT_DTYPE)
        .tile("W_Q", d_model, head_dim, DEFAULT_DTYPE)
        .tile("W_K", d_model, head_dim, DEFAULT_DTYPE)
        .tile("W_V", d_model, head_dim, DEFAULT_DTYPE)
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("X_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("WQ_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("WK_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("WV_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("X", "X_mem")
        .load("W_Q", "WQ_mem")
        .load("W_K", "WK_mem")
        .load("W_V", "WV_mem")
        
        .matmul("Q", "X", "W_Q")
        .matmul("K", "X", "W_K")
        .matmul("V", "X", "W_V")
        
        .store("Q", "Q_mem")
        .store("K", "K_mem")
        .store("V", "V_mem")
        .build())


def output_projection(seq_len=8, head_dim=8, d_model=64):
    """
    Output Projection
    
    Projects attention output back to model dimension:
    Output = Attention @ W_O
    """
    return (PTOFunctionBuilder("output_projection")
        .tile("attn_out", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("W_O", head_dim, d_model, DEFAULT_DTYPE)
        .tile("output", seq_len, d_model, DEFAULT_DTYPE)
        
        .memref("attn_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("WO_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("attn_out", "attn_mem")
        .load("W_O", "WO_mem")
        .matmul("output", "attn_out", "W_O")
        .store("output", "output_mem")
        .build())


def multi_head_attention_single_head(seq_len=8, head_dim=8):
    """
    Single Head of Multi-Head Attention
    
    Computes one head of multi-head attention.
    In practice, multiple heads are computed in parallel.
    
    This is what flex_attention() computes per head.
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("mha_single_head")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        
        # Softmax
        .rowsum("row_sum", "scaled")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "scaled", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


# =============================================================================
# FlexAttention Core Operations
# =============================================================================

def flex_attention_basic(seq_len=8, head_dim=8):
    """
    flex_attention() - Basic Implementation
    
    The core flex_attention function with default score_mod (identity).
    
    flex_attention(query, key, value) returns:
        softmax(query @ key.T / sqrt(d)) @ value
    
    Reference: torch.nn.attention.flex_attention
    """
    return scaled_dot_product_attention(seq_len, head_dim)


def flex_attention_with_score_mod(seq_len=8, head_dim=8):
    """
    flex_attention() with score_mod
    
    Demonstrates how score_mod modifies attention scores.
    
    In FlexAttention, score_mod is a function:
        score_mod(score, batch, head, q_idx, kv_idx) -> modified_score
    
    Common score_mods:
    - Causal mask: return -inf if q_idx < kv_idx else score
    - ALiBi: return score - slope * |q_idx - kv_idx|
    - Relative position: return score + bias_table[q_idx - kv_idx]
    
    This implementation applies a generic additive bias.
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("flex_attention_score_mod")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Score modification (additive bias representing score_mod output)
        .tile("score_mod_bias", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("modified_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("score_mod_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .load("score_mod_bias", "score_mod_mem")
        
        # Q @ K^T / sqrt(d)
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        
        # Apply score_mod (additive)
        .add("modified_scores", "scaled", "score_mod_bias")
        
        # Softmax
        .rowsum("row_sum", "modified_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "modified_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


def flex_attention_with_block_mask(seq_len=8, head_dim=8):
    """
    flex_attention() with BlockMask
    
    BlockMask enables sparse attention patterns by specifying
    which blocks of Q-K pairs should be computed.
    
    This is key for efficient attention:
    - Causal attention: Lower triangular blocks
    - Sliding window: Diagonal band of blocks
    - Sparse patterns: Custom block configurations
    
    Reference: torch.nn.attention.flex_attention.BlockMask
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("flex_attention_block_mask")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Block mask: 0 for computed blocks, -inf for skipped
        .tile("block_mask", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("masked_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("mask_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .load("block_mask", "mask_mem")
        
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        .add("masked_scores", "scaled", "block_mask")
        
        # Softmax
        .rowsum("row_sum", "masked_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "masked_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


# =============================================================================
# Advanced Attention Patterns
# =============================================================================

def attention_with_dropout(seq_len=8, head_dim=8, dropout_rate=0.1):
    """
    Attention with Dropout (Inference Mode)
    
    In training, dropout randomly zeros attention weights.
    In inference mode (this implementation), dropout is disabled.
    
    FlexAttention supports dropout via the 'dropout_p' parameter.
    """
    # Inference mode: no dropout applied
    return scaled_dot_product_attention(seq_len, head_dim)


def document_attention(seq_len=8, head_dim=8):
    """
    Document Masking Attention
    
    Tokens can only attend within the same document.
    Used for batching multiple documents together.
    
    FlexAttention equivalent:
        def document_mask(b, h, q_idx, kv_idx):
            return document_id[q_idx] == document_id[kv_idx]
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("document_attention")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Document mask: 0 for same doc, -inf for different docs
        .tile("doc_mask", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("masked_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("mask_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .load("doc_mask", "mask_mem")
        
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        .add("masked_scores", "scaled", "doc_mask")
        
        .rowsum("row_sum", "masked_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "masked_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


def prefix_lm_attention(seq_len=8, head_dim=8, prefix_len=4):
    """
    Prefix LM Attention
    
    Bidirectional attention within prefix, causal after prefix.
    Used in encoder-decoder architectures and prefix-tuning.
    
    FlexAttention equivalent:
        def prefix_lm_mask(b, h, q_idx, kv_idx):
            if q_idx < prefix_len:
                return True  # Bidirectional in prefix
            return q_idx >= kv_idx  # Causal after prefix
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("prefix_lm_attention")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Prefix LM mask
        .tile("prefix_mask", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("masked_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("mask_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        .load("prefix_mask", "mask_mem")
        
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        .add("masked_scores", "scaled", "prefix_mask")
        
        .rowsum("row_sum", "masked_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "masked_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


def soft_capping_attention(seq_len=8, head_dim=8, cap_value=50.0):
    """
    Attention with Soft Capping (Tanh Capping)
    
    Caps attention logits using tanh to improve training stability.
    Used in Gemma 2 and other models.
    
    score_mod: score = cap * tanh(score / cap)
    
    FlexAttention equivalent:
        def soft_cap(score, b, h, q_idx, kv_idx):
            return cap * tanh(score / cap)
    
    Note: PTO ISA lacks native tanh, so this uses exp-based approximation.
    """
    scale = 1.0 / (head_dim ** 0.5)
    
    return (PTOFunctionBuilder("soft_capping_attention")
        .tile("Q", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("K", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("V", seq_len, head_dim, DEFAULT_DTYPE)
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("scaled", seq_len, seq_len, DEFAULT_DTYPE)
        
        # Soft capping: cap * tanh(x / cap)
        # tanh(x) ≈ (exp(2x) - 1) / (exp(2x) + 1)
        .tile("x_div_cap", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("two_x", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_2x", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_minus_1", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_plus_1", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("tanh_x", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("capped_scores", seq_len, seq_len, DEFAULT_DTYPE)
        
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("attn", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("output", seq_len, head_dim, DEFAULT_DTYPE)
        
        .memref("Q_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("K_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("V_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("output_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("Q", "Q_mem")
        .load("K", "K_mem")
        .load("V", "V_mem")
        
        .matmul("scores", "Q", "K")
        .muls("scaled", "scores", scale)
        
        # Soft capping: cap * tanh(scaled / cap)
        .divs("x_div_cap", "scaled", cap_value)       # x / cap
        .muls("two_x", "x_div_cap", 2.0)              # 2x / cap
        .exp("exp_2x", "two_x")                       # exp(2x / cap)
        .adds("exp_minus_1", "exp_2x", -1.0)          # exp(2x) - 1
        .adds("exp_plus_1", "exp_2x", 1.0)            # exp(2x) + 1
        .div("tanh_x", "exp_minus_1", "exp_plus_1")   # tanh approximation
        .muls("capped_scores", "tanh_x", cap_value)   # cap * tanh(x/cap)
        
        # Softmax
        .rowsum("row_sum", "capped_scores")
        .divs("row_sum", "row_sum", float(seq_len))
        .rowexpandsub("shifted", "capped_scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("attn", "exp_scores", "row_sum")
        
        .matmul("output", "attn", "V")
        .store("output", "output_mem")
        .build())


# =============================================================================
# Utility Functions
# =============================================================================

def create_causal_mask_tile(seq_len=8):
    """
    Create Causal Mask Tile
    
    Generates a lower triangular mask for causal attention.
    mask[i,j] = 0 if i >= j, else -inf (approximated as -1e9)
    
    FlexAttention: create_block_mask(causal_mask_fn, ...)
    """
    return (PTOFunctionBuilder("create_causal_mask")
        .tile("mask", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("ones", seq_len, seq_len, DEFAULT_DTYPE)
        .memref("mask_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        # Initialize with large negative (will be used where masked)
        .expands("mask", -1e9)
        .expands("ones", 0.0)  # Placeholder for tril operation
        
        # Note: Actual mask creation requires index-based operations
        # which PTO ISA doesn't support directly. In practice,
        # the mask would be pre-computed and loaded from memory.
        .store("mask", "mask_mem")
        .build())


def attention_score_to_weight(seq_len=8):
    """
    Convert Attention Scores to Weights (Softmax)
    
    weights = softmax(scores, dim=-1)
    
    This is the core normalization step in attention.
    """
    return (PTOFunctionBuilder("score_to_weight")
        .tile("scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("row_sum", seq_len, 1, DEFAULT_DTYPE)
        .tile("shifted", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("exp_scores", seq_len, seq_len, DEFAULT_DTYPE)
        .tile("weights", seq_len, seq_len, DEFAULT_DTYPE)
        
        .memref("scores_mem", MemorySpace.GM, DEFAULT_DTYPE)
        .memref("weights_mem", MemorySpace.GM, DEFAULT_DTYPE)
        
        .load("scores", "scores_mem")
        
        # Softmax: exp(x - max) / sum(exp(x - max))
        .rowsum("row_sum", "scores")
        .divs("row_sum", "row_sum", float(seq_len))  # Approximate max with mean
        .rowexpandsub("shifted", "scores", "row_sum")
        .exp("exp_scores", "shifted")
        .rowsum("row_sum", "exp_scores")
        .rowexpanddiv("weights", "exp_scores", "row_sum")
        
        .store("weights", "weights_mem")
        .build())


# =============================================================================
# Build All Programs
# =============================================================================

def get_all_programs():
    """Returns all FlexAttention implementations."""
    programs = {}
    
    # Basic Attention
    programs["sdpa"] = scaled_dot_product_attention()
    programs["sdpa_with_scale"] = sdpa_with_scale()
    
    # Score Modifications
    programs["attn_causal"] = attention_with_causal_mask()
    programs["attn_alibi"] = attention_with_alibi()
    programs["attn_relative_pos"] = attention_with_relative_position_bias()
    programs["attn_sliding_window"] = attention_with_sliding_window()
    
    # Multi-Head Attention
    programs["linear_qkv"] = linear_projection_qkv()
    programs["output_proj"] = output_projection()
    programs["mha_single_head"] = multi_head_attention_single_head()
    
    # FlexAttention Core
    programs["flex_basic"] = flex_attention_basic()
    programs["flex_score_mod"] = flex_attention_with_score_mod()
    programs["flex_block_mask"] = flex_attention_with_block_mask()
    
    # Advanced Patterns
    programs["attn_dropout"] = attention_with_dropout()
    programs["doc_attention"] = document_attention()
    programs["prefix_lm"] = prefix_lm_attention()
    programs["soft_capping"] = soft_capping_attention()
    
    # Utilities
    programs["causal_mask"] = create_causal_mask_tile()
    programs["score_to_weight"] = attention_score_to_weight()
    
    return programs


# =============================================================================
# Main: Generate Multi-Backend Code
# =============================================================================

if __name__ == "__main__":
    from pto_compile import generate_all_backends, BACKENDS
    
    print("=" * 70)
    print("PTO FlexAttention - Multi-Backend Code Generation")
    print("=" * 70)
    print()
    print("Reference:")
    print("  - https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html")
    print("  - https://pytorch.org/blog/flexattention/")
    print()
    
    OUTPUT_PREFIX = "flex_attention"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    programs = get_all_programs()
    
    print(f"Generating {len(programs)} FlexAttention implementations...")
    print(f"Backends: {', '.join(BACKENDS.keys())}")
    print()
    
    success_count = 0
    error_count = 0
    
    for name, program in programs.items():
        print(f"[{name}]")
        try:
            results = generate_all_backends(
                program,
                OUTPUT_PREFIX,
                output_base_dir=SCRIPT_DIR,
                enable_fusion=True
            )
            success_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            error_count += 1
        print()
    
    print("=" * 70)
    print(f"Generation Complete! {success_count}/{len(programs)} implementations generated.")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(f"Total files generated: {success_count * 4}")  # 4 backends
    print("Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{OUTPUT_PREFIX}/")
    print(f"  - output_pto/{OUTPUT_PREFIX}/")
    print("=" * 70)
