# FlexAttention 算法深度分析文档

## 1. 概述

本文档详细分析 `examples/pto_torch_flexattention.py` 中实现的 FlexAttention 算法，该实现基于 PyTorch 的 `torch.nn.attention.flex_attention` API，并使用 PTO ISA (Programmable Tensor Operations Instruction Set Architecture) 进行底层指令级编程。

### 1.1 什么是 FlexAttention？

FlexAttention 是 PyTorch 2.5+ 引入的灵活注意力机制框架，它允许用户通过可定制的 `score_mod` 函数和稀疏 `BlockMask` 来实现各种注意力变体，同时保持高效的硬件性能。

**核心公式：**
```
Score = Q @ K^T / sqrt(d_k)
Score = score_mod(Score, ...)  # 可选的分数修改
Score = Score + mask           # 应用掩码（加性）
Attention = softmax(Score)
Output = Attention @ V
```

### 1.2 实现的功能分类

| 类别 | 实现的函数 | 用途 |
|------|-----------|------|
| **基础注意力** | `scaled_dot_product_attention`, `sdpa_with_scale` | Transformer 核心注意力机制 |
| **分数修改** | `attention_with_causal_mask`, `attention_with_alibi`, `attention_with_relative_position_bias`, `attention_with_sliding_window` | 各种位置编码和掩码策略 |
| **多头注意力** | `linear_projection_qkv`, `output_projection`, `multi_head_attention_single_head` | MHA 组件 |
| **FlexAttention 核心** | `flex_attention_basic`, `flex_attention_with_score_mod`, `flex_attention_with_block_mask` | FlexAttention API 实现 |
| **高级模式** | `document_attention`, `prefix_lm_attention`, `soft_capping_attention` | 特殊场景注意力 |
| **工具函数** | `create_causal_mask_tile`, `attention_score_to_weight` | 辅助功能 |

---

## 2. 核心算法详解

### 2.1 Scaled Dot-Product Attention (SDPA)

这是所有 Transformer 注意力的基础，源自论文 "Attention Is All You Need" (Vaswani et al., 2017)。

#### 2.1.1 数学公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

其中：
- $Q$ (Query): 查询矩阵，形状 `[seq_len, head_dim]`
- $K$ (Key): 键矩阵，形状 `[seq_len, head_dim]`
- $V$ (Value): 值矩阵，形状 `[seq_len, head_dim]`
- $d_k$: 每个注意力头的维度 (head_dim)
- $\frac{1}{\sqrt{d_k}}$: 缩放因子，防止点积过大导致 softmax 梯度消失

#### 2.1.2 计算步骤分解

```
步骤 1: scores = Q @ K^T                    [seq_len, seq_len]
步骤 2: scaled_scores = scores * (1/√d_k)   [seq_len, seq_len]
步骤 3: attention_weights = softmax(scaled_scores, dim=-1)  [seq_len, seq_len]
步骤 4: output = attention_weights @ V      [seq_len, head_dim]
```

#### 2.1.3 PTO ISA 指令序列

```pto
// 加载输入
%Q = tload %Q_mem[0, 0]
%K = tload %K_mem[0, 0]
%V = tload %V_mem[0, 0]

// 步骤1: 矩阵乘法计算注意力分数
%scores = tmatmul %Q, %K

// 步骤2: 缩放
%scaled_scores = tmuls %scores, %0.35355339059327373  // 1/√8 ≈ 0.354

// 步骤3: Softmax (数值稳定版本)
//   3a. 计算行均值作为稳定性偏移
%row_max = trowsum %scaled_scores
%row_max = tdivs %row_max, %8.0

//   3b. 减去偏移
%shifted = trowexpandsub %scaled_scores, %row_max

//   3c. 指数运算
%exp_scores = texp %shifted

//   3d. 计算行和
%row_sum = trowsum %exp_scores

//   3e. 归一化
%attention_weights = trowexpanddiv %exp_scores, %row_sum

// 步骤4: 加权求和
%output = tmatmul %attention_weights, %V

// 存储输出
tstore %output, %output_mem[0, 0]
```

#### 2.1.4 关键 PTO 指令说明

| 指令 | 功能 | 数学表示 |
|------|------|----------|
| `tmatmul` | 矩阵乘法 | $C = A \times B$ |
| `tmuls` | 标量乘法 | $C = A \times s$ |
| `trowsum` | 行求和归约 | $c_i = \sum_j A_{ij}$ |
| `trowexpandsub` | 行广播减法 | $C_{ij} = A_{ij} - b_i$ |
| `texp` | 逐元素指数 | $C_{ij} = e^{A_{ij}}$ |
| `trowexpanddiv` | 行广播除法 | $C_{ij} = A_{ij} / b_i$ |

---

### 2.2 Causal Mask (因果掩码)

用于自回归模型（如 GPT），确保每个位置只能看到之前的位置。

#### 2.2.1 数学定义

$$
\text{mask}[i,j] = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{otherwise} \end{cases}
$$

#### 2.2.2 掩码可视化

```
Position:  0   1   2   3   4   5   6   7
    0    [ 0  -∞  -∞  -∞  -∞  -∞  -∞  -∞ ]
    1    [ 0   0  -∞  -∞  -∞  -∞  -∞  -∞ ]
    2    [ 0   0   0  -∞  -∞  -∞  -∞  -∞ ]
    3    [ 0   0   0   0  -∞  -∞  -∞  -∞ ]
    4    [ 0   0   0   0   0  -∞  -∞  -∞ ]
    5    [ 0   0   0   0   0   0  -∞  -∞ ]
    6    [ 0   0   0   0   0   0   0  -∞ ]
    7    [ 0   0   0   0   0   0   0   0 ]
```

#### 2.2.3 FlexAttention 等效定义

```python
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
```

#### 2.2.4 PTO 实现关键步骤

```pto
// 加载预计算的因果掩码
%causal_mask = tload %mask_mem[0, 0]

// 计算原始分数
%scores = tmatmul %Q, %K
%scaled = tmuls %scores, %scale

// 应用因果掩码（加性）
%masked_scores = tadd %scaled, %causal_mask

// 后续 softmax...
```

---

### 2.3 ALiBi (Attention with Linear Biases)

来自论文 "Train Short, Test Long" (Press et al., 2021)，允许模型泛化到比训练时更长的序列。

#### 2.3.1 数学定义

$$
\text{bias}[i,j] = -\text{slope} \times |i - j|
$$

其中 `slope` 是每个注意力头的线性偏置斜率。

#### 2.3.2 偏置矩阵示例 (slope=0.1)

```
Position:  0     1     2     3     4     5     6     7
    0    [ 0.0  -0.1  -0.2  -0.3  -0.4  -0.5  -0.6  -0.7 ]
    1    [-0.1   0.0  -0.1  -0.2  -0.3  -0.4  -0.5  -0.6 ]
    2    [-0.2  -0.1   0.0  -0.1  -0.2  -0.3  -0.4  -0.5 ]
    3    [-0.3  -0.2  -0.1   0.0  -0.1  -0.2  -0.3  -0.4 ]
    ...
```

#### 2.3.3 FlexAttention 等效定义

```python
def alibi_bias(b, h, q_idx, kv_idx):
    return -alibi_slope * abs(q_idx - kv_idx)
```

#### 2.3.4 优势

- **无需位置嵌入**: 不需要学习位置编码
- **长度外推**: 可以处理比训练时更长的序列
- **计算高效**: 偏置可以预计算

---

### 2.4 Sliding Window Attention (滑动窗口注意力)

用于 Longformer、BigBird、Mistral 等模型，将复杂度从 O(n²) 降低到 O(n × window_size)。

#### 2.4.1 数学定义

$$
\text{mask}[i,j] = \begin{cases} 0 & \text{if } |i - j| \leq \frac{\text{window\_size}}{2} \\ -\infty & \text{otherwise} \end{cases}
$$

#### 2.4.2 窗口掩码示例 (window_size=4)

```
Position:  0   1   2   3   4   5   6   7
    0    [ 0   0  -∞  -∞  -∞  -∞  -∞  -∞ ]  // 位置0只能看到0,1
    1    [ 0   0   0  -∞  -∞  -∞  -∞  -∞ ]  // 位置1可以看到0,1,2
    2    [-∞   0   0   0  -∞  -∞  -∞  -∞ ]  // 位置2可以看到1,2,3
    3    [-∞  -∞   0   0   0  -∞  -∞  -∞ ]  // 位置3可以看到2,3,4
    ...
```

#### 2.4.3 FlexAttention 等效定义

```python
def sliding_window(b, h, q_idx, kv_idx):
    return abs(q_idx - kv_idx) <= window_size // 2
```

---

### 2.5 Soft Capping Attention (软截断注意力)

用于 Gemma 2 等模型，通过 tanh 截断提高训练稳定性。

#### 2.5.1 数学定义

$$
\text{score\_mod}(x) = \text{cap} \times \tanh\left(\frac{x}{\text{cap}}\right)
$$

#### 2.5.2 tanh 近似实现

由于 PTO ISA 没有原生 tanh 指令，使用指数近似：

$$
\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

#### 2.5.3 PTO 实现

```pto
// 软截断: cap * tanh(scaled / cap)
%x_div_cap = tdivs %scaled, %50.0        // x / cap
%two_x = tmuls %x_div_cap, %2.0          // 2x / cap
%exp_2x = texp %two_x                    // exp(2x / cap)
%exp_minus_1 = tadds %exp_2x, %-1.0      // exp(2x) - 1
%exp_plus_1 = tadds %exp_2x, %1.0        // exp(2x) + 1
%tanh_x = tdiv %exp_minus_1, %exp_plus_1 // tanh 近似
%capped_scores = tmuls %tanh_x, %50.0    // cap * tanh(x/cap)
```

---

## 3. 多头注意力 (Multi-Head Attention) 组件

### 3.1 线性投影 (Linear Projection)

将输入嵌入投影到 Q、K、V 空间：

```
Q = X @ W_Q   [seq_len, d_model] @ [d_model, head_dim] → [seq_len, head_dim]
K = X @ W_K
V = X @ W_V
```

### 3.2 输出投影 (Output Projection)

将注意力输出投影回模型维度：

```
Output = Attention_Out @ W_O   [seq_len, head_dim] @ [head_dim, d_model] → [seq_len, d_model]
```

### 3.3 完整 MHA 流程

```
1. 输入 X [batch, seq_len, d_model]
2. 线性投影生成 Q, K, V (可并行处理多个头)
3. 对每个头执行 SDPA
4. 拼接所有头的输出
5. 输出投影
```

---

## 4. FlexAttention 核心 API 实现

### 4.1 flex_attention_basic

基础实现，等同于标准 SDPA：

```python
flex_attention(query, key, value)
# 等价于
softmax(query @ key.T / sqrt(d)) @ value
```

### 4.2 flex_attention_with_score_mod

支持 `score_mod` 函数的实现：

```python
def score_mod(score, batch, head, q_idx, kv_idx):
    return modified_score

# 常见 score_mod:
# - Causal: return -inf if q_idx < kv_idx else score
# - ALiBi: return score - slope * |q_idx - kv_idx|
# - Relative: return score + bias_table[q_idx - kv_idx]
```

### 4.3 flex_attention_with_block_mask

支持 `BlockMask` 的稀疏注意力：

```python
block_mask = create_block_mask(mask_fn, B, H, Q_LEN, KV_LEN)
flex_attention(query, key, value, block_mask=block_mask)
```

**BlockMask 优势:**
- 只计算需要的 Q-K 块
- 支持因果、滑动窗口、稀疏等模式
- 显著减少计算量

---

## 5. 数据流图

### 5.1 基础 SDPA 数据流

```
     Q [seq×dim]      K [seq×dim]      V [seq×dim]
         │                │                │
         │     ┌──────────┘                │
         │     │                           │
         ▼     ▼                           │
    ┌─────────────┐                        │
    │   TMATMUL   │ scores = Q @ K^T       │
    └─────────────┘                        │
           │                               │
           ▼                               │
    ┌─────────────┐                        │
    │    TMULS    │ scaled = scores × scale│
    └─────────────┘                        │
           │                               │
           ▼                               │
    ┌─────────────┐                        │
    │   SOFTMAX   │                        │
    │  (多步实现)  │                        │
    └─────────────┘                        │
           │                               │
           │      ┌────────────────────────┘
           │      │
           ▼      ▼
    ┌─────────────┐
    │   TMATMUL   │ output = attn @ V
    └─────────────┘
           │
           ▼
      Output [seq×dim]
```

### 5.2 带 Score Modification 的数据流

```
     Q, K, V                score_mod_bias
         │                       │
         ▼                       │
    ┌─────────────┐              │
    │ Q @ K^T / √d│              │
    └─────────────┘              │
           │                     │
           │     ┌───────────────┘
           │     │
           ▼     ▼
    ┌─────────────┐
    │    TADD     │ modified = scaled + bias
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   SOFTMAX   │
    └─────────────┘
           │
           ▼
      ...继续...
```

---

## 6. Softmax 数值稳定性实现

### 6.1 标准 Softmax 问题

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

当 x 值很大时，`exp(x)` 会溢出。

### 6.2 数值稳定版本

```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

### 6.3 PTO 实现 (近似版本)

代码中使用**行均值**代替行最大值作为偏移量，这是一种简化：

```pto
// 计算行均值作为偏移
%row_sum = trowsum %scores
%row_avg = tdivs %row_sum, %seq_len

// 减去偏移
%shifted = trowexpandsub %scores, %row_avg

// 指数和归一化
%exp_scores = texp %shifted
%row_sum = trowsum %exp_scores
%attention_weights = trowexpanddiv %exp_scores, %row_sum
```

> **注意**: 使用均值而非最大值是一种近似，在实际生产中应使用 `trowmax` 获取真正的最大值。

---

## 7. 编译目标

PTO 程序可以编译到多个硬件后端：

| 后端 | 文件扩展名 | 目标平台 |
|------|-----------|----------|
| PTO | `.pto` | 中间表示 |
| ARM64 | `.c` | Apple Silicon, ARM 服务器 |
| CUDA | `.cu` | NVIDIA GPU |
| Ascend910B | `.cpp` | 华为昇腾 NPU |

### 7.1 生成的 CUDA 代码示例

```cuda
__global__ void scaled_dot_product_attention_kernel(
    float* Q_mem, float* K_mem, float* V_mem, float* output_mem) {
    
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // FUSED (3 ops): Q=TLOAD(...); K=TLOAD(...); V=TLOAD(...)
    if (_row < 8 && _col < 8) {
        Q[_row][_col] = Q_mem[_row * 8 + _col];
        K[_row][_col] = K_mem[_row * 8 + _col];
        V[_row][_col] = V_mem[_row * 8 + _col];
    }
    
    // ... 后续计算 ...
}
```

---

## 8. 性能优化特性

### 8.1 循环融合 (Loop Fusion)

编译器会自动融合连续的逐元素操作：

```
// 原始: 3个独立循环
for: Q = load(...)
for: K = load(...)
for: V = load(...)

// 融合后: 1个循环
for: Q, K, V = load(...)  // 2 loop overheads saved
```

### 8.2 Tile-Based 计算

所有操作基于 Tile（瓦片）进行，默认 8×8：

```python
DEFAULT_TILE_ROWS = 8
DEFAULT_TILE_COLS = 8
```

这允许：
- 高效的寄存器使用
- 更好的缓存局部性
- 适配硬件张量核心

### 8.3 内存层次

```python
class MemorySpace(Enum):
    GM = "gm"    # 全局内存 (Global Memory)
    L1 = "l1"    # L1 缓存
    UB = "ub"    # 统一缓冲区 (Unified Buffer)
    L0A = "l0a"  # L0 A 缓冲区 (矩阵乘法输入 A)
    L0B = "l0b"  # L0 B 缓冲区 (矩阵乘法输入 B)
    L0C = "l0c"  # L0 C 缓冲区 (矩阵乘法输出)
```

---

## 9. 实现对比表

| 注意力变体 | 额外输入 | 核心修改步骤 | 应用场景 |
|-----------|---------|-------------|---------|
| Basic SDPA | 无 | 无 | 基础 Transformer |
| Causal Mask | mask | `tadd %scaled, %mask` | GPT, 解码器 |
| ALiBi | alibi_bias | `tadd %scaled, %alibi` | 长序列生成 |
| Relative Position | rel_pos_bias | `tadd %scaled, %bias` | T5, Swin |
| Sliding Window | window_mask | `tadd %scaled, %mask` | Longformer, Mistral |
| Soft Capping | cap_value | tanh 截断计算 | Gemma 2 |
| Document Mask | doc_mask | `tadd %scaled, %mask` | 多文档批处理 |
| Prefix LM | prefix_mask | 混合掩码 | Encoder-Decoder |

---

## 10. 使用示例

### 10.1 Python API

```python
from pto_torch_flexattention import (
    scaled_dot_product_attention,
    attention_with_causal_mask,
    get_all_programs
)

# 单个程序
program = scaled_dot_product_attention(seq_len=16, head_dim=64)

# 获取所有实现
all_programs = get_all_programs()
for name, prog in all_programs.items():
    print(f"{name}: {len(prog.instructions)} instructions")
```

### 10.2 编译到多后端

```python
from pto_compile import generate_all_backends

program = scaled_dot_product_attention()
results = generate_all_backends(
    program,
    output_prefix="flex_attention",
    enable_fusion=True
)
```

---

## 11. 总结

FlexAttention 的 PTO 实现展示了如何将高级注意力机制分解为基本的张量操作指令。核心洞察包括：

1. **统一的计算模式**: 所有注意力变体共享相同的核心计算流程 (QK^T → scale → modify → softmax → @V)

2. **模块化设计**: 通过 `score_mod` 和 `block_mask` 参数化实现灵活性

3. **硬件无关的中间表示**: PTO ISA 作为桥梁，支持跨平台代码生成

4. **自动优化**: 编译器提供循环融合等优化

5. **数值稳定性**: 所有 softmax 实现都考虑了数值稳定性

这种分层设计使得算法研究者可以专注于注意力模式的创新，而底层优化由编译器自动处理。

---

## 参考文献

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", ICLR 2022
3. PyTorch FlexAttention Documentation: https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
4. PyTorch FlexAttention Blog: https://pytorch.org/blog/flexattention/
