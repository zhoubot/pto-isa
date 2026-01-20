# PTO ISA Improvement Ideas

本文档记录了在使用 PTO ISA 实现 PyTorch API 过程中发现的缺失功能和改进建议。

## 概述

在实现以下 PyTorch API 时发现了 PTO ISA 的局限性：
- **ATen IR Primitives**: 27 个原语
- **torch.nn.functional**: 38 个函数
- **torch.nn 模块**: 24 个算子
- **torch.Tensor 方法**: 55 个方法

---

## 1. 缺失的 ISA 指令

### 1.1 三角函数指令

**现状**: PTO ISA 完全缺少三角函数指令

**建议添加**:
| 指令 | 操作 | 用途 |
|------|------|------|
| `TSIN` | `dst = sin(src)` | 正弦函数 |
| `TCOS` | `dst = cos(src)` | 余弦函数 |
| `TTAN` | `dst = tan(src)` | 正切函数 |
| `TSINH` | `dst = sinh(src)` | 双曲正弦 |
| `TCOSH` | `dst = cosh(src)` | 双曲余弦 |
| `TTANH` | `dst = tanh(src)` | 双曲正切（神经网络激活函数常用） |
| `TASIN` | `dst = arcsin(src)` | 反正弦 |
| `TACOS` | `dst = arccos(src)` | 反余弦 |
| `TATAN` | `dst = arctan(src)` | 反正切 |
| `TATAN2` | `dst = atan2(y, x)` | 二参数反正切 |

**影响的 API**:
- `torch.sin()`, `torch.cos()`, `torch.tan()`
- `torch.sinh()`, `torch.cosh()`, `torch.tanh()`
- `torch.asin()`, `torch.acos()`, `torch.atan()`
- `F.tanh()` - 常用激活函数

**当前解决方案**: 使用 Taylor 级数展开近似实现，如 `sinh(x)`:
```python
# 7项 Taylor 展开: sinh(x) = x + x³/6 + x⁵/120 + x⁷/5040 + ...
.mul("x2", "x", "x")           # x²
.mul("x3", "x2", "x")          # x³
.divs("term3", "x3", 6.0)      # x³/6
# ... 更多项
```
**问题**: 精度有限，代码量大，性能差。

---

### 1.2 幂运算指令

**现状**: 缺少通用幂运算指令

**建议添加**:
| 指令 | 操作 | 用途 |
|------|------|------|
| `TPOW` | `dst = src0 ^ src1` | 幂运算 (tile ^ tile) |
| `TPOWS` | `dst = src ^ scalar` | 标量幂运算 (tile ^ scalar) |

**影响的 API**:
- `torch.pow()`
- `Tensor.pow()`
- 任何需要 `x ** n` 的计算

**当前解决方案**:
- `x²` → `TMUL(x, x)`
- `x⁰·⁵` → `TSQRT(x)`
- `x ** n` (通用) → 无法直接实现

---

### 1.3 Clamp/Clip 指令

**现状**: 缺少范围限制指令

**建议添加**:
| 指令 | 操作 | 用途 |
|------|------|------|
| `TCLAMP` | `dst = clamp(src, min, max)` | 范围限制 |
| `TCLAMPS` | `dst = clamp(src, min_scalar, max_scalar)` | 标量范围限制 |

**影响的 API**:
- `torch.clamp()`
- `F.hardtanh()`
- `F.relu6()` (需要 clamp 到 [0, 6])
- 数值稳定性处理

**当前解决方案**: 组合 `TMAX` + `TMIN`:
```python
.max("tmp", "x", "min_tile")   # max(x, min)
.min("result", "tmp", "max_tile")  # min(max(x, min), max)
```
**问题**: 需要额外的临时 tile 和两条指令。

---

## 2. ISA 已定义但 Builder 未实现的指令

以下指令在 `pto_isa_definition.py` 中已定义，但 `PTOFunctionBuilder` 没有对应方法：

### 2.1 Reduction 指令

| ISA 指令 | 定义行号 | Builder 方法 | 状态 |
|----------|---------|--------------|------|
| `TROWMAX` | 2678 | `rowmax()` | ❌ 未实现 |
| `TROWMIN` | 2692 | `rowmin()` | ❌ 未实现 |
| `TCOLMAX` | 2740 | `colmax()` | ❌ 未实现 |
| `TCOLMIN` | 2754 | `colmin()` | ❌ 未实现 |

**影响的 API**:
- `torch.max(dim=1)`, `torch.min(dim=1)`
- `F.softmax()` - 数值稳定性需要减去 row max
- `F.log_softmax()`

**当前解决方案** (softmax):
```python
# 错误的近似：用均值代替最大值
.rowsum("row_max", "x")
.divs("row_max", "row_max", float(cols))  # mean, not max!
```
**问题**: 数值不稳定，大值输入会导致 exp() 溢出。

---

### 2.2 比较和选择指令

| ISA 指令 | 定义行号 | Builder 方法 | 状态 |
|----------|---------|--------------|------|
| `TCMP` | 2482 | `cmp()` | ❌ 未实现 |
| `TCMPS` | 2498 | `cmps()` | ❌ 未实现 |
| `TSEL` | 2518 | `sel()` | ❌ 未实现 |
| `TSELS` | 2534 | `sels()` | ❌ 未实现 |

**影响的 API**:
- `torch.where(condition, x, y)`
- `F.threshold()`
- `torch.masked_fill()`
- 条件赋值操作

**当前解决方案** (threshold):
```python
# 无法实现真正的条件选择
.max("result", "x", "thresh_tile")  # 近似，功能不完整
```

---

### 2.3 其他未实现指令

| ISA 指令 | 定义行号 | Builder 方法 | 用途 |
|----------|---------|--------------|------|
| `TREM` | 2022 | `rem()` | 取模运算 |
| `TTRANS` | 2886 | `trans()` | 矩阵转置 |
| `TLRELU` | 1854 | `lrelu()` | Leaky ReLU |
| `TRESHAPE` | 2900 | `reshape()` | 形状变换 |
| `TEXTRACT` | 2914 | `extract()` | 子块提取 |
| `TGATHER` | 2930 | `gather()` | 索引收集 |
| `TSCATTER` | 2960 | `scatter()` | 索引分散 |
| `TCVT` | 2979 | `cvt()` | 类型转换 |

---

## 3. 代码生成器的缺失功能

### 3.1 ARM64 后端

以下指令缺少 ARM64 NEON 代码生成：
- `TROWMAX`, `TROWMIN` - 需要水平归约
- `TCMP`, `TSEL` - 需要向量比较和混合
- `TTRANS` - 需要 `vtrn` 等转置指令

### 3.2 CUDA 后端

以下操作需要特殊处理：
- Reduction 操作需要 warp-level 原语 (`__shfl_down_sync`)
- `TMATMUL` 应使用 Tensor Cores

### 3.3 Ascend 后端

以下操作需要映射到 Ascend C API：
- `TROWMAX` → `ReduceMax`
- `TCMP` → `Compare`
- `TSEL` → `Select`

---

## 4. 优先级建议

### 高优先级 (P0) - 基础功能缺失

| 功能 | 原因 |
|------|------|
| `TROWMAX` Builder 实现 | Softmax 数值稳定性必需 |
| `TCMP` + `TSEL` Builder 实现 | 条件操作基础设施 |
| `TTANH` ISA 指令 | 最常用的激活函数之一 |

### 中优先级 (P1) - 常用功能

| 功能 | 原因 |
|------|------|
| `TPOW` / `TPOWS` ISA 指令 | 幂运算常用 |
| `TCLAMP` ISA 指令 | 数值处理常用 |
| `TTRANS` Builder 实现 | 矩阵操作基础 |
| 其他三角函数 | 科学计算需要 |

### 低优先级 (P2) - 完整性

| 功能 | 原因 |
|------|------|
| `TGATHER` / `TSCATTER` Builder | 高级索引操作 |
| `TRESHAPE` Builder | 形状变换 |
| 完整三角函数族 | 完整性 |

---

## 5. 实现示例

### 5.1 建议的 `TTANH` 实现

```python
# pto_isa_definition.py
@dataclass
class TTANH(TileInstruction):
    """Hyperbolic tangent: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)"""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TTANH"
    
    def codegen_arm64_ir(self, ctx) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            body=[LoopBodyOp("TTANH", self.dst.name, [self.src.name])],
            vectorizable=False  # ARM64 lacks native tanh
        )
    
    def codegen_cuda(self, ctx) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            body=[LoopBodyOp("TTANH", self.dst.name, [self.src.name])],
            vectorizable=True  # CUDA has __tanhf
        )
```

### 5.2 建议的 Builder 扩展

```python
# pto_compile.py - PTOFunctionBuilder
def rowmax(self, dst: str, src: str) -> "PTOFunctionBuilder":
    """Row-wise maximum reduction."""
    from pto_isa_definition import TROWMAX
    self._instructions.append(TROWMAX(
        dst=self._get_tile_operand(dst),
        src=self._get_tile_operand(src)
    ))
    return self

def cmp(self, dst: str, src0: str, src1: str, mode: str = "gt") -> "PTOFunctionBuilder":
    """Element-wise comparison producing mask."""
    from pto_isa_definition import TCMP, CompareMode
    mode_map = {"eq": CompareMode.EQ, "ne": CompareMode.NE, 
                "gt": CompareMode.GT, "ge": CompareMode.GE,
                "lt": CompareMode.LT, "le": CompareMode.LE}
    self._instructions.append(TCMP(
        dst=self._get_tile_operand(dst),
        src0=self._get_tile_operand(src0),
        src1=self._get_tile_operand(src1),
        cmp_mode=mode_map.get(mode, CompareMode.GT)
    ))
    return self

def sel(self, dst: str, mask: str, src0: str, src1: str) -> "PTOFunctionBuilder":
    """Select elements based on mask: dst = mask ? src0 : src1"""
    from pto_isa_definition import TSEL
    self._instructions.append(TSEL(
        dst=self._get_tile_operand(dst),
        mask=self._get_tile_operand(mask),
        src0=self._get_tile_operand(src0),
        src1=self._get_tile_operand(src1)
    ))
    return self
```

---

## 6. 总结

| 类别 | 数量 | 影响 |
|------|------|------|
| ISA 缺失指令 | 12+ | 无法精确实现三角函数、幂运算等 |
| Builder 未实现 | 15+ | 已有 ISA 定义但无法使用 |
| 代码生成缺失 | 多个 | 部分后端功能不完整 |

**最紧迫的改进**:
1. 实现 `TROWMAX` Builder 方法和代码生成 → 修复 Softmax 数值稳定性
2. 实现 `TCMP` + `TSEL` Builder 方法 → 支持条件操作
3. 添加 `TTANH` ISA 指令 → 常用激活函数原生支持
