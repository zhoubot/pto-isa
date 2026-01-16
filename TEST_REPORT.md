# PTO ISA Compiler - ARM64 测试报告

**生成时间**: 2026-01-16  
**平台**: macOS Darwin arm64 (Apple Silicon)  
**编译器**: clang with ARM64 NEON  

## 测试摘要

| 测试套件 | 通过 | 失败 | 总计 | 通过率 |
|---------|------|------|------|--------|
| sinh() Taylor Expansion | 6 | 0 | 6 | 100% |
| ATen IR Primitives | 12 | 10 | 22 | 54.5% |
| torch.Tensor Methods | 11 | 13 | 24 | 45.8% |
| torch.nn.functional | 4 | 12 | 16 | 25% |
| torch.nn Operators | 5 | 11 | 16 | 31.3% |
| FlexAttention | 2 | 4 | 6 | 33.3% |
| **总计** | **40** | **50** | **90** | **44.4%** |

---

## 详细结果

### 1. sinh() Taylor Expansion

**通过率: 100% (6/6)**

| 测试用例 | 状态 | 最大误差 |
|----------|------|----------|
| sinh_random_small | ✅ PASS | 1.19e-07 |
| sinh_zeros | ✅ PASS | 0.00e+00 |
| sinh_small_positive | ✅ PASS | 2.98e-08 |
| sinh_small_negative | ✅ PASS | 2.98e-08 |
| sinh_moderate_values | ✅ PASS | 2.38e-07 |
| sinh_linspace | ✅ PASS | 1.19e-07 |

---

### 2. ATen IR Primitives

**通过率: 54.5% (12/22)**

#### 通过的测试

| 测试用例 | 状态 | 最大误差 | 说明 |
|----------|------|----------|------|
| prims_abs | ✅ PASS | 0.00e+00 | 绝对值 |
| prims_neg | ✅ PASS | 0.00e+00 | 取反 |
| prims_sqrt | ✅ PASS | 0.00e+00 | 平方根 |
| aten_relu | ✅ PASS | 0.00e+00 | ReLU激活 |
| aten_sinh | ✅ PASS | 3.29e-05 | 双曲正弦 |
| prims_add | ✅ PASS | 0.00e+00 | 加法 |
| prims_sub | ✅ PASS | 0.00e+00 | 减法 |
| prims_mul | ✅ PASS | 0.00e+00 | 乘法 |
| prims_div | ✅ PASS | 0.00e+00 | 除法 |
| prims_maximum | ✅ PASS | 0.00e+00 | 最大值 |
| prims_minimum | ✅ PASS | 0.00e+00 | 最小值 |
| aten_add_scalar | ✅ PASS | 0.00e+00 | 标量加法 |

#### 失败的测试

| 测试用例 | 状态 | 最大误差 | 原因分析 |
|----------|------|----------|----------|
| prims_exp | ❌ FAIL | 4.67e+00 | 指数函数实现精度不足 |
| prims_log | ❌ FAIL | 7.43e+00 | 对数函数实现精度不足 |
| prims_rsqrt | ❌ FAIL | 1.92e-03 | 倒数平方根精度问题 |
| prims_reciprocal | ❌ FAIL | 1.48e+00 | 倒数函数实现问题 |
| aten_sigmoid | ❌ FAIL | 5.79e+00 | sigmoid使用近似公式 |
| aten_tanh | ❌ FAIL | inf | tanh实现有数值问题 |
| aten_cosh | ❌ FAIL | 3.48e+00 | 双曲余弦精度不足 |
| aten_gelu | ❌ FAIL | 1.70e+01 | GELU使用近似实现 |
| aten_silu | ❌ FAIL | 1.10e+01 | SiLU依赖sigmoid |
| aten_mul_scalar | ❌ FAIL | 1.44e+00 | 标量乘法实现问题 |

---

### 3. torch.Tensor Methods

**通过率: 45.8% (11/24)**

#### 通过的测试

| 测试用例 | 状态 | 最大误差 |
|----------|------|----------|
| tensor_add | ✅ PASS | 0.00e+00 |
| tensor_sub | ✅ PASS | 0.00e+00 |
| tensor_mul | ✅ PASS | 0.00e+00 |
| tensor_div | ✅ PASS | 0.00e+00 |
| tensor_neg | ✅ PASS | 0.00e+00 |
| tensor_abs | ✅ PASS | 0.00e+00 |
| tensor_sqrt | ✅ PASS | 0.00e+00 |
| tensor_relu | ✅ PASS | 0.00e+00 |
| tensor_max_elementwise | ✅ PASS | 0.00e+00 |
| tensor_min_elementwise | ✅ PASS | 0.00e+00 |
| tensor_clamp | ✅ PASS | 0.00e+00 |

#### 失败的测试

| 测试用例 | 状态 | 最大误差 |
|----------|------|----------|
| tensor_rsqrt | ❌ FAIL | 1.92e-03 |
| tensor_reciprocal | ❌ FAIL | 1.48e+00 |
| tensor_sin | ❌ FAIL | 3.93e-01 |
| tensor_cos | ❌ FAIL | 8.83e-01 |
| tensor_tan | ❌ FAIL | 2.38e-03 |
| tensor_sinh | ❌ FAIL | 1.41e+00 |
| tensor_cosh | ❌ FAIL | 3.48e+00 |
| tensor_tanh | ❌ FAIL | 9.52e+00 |
| tensor_exp | ❌ FAIL | 4.67e+00 |
| tensor_log | ❌ FAIL | 2.11e+00 |
| tensor_log2 | ❌ FAIL | 3.05e+00 |
| tensor_log10 | ❌ FAIL | 9.17e-01 |
| tensor_sigmoid | ❌ FAIL | 5.79e+00 |

---

### 4. torch.nn.functional

**通过率: 25% (4/16)**

#### 通过的测试

| 测试用例 | 状态 | 最大误差 |
|----------|------|----------|
| F_relu | ✅ PASS | 0.00e+00 |
| F_relu6 | ✅ PASS | 0.00e+00 |
| F_leaky_relu | ✅ PASS | 0.00e+00 |
| F_normalize | ✅ PASS | 5.96e-08 |

#### 失败的测试

| 测试用例 | 状态 | 原因 |
|----------|------|------|
| F_elu | ❌ FAIL | 精度问题 |
| F_gelu | ❌ FAIL | 近似实现 |
| F_sigmoid | ❌ FAIL | 精度问题 |
| F_silu | ❌ FAIL | 依赖sigmoid |
| F_tanh | ❌ FAIL | 数值问题 |
| F_softplus | ❌ FAIL | 精度问题 |
| F_softmax | ❌ FAIL | 数值稳定性 |
| F_log_softmax | ❌ FAIL | 数值稳定性 |
| F_logsigmoid | ❌ FAIL | 精度问题 |
| F_mse_loss | ❌ FAIL | 编译错误 |
| F_l1_loss | ❌ FAIL | 编译错误 |
| F_smooth_l1_loss | ❌ FAIL | 编译错误 |

---

### 5. torch.nn Operators

**通过率: 31.3% (5/16)**

#### 通过的测试

| 测试用例 | 状态 | 最大误差 |
|----------|------|----------|
| nn_ReLU | ✅ PASS | 0.00e+00 |
| nn_ReLU6 | ✅ PASS | 0.00e+00 |
| nn_LeakyReLU | ✅ PASS | 0.00e+00 |
| nn_Hardsigmoid | ✅ PASS | 0.00e+00 |
| nn_Hardswish | ✅ PASS | 2.38e-07 |

#### 失败的测试

| 测试用例 | 状态 | 原因 |
|----------|------|------|
| nn_ELU | ❌ FAIL | 精度问题 |
| nn_Sigmoid | ❌ FAIL | 精度问题 |
| nn_Tanh | ❌ FAIL | 数值溢出 |
| nn_GELU | ❌ FAIL | 近似实现 |
| nn_SiLU | ❌ FAIL | 依赖sigmoid |
| nn_Mish | ❌ FAIL | 数值溢出 |
| nn_Softmax | ❌ FAIL | 数值稳定性 |
| nn_Softplus | ❌ FAIL | 精度问题 |
| nn_MSELoss | ❌ FAIL | 编译错误 |
| nn_L1Loss | ❌ FAIL | 编译错误 |
| nn_SmoothL1Loss | ❌ FAIL | 编译错误 |

---

### 6. FlexAttention

**通过率: 33.3% (2/6)**

| 测试用例 | 状态 | 原因 |
|----------|------|------|
| matmul_qk | ✅ PASS | (简化测试) |
| attention_scale | ✅ PASS | (简化测试) |
| score_to_weight | ❌ FAIL | 编译错误 |
| create_causal_mask | ❌ FAIL | 编译错误 |
| softmax_row | ❌ FAIL | 编译错误 |
| output_projection | ❌ FAIL | 编译错误 |

---

## 问题分析

### 完美通过的操作类别

以下操作类型生成的代码正确性完美：

- **基本算术运算**: add, sub, mul, div
- **简单一元运算**: abs, neg, sqrt
- **比较运算**: max, min, clamp
- **简单激活函数**: relu, relu6, leaky_relu
- **硬编码激活**: hardsigmoid, hardswish

### 需要改进的操作类别

1. **超越函数** (exp, log, sin, cos, tan, sinh, cosh, tanh)
   - 问题：使用 Taylor 展开或近似公式，精度不足
   - 建议：使用更高精度的实现或调用标准库函数

2. **复合激活函数** (sigmoid, gelu, silu, mish, elu, selu)
   - 问题：依赖超越函数，累积误差
   - 建议：优化基础函数实现

3. **数值稳定性** (softmax, log_softmax)
   - 问题：存在数值溢出/下溢
   - 建议：添加数值稳定性处理 (max减法)

4. **多输入操作** (loss functions, attention)
   - 问题：内存引用名称不匹配导致编译错误
   - 建议：修复测试框架中的变量名映射

---

## 结论

### 核心优势

- 基础张量运算（加减乘除、最大最小值）实现正确
- 简单激活函数（ReLU系列）工作良好
- ARM64 NEON 向量化代码生成正确

### 改进方向

1. **提高超越函数精度**: 增加 Taylor 展开项数或使用混合算法
2. **数值稳定性**: 为 softmax 等操作添加数值稳定处理
3. **测试框架**: 修复多输入操作的变量名映射问题
4. **文档**: 标注各操作的精度保证等级

---

## 运行测试

```bash
# 运行所有测试
cd examples
python run_all_tests.py

# 运行单个测试套件
python test_pto_isa_sinh.py
python test_pto_aten_ir_primitives.py
python test_pto_torch_tensor.py

# 快速测试
python run_all_tests.py --quick
```

---

*报告由 PTO ISA Compiler 测试框架自动生成*
