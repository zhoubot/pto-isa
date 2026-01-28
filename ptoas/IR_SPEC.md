# PTOAS IR Spec (v1)

> 目标：在高性能计算/算子开发场景中，缩小“逻辑 Tensor（多维、动态）”与“物理指令（定长、分型）”之间的鸿沟。
>
> 本 IR 通过解耦 **存储布局 (Storage)**、**访问窗口 (View)** 与 **搬运载体 (Tile)**，将复杂的指针算术与 stride 计算交给编译器处理，
> 在保持 PTO C++ 底层性能的同时提升算子开发直觉性。

---

## 1. Core Concepts

### 1.1 Storage: `tensor_view`

`tensor_view` 描述 **Global Memory** 上原始数据的“物理大底座”，包含：

- `dtype`
- `shape`（2D 或 5D；推荐 5D）
- `strides`（2D 或 5D；推荐 5D）
- `layout`（ND/DN/NZ/...）

该层只做**声明**，不做任何数据搬运；它是后续所有 view 变换与 slice 的基准。

### 1.2 View Window: `partition_tensor_view`

`partition_tensor_view` 是从 `tensor_view` 上截取的逻辑窗口：

- `offsets`：从哪开始读（逻辑坐标）
- `sizes`：读多少（逻辑形状）

它承载“计算区域”的语义；编译器负责把它转换成：

- `base_ptr + linear_offset`（物理地址）
- 以及对应的 `Shape<>` / `Stride<>` 组合（供后续搬运指令使用）

### 1.3 Tile Buffer: `tile_buf`

`tile_buf` 是 **物理 2D** 的搬运/计算载体（Tile）。

Tile 具有：

- `rows/cols`：tile 的编译期容量
- `valid_row/valid_col`：本次有效区域（可静态或动态）
- `loc/blayout/slayout/fractal/pad`：底层存储/布局参数

---

## 2. Types (textual contract)

> 本项目当前实现以“文本约定”为主（由 `ptoas` frontend/parser + emitter 实现），并保持与 `include/pto/*` C++ 模板一致。

### 2.1 `!pto.tensor_view<...>`

推荐 5D 形态（对应 `pto::GlobalTensor<Element_, Shape_, Stride_, Layout_>`）：

```text
!pto.tensor_view<
  dtype=f16,
  shape=[n0,n1,n2,n3,n4],
  strides=[s0,s1,s2,s3,s4],
  layout=ND
>
```

说明：

- `shape/strides`：支持 2D 或 5D。2D 视作 `[1,1,1,H,W]`。
- `dyn` 表示 `pto::DYNAMIC`（运行期值）。

### 2.2 `!pto.partition_tensor_view<...>`

逻辑窗口的类型约定（与 `tensor_view` 同构，但语义上要求由 `pto.partition_view` 产生）：

```text
!pto.partition_tensor_view<
  dtype=f16,
  shape=[p0,p1,p2,p3,p4],
  strides=[s0,s1,s2,s3,s4],
  layout=ND
>
```

### 2.3 `!pto.tile_buf<...>`

tile buffer 类型约定（对应 `pto::Tile<...>`）：

```text
!pto.tile_buf<
  loc=Vec,
  dtype=f16,
  rows=256, cols=16,
  v_row=dyn, v_col=16,
  blayout=RowMajor,
  slayout=NoneBox,
  fractal=512,
  pad=Null
>
```

说明：

- `v_row/v_col` 可为 `dyn`（运行期通过 `pto.alloc_tile valid_row=... valid_col=...` 提供）。

---

## 3. Core Instructions

### 3.1 `pto.make_tensor_view`

功能：通过指针建立 `tensor_view`（GlobalTensor 的构造函数语义）。

- 不涉及数据搬运
- 仅声明物理内存排列（strides）与边界（shape）
- 编译器可在此阶段根据 `strides` 识别布局并注入 `layout=NZ/ND/DN` 等硬件提示

IR (示例)：

```text
%x = pto.make_tensor_view %arg0,
  dtype=f32,
  shape=[1,1,16,1024,1024],
  strides=[1048576,1048576,1048576,1024,1]
  : !pto.tensor_view<...>
```

### 3.2 `pto.partition_view`

功能：逻辑窗口切分，在大视图上截取特定的计算区域，生成 `partition_tensor_view`。

- `offsets`：决定“从哪开始读”
- `sizes`：决定“读多少”

IR (示例)：

```text
%p = pto.partition_view %x,
  offsets=[0,0,0,0,0],
  sizes=[1,1,16,16,16]
  : !pto.tensor_view<...> -> !pto.partition_tensor_view<...>
```

### 3.3 `pto.tload`

功能：物理搬运 + 维度塌缩（logical ND -> physical 2D tile）。

约束（ND 布局下的规范约束，和 PTO 底层实现一致）：

- 输入必须为 `partition_tensor_view`
- 输出必须为 `tile_buf`
- 必须满足：
  - `tile.valid_col == shape[4]`
  - `tile.valid_row == shape[0] * shape[1] * shape[2] * shape[3]`
  - 因此总元素量满足 `∏sizes == valid_row * valid_col`

IR-Level2（推荐，显式 ins/outs）：

```text
pto.tload  ins(%p   : !pto.partition_tensor_view<...>)
          outs(%tile: !pto.tile_buf<...>)
```

IR-Level1（SSA sugar）：

```text
%tile = pto.tload %p : !pto.partition_tensor_view<...> -> !pto.tile_buf<...>
```

---

## 4. Layout Inference (from `strides`)

编译器可通过 `pto.make_tensor_view` 的 `strides` 推导物理存储模式，从而选择/注入 `layout`：

### 4.1 ND（Row-Major）

典型条件（5D）：

```text
stride[i] == stride[i+1] * shape[i+1]   (i=0..3), 且 stride[4] == 1
```

### 4.2 DN（Col-Major）

典型条件（5D）：

```text
stride[i+1] == stride[i] * shape[i]     (i=0..3), 且 stride[0] == 1
```

### 4.3 NZ（Fractal）

需要满足硬件对齐约束，方可触发高速分型搬运路径：

1) 维度对齐：`shape[2] == 16`
2) 内存对齐：`shape[2] * shape[3] * sizeof(dtype) == 512 bytes`
3) 步长特征：`stride[4] == 1` 且 `stride[3] == shape[4]`

伪代码：

```c++
Layout InferLayout(const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& stride,
                   size_t dataTypeSize) {
  if (shape.size() >= 5) {
    int64_t sh3 = shape[2], sh4 = shape[3], sh5 = shape[4];
    int64_t st4 = stride[3], st5 = stride[4];
    bool alignMatch = (sh3 == 16) && (sh3 * sh4 * dataTypeSize == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch) return Layout::NZ;
  }
  // ... ND / DN fallback ...
  return Layout::ND;
}
```

---

## 5. Tile Allocation

### 5.1 Static valid shape

```text
%t = pto.alloc_tile : !pto.tile_buf<..., rows=32, cols=32, v_row=32, v_col=32, ...>
```

### 5.2 Dynamic valid shape

```text
%t = pto.alloc_tile valid_row=%vr valid_col=%vc
  : !pto.tile_buf<..., rows=32, cols=32, v_row=dyn, v_col=dyn, ...>
```

### 5.3 Tile valid masks (irregular shapes)

`tile_buf` 的 `v_row/v_col` + `pto.alloc_tile valid_row/valid_col` 用于支持 **partial tile**（矩阵维度不是 tile 大小整数倍时的边界 tile）。

语义要点：

- `rows/cols`：tile 的 **容量**（编译期固定）。
- `valid_row/valid_col`：tile 的 **本次有效窗口**（可运行期变化），用于约束 `TLOAD/TSTORE` 的读写范围，并对无效区域做 pad/忽略。

示例：18x19 上以 16x16 tile 计算边界 valid（运行期按 block/tile 选择）：

```text
%bid  = pto.get_block_idx
%tile_c = pto.irem %bid, 2 : index
%tile_r = pto.idiv %bid, 2 : index
%r0 = pto.imul %tile_r, 16 : index
%c0 = pto.imul %tile_c, 16 : index

%rem_r = pto.isub 18, %r0 : index
%rem_c = pto.isub 19, %c0 : index
%vr = pto.imin 16, %rem_r
%vc = pto.imin 16, %rem_c

%tx = pto.alloc_tile valid_row=%vr valid_col=%vc
  : !pto.tile_buf<loc=Vec, dtype=f32, rows=16, cols=16, v_row=dyn, v_col=dyn,
                  blayout=RowMajor, slayout=NoneBox, fractal=512, pad=Null>
```

> 实现说明：`ptoas` 在 lowering 旧式 indexed 形式（如 `pto.tload %x[%r0,%c0]` / `pto.tstore %y[%r0,%c0], %t`）
> 时，会按 tile 的 `GetValidRow()/GetValidCol()` 构造一个 tile-shaped `GlobalTensor` view（shape 为动态），确保
> `TLOAD/TSTORE` 的 “src/dst shape == valid_row/valid_col” 约束在 partial tile 场景下依然成立。

---

## 6. Notes on Compatibility

- 现有 PTO-AS（基于 `pto.make_tensor_view` / `pto.subview` / `tload %x[r,c]`）仍被视为兼容输入；
  新 IR 推荐逐步迁移到 `partition_view + ins/outs` 的显式形式。
