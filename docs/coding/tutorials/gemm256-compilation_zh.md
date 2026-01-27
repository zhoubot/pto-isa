# 教程：工具链如何“变换”一个 Kernel（以 GEMM256 为例）

本文用 `kernels/python/gemm256.py` 做示例，面向用户讲清楚整个工具链在每一步：

- 输入是什么
- 工具做了什么
- 输出是什么（并把**完整中间产物**贴在文档里，代码块中不使用省略号）
- 怎么运行/验证

总体流程（概念上）：

1) **Python 前端（用户代码）** → 结构化 PTO-AS 文本（`.pto`）
2) **`ptoas`（编译器）** → C++ 源码（CPU 版本 `.cpu.cpp` / NPU 版本 `.cce.cpp`）+（可选）`.bin`
3) **`bisheng`（Ascend 工具链）** → 可加载的 fatobj 动态库（`.so`）
4) **运行器（ACL）** → 在 NPU（或模拟器）上 launch kernel

![GEMM pipeline](../../figures/gemm_pipeline.png)

## 先决条件

- 从 repo 根目录执行命令。
- Python `>= 3.8`，并安装 `numpy`。
- 已构建 `ptoas` 二进制：`ptoas/mlir/build/bin/ptoas`（见 `ptoas/mlir/README.md`）。
- 若要跑 **NPU/模拟器**：安装 Ascend CANN/toolkit，设置 `ASCEND_HOME_PATH`；`bisheng` 可用；Python 能 `import acl`。

建议先设置环境变量（本文后续命令默认用它们）：

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
export PTOAS=$PWD/ptoas/mlir/build/bin/ptoas
export OUT=/tmp/pto_gemm256
```

## 阶段 A：用户输入（Python kernel）

这一阶段你写的是“正常 Python”，用 `pto_as.PTO` builder 描述 tile 级算子和控制流。

`kernels/python/gemm256.py`（完整内容）：

```python
from __future__ import annotations

from pto_as import PTO


def gemm256():
    # C[256,256] = A[256,256] @ B[256,256], using (16,16,16) tiles.
    pto = PTO("gemm256")
    pto.prologue()

    a = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    b = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    c = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    a_mat = pto.mat(dtype="f16", shape=(16, 16))
    b_mat = pto.mat(dtype="f16", shape=(16, 16))

    a_left_0 = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    a_left_1 = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_right_0 = pto.right(dtype="f16", shape=(16, 16))
    b_right_1 = pto.right(dtype="f16", shape=(16, 16))
    c_acc = pto.acc(dtype="f32", shape=(16, 16))

    for mi in range(0, 256, 16):
        for nj in range(0, 256, 16):
            for kk in range(0, 256, 16):
                a_mat = pto.load(a, mi, kk)
                b_mat = pto.load(b, kk, nj)

                it0 = kk // 16
                lane = it0 % 2
                if lane == 0:
                    a_left_0 = pto.mov(a_mat)
                    b_right_0 = pto.mov(b_mat)
                    if kk == 0:
                        c_acc = pto.tmatmul(a_left_0, b_right_0)
                    else:
                        c_acc = pto.tmatmul_acc(c_acc, a_left_0, b_right_0)
                else:
                    a_left_1 = pto.mov(a_mat)
                    b_right_1 = pto.mov(b_mat)
                    c_acc = pto.tmatmul_acc(c_acc, a_left_1, b_right_1)

            pto.store(c, mi, nj, c_acc)

    pto.epilogue()
    return pto.program()
```

这一段 Python 代码的“语义”可以理解为：

- 以 16×16×16 的 tile 在 (M,N,K) 维度上三重循环；
- 每个 K tile：`TLOAD(A)` + `TLOAD(B)` → `TMOV` 到 Left/Right → `TMATMUL`/`TMATMUL_ACC`；
- 最后把 `c_acc` 写回 GM（`TSTORE`）。

## 阶段 B：Python → PTO-AS（输出 `.pto`，不插同步/事件）

### 这一步工具做了什么

Python 前端会把你的 Python 代码编译成一个结构化的 PTO-AS 文本：

- Python `for/if` 会变成 `scf.for` / `scf.if`
- `pto.load/mov/tmatmul_acc/store` 会变成 `pto.tload/pto.tmov/pto.tmatmul_acc/pto.tstore`
- 默认会在文件头部加一个 host spec（用于运行器分配输入输出张量）

### 如何生成

```bash
mkdir -p "$OUT"
python3 - <<'PY'
from pathlib import Path
from ptoas.python import binding

pto_path = Path(__import__("os").environ["OUT"]) / "gemm256.pto"
binding.write_pto(Path("kernels/python/gemm256.py"), kernel="gemm256", out_path=pto_path, universal=True)
print("wrote:", pto_path)
PY
```

### 输出（`$OUT/gemm256.pto`，完整内容）

<details>
<summary>点击展开：完整 gemm256.pto</summary>

```text
; PTO_HOST_SPEC_BEGIN v1
; {
;   "args": [
;     {
;       "dtype": "f16",
;       "role": "in",
;       "shape": [
;         256,
;         256
;       ]
;     },
;     {
;       "dtype": "f16",
;       "role": "in",
;       "shape": [
;         256,
;         256
;       ]
;     },
;     {
;       "dtype": "f32",
;       "role": "out",
;       "shape": [
;         256,
;         256
;       ]
;     }
;   ],
;   "block_dim": 1,
;   "kernel_name": "pto_kernel",
;   "seed": 0
; }
; PTO_HOST_SPEC_END
prologue
%a = pto.make_tensor_view %arg0, dtype=f16, shape=[256,256] strides=[256,1], layout=ND
%b = pto.make_tensor_view %arg1, dtype=f16, shape=[256,256] strides=[256,1], layout=ND
%c = pto.make_tensor_view %arg2, dtype=f32, shape=[256,256] strides=[256,1], layout=ND
%a_mat = pto.alloc_tile : !pto.tile<loc=Mat, dtype=f16, rows=16, cols=16, blayout=ColMajor, valid=16x16, slayout=RowMajor, fractal=512, pad=Null>
%b_mat = pto.alloc_tile : !pto.tile<loc=Mat, dtype=f16, rows=16, cols=16, blayout=ColMajor, valid=16x16, slayout=RowMajor, fractal=512, pad=Null>
%a_left_0 = pto.alloc_tile : !pto.tile<loc=Left, dtype=f16, rows=16, cols=16, blayout=RowMajor, valid=16x16, slayout=RowMajor, fractal=512, pad=Null>
%a_left_1 = pto.alloc_tile : !pto.tile<loc=Left, dtype=f16, rows=16, cols=16, blayout=RowMajor, valid=16x16, slayout=RowMajor, fractal=512, pad=Null>
%b_right_0 = pto.alloc_tile : !pto.tile<loc=Right, dtype=f16, rows=16, cols=16, blayout=RowMajor, valid=16x16, slayout=ColMajor, fractal=512, pad=Null>
%b_right_1 = pto.alloc_tile : !pto.tile<loc=Right, dtype=f16, rows=16, cols=16, blayout=RowMajor, valid=16x16, slayout=ColMajor, fractal=512, pad=Null>
%c_acc = pto.alloc_tile : !pto.tile<loc=Acc, dtype=f32, rows=16, cols=16, blayout=ColMajor, valid=16x16, slayout=RowMajor, fractal=1024, pad=Null>
scf.for %mi = 0 to 256 step 16 {
  scf.for %nj = 0 to 256 step 16 {
    scf.for %kk = 0 to 256 step 16 {
      %a_mat = pto.tload %a[%mi, %kk]
      %b_mat = pto.tload %b[%kk, %nj]
      %it0 = pto.idiv %kk, 16 : index
      %lane = pto.irem %it0, 2 : index
      %t1 = pto.icmp_eq %lane, 0 : i1
      scf.if %t1 {
        %a_left_0 = pto.tmov %a_mat
        %b_right_0 = pto.tmov %b_mat
        %t2 = pto.icmp_eq %kk, 0 : i1
        scf.if %t2 {
          %c_acc = pto.tmatmul %a_left_0, %b_right_0
        } else {
          %c_acc = pto.tmatmul_acc %c_acc, %a_left_0, %b_right_0
        }
      } else {
        %a_left_1 = pto.tmov %a_mat
        %b_right_1 = pto.tmov %b_mat
        %c_acc = pto.tmatmul_acc %c_acc, %a_left_1, %b_right_1
      }
    }
    pto.tstore %c[%mi, %nj], %c_acc
  }
}
epilogue
```

</details>

### 额外检查：确认没有插同步/事件

```bash
rg -n "tsync|record_event|wait_event|event" "$OUT/gemm256.pto" || true
```

## 阶段 C：`ptoas` 的 pass 做了什么（理解“编译器中间步骤”）

`ptoas` 的入口在 `ptoas/mlir/tools/ptoas_main.cpp`。它做的事情按顺序可以理解为：

1) 解析 `.pto` → 得到一个 MLIR `module`
2) 运行可选的 pass（由命令行开关控制）
3) 选择后端 emitter：生成 CPU C++ 或 NPU CCE C++
4) 若指定 `--emit-bin`：调用 `bisheng` 把 CCE 源码编译成 `.bin`

当前 `ptoas_main.cpp` 里明确串了两个 prototype pass：

- `--assign-tile-addrs`（默认开启）：`ptoas::createAssignTileAddressesPass()`，源码在 `ptoas/mlir/lib/AssignTileAddressesPass.cpp`
- `--insert-events`（默认关闭）：`ptoas::createInsertEventsPass()`，源码在 `ptoas/mlir/lib/InsertEventsPass.cpp`

下面用“用户视角”解释这两个 pass 到底在解决什么问题。

补充说明：仓库里还有其它 PTO dialect 相关的 transform/pass（例如 `ptoas/mlir/lib/InferPTOMemScope.cpp`），但它们并不在 `ptoas/mlir/tools/ptoas_main.cpp` 的默认 CLI pipeline 里启用；本文只解释 CLI 里实际会跑到的两条 pass。

### Pass 1：AssignTileAddresses（`--assign-tile-addrs`，默认开启）

它解决的问题：在 PTO-AS 里你可以只写 `pto.alloc_tile`，但 tile 本质上需要落到某个 on-core/local buffer 的地址上（例如 L0A/L0B/L0C/UB/L1 等）。

这个 pass 会扫描所有 `pto.alloc_tile`：

- 若某个 tile 已经在 `pto.alloc_tile <name, addr>` 给出了地址，它会尊重你给的值；
- 若没有给地址，它会根据 tile 的 `loc=`（例如 `Mat/Left/Right/Acc/Vec`）和 `fractal=` 字段，按一套保守的策略分配地址，避免不同 tile 的地址重叠。

你在后面看到的 `TASSIGN(t_a_mat, 0x0)`、`TASSIGN(t_b_mat, 0x20000)`、`TASSIGN(t_a_left_1, 0x8000)` 这类语句，就是这个 pass（或前端显式地址）最终带来的效果。

如果你手动关掉它：

```bash
$PTOAS "$OUT/gemm256.pto" --target cpu --no-assign-tile-addrs -o "$OUT/gemm256.noaddr.cpu.cpp" --repo-root "$PWD"
```

你需要自己在 `.pto` 里把 `pto.alloc_tile` 写成带地址的形式（否则生成的 C++ 通常无法正确运行/映射存储）。

### Pass 2：InsertEvents（`--insert-events`，默认关闭）

它解决的问题：真实硬件上，不同“流水线/pipe”（例如 MTE、M、V、FIX 等）之间存在依赖关系时，需要显式同步，否则可能出现 RAW/WAR/WAW hazards。

这个 pass 的策略（prototype，启发式）大致是：

- 把每条 PTO 指令（例如 `pto.tload`、`pto.tmov`、`pto.tmatmul`、`pto.tmatmul_acc`、`pto.tstore`）映射为一个 “OpEnum”（例如 `TLOAD`、`TMOV_M2L`、`TMATMUL`、`TSTORE_MAT`）；
- 再把 OpEnum 映射到一个 “pipe”（例如 `TLOAD -> MTE2`，`TMATMUL -> M`，`TMOV_M2L -> MTE1` 等）；
- 跟踪每个 tile 的“最近一次定义”（哪条 op 产生了它）；
- 若某条 consumer 指令在 pipe B 上使用了某个 tile，而该 tile 最近一次由 pipe A 的 producer 产生（A != B），则：
  - 在 producer 之后插入 `pto.record_event`（相当于 set_flag）
  - 在 consumer 之前插入 `pto.wait_event`（相当于 wait_flag）
- 为了避免 token 消耗/复用导致的死锁或错误，它会为每个 `(srcOp,dstOp,key)` 分配 0..7 的 token（不足时退化为 hash）
- 针对 `scf.for` 还会额外插入一套“每迭代握手”来保守处理 loop-carried 的 tile 复用风险

打开它的方式：

```bash
$PTOAS "$OUT/gemm256.pto" --target npu --insert-events -o "$OUT/gemm256.events.cce.cpp" --kernel-name pto_kernel_gemm256 --arch dav-c220-cube --memory-model MEMORY_BASE --repo-root "$PWD"
```

在很多简单 demo 里不开 `--insert-events` 也能跑通；但一旦你做更复杂的流水化（例如 ping-pong buffer、异步 load/compute overlap），一般需要更严格的同步策略。

### 附录：`ptoas` CLI 入口（`ptoas/mlir/tools/ptoas_main.cpp`，完整内容）

<details>
<summary>点击展开：完整 ptoas_main.cpp</summary>

```cpp
#include "ptoas/BishengDriver.h"
#include "ptoas/CCEmitter.h"
#include "ptoas/PTOASFrontend.h"
#include "ptoas/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

namespace {

static std::string getEnvOrEmpty(const char *name) {
  if (const char *env = std::getenv(name))
    return env;
  return "";
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::Required, llvm::cl::desc("<input.pto>"));
  llvm::cl::opt<std::string> output("o", llvm::cl::init(""), llvm::cl::desc("Output source path"));
  llvm::cl::opt<std::string> target("target", llvm::cl::init("npu"),
                                    llvm::cl::desc("Target: npu (CCE) or cpu (CPU simulator C++)"));
  llvm::cl::opt<std::string> kernelName("kernel-name", llvm::cl::init("pto_kernel"),
                                        llvm::cl::desc("Generated kernel function name"));
  llvm::cl::opt<std::string> emitBin("emit-bin", llvm::cl::init(""), llvm::cl::desc("Also compile and emit .bin"));
  llvm::cl::opt<std::string> arch("arch", llvm::cl::init("dav-c220-vec"),
                                  llvm::cl::desc("CCE arch (dav-c220-vec/dav-c220-cube/dav-c310)"));
  llvm::cl::opt<std::string> memoryModel("memory-model", llvm::cl::init("MEMORY_BASE"),
                                         llvm::cl::desc("MEMORY_BASE or REGISTER_BASE"));
  llvm::cl::opt<std::string> repoRootOpt("repo-root", llvm::cl::init(""),
                                         llvm::cl::desc("Repo root path (for -I<repo>/include)"));
  llvm::cl::opt<std::string> ascendHomeOpt("ascend-home", llvm::cl::init(""),
                                           llvm::cl::desc("ASCEND_HOME_PATH (for bisheng includes)"));
  llvm::cl::opt<bool> insertEvents("insert-events", llvm::cl::init(false),
                                   llvm::cl::desc("Insert record_event/wait_event for cross-pipe deps (prototype)"));
  llvm::cl::opt<bool> assignTileAddrs("assign-tile-addrs", llvm::cl::init(true),
                                      llvm::cl::desc("Assign default addresses to tiles (prototype)"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "ptoas (MLIR-based prototype)\n");

  mlir::MLIRContext ctx;
  std::string err;
  auto module = ptoas::parsePTOASFile(input, ctx, err);
  if (!module) {
    llvm::errs() << "parse failed: " << err << "\n";
    return 1;
  }

  if (assignTileAddrs || insertEvents) {
    mlir::PassManager pm(&ctx);
    if (assignTileAddrs)
      pm.addPass(ptoas::createAssignTileAddressesPass());
    if (insertEvents)
      pm.addPass(ptoas::createInsertEventsPass());
    if (mlir::failed(pm.run(module))) {
      llvm::errs() << "pass pipeline failed\n";
      return 1;
    }
  }

  // Default output path: <input>.<ext> in CWD.
  std::string outPath = output;
  if (outPath.empty()) {
    llvm::SmallString<256> p(input);
    // NPU sources are still compiled as CCE via `bisheng -xcce`; `.cpp` is used for
    // better editor/tooling compatibility (matches the manual kernels style).
    llvm::sys::path::replace_extension(p, "cpp");
    if (target == "cpu")
      llvm::sys::path::replace_extension(p, "cpu.cpp");
    outPath = p.str().str();
  }

  auto repoRoot = !repoRootOpt.empty() ? repoRootOpt : getEnvOrEmpty("PTO_REPO_ROOT");
  if (repoRoot.empty())
    repoRoot = ".";

  std::string outText;
  if (target == "cpu") {
    outText = ptoas::emitCpuCppFromModule(module, repoRoot);
  } else if (target == "npu") {
    outText = ptoas::emitCceFromModule(module, repoRoot, memoryModel, kernelName);
  } else {
    llvm::errs() << "unknown --target: " << target << " (expected: npu|cpu)\n";
    return 1;
  }
  std::error_code ec;
  llvm::raw_fd_ostream os(outPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "failed to write " << outPath << ": " << ec.message() << "\n";
    return 1;
  }
  os << outText;
  os.flush();

  if (!emitBin.empty() && target != "cpu") {
    ptoas::BishengCompileOptions opts;
    opts.ascendHomePath = !ascendHomeOpt.empty() ? ascendHomeOpt : getEnvOrEmpty("ASCEND_HOME_PATH");
    opts.repoRoot = repoRoot;
    opts.arch = arch;
    opts.memoryModel = memoryModel;
    auto err2 = ptoas::compileCceToBin(outPath, emitBin, opts);
    if (!err2.empty()) {
      llvm::errs() << err2 << "\n";
      return 1;
    }
  }

  llvm::outs() << "wrote " << outPath << "\n";
  if (!emitBin.empty() && target != "cpu")
    llvm::outs() << "wrote " << emitBin << "\n";
  return 0;
}
```

</details>

## 阶段 D：PTO-AS → C++（CPU 版本：`.cpu.cpp`，用于 CPU 参考/仿真）

### 这一步工具做了什么

`ptoas --target cpu` 会把 `.pto` 变成一个可在 CPU 上运行的 C++，其核心思路是：

- 用 `GlobalTensor`/`Tile` 模板把张量/Tile 的类型、shape、stride 明确出来；
- 把 `.pto` 里的 `pto.tload/tmov/tmatmul_acc/tstore` 变成 C++ 里的 `TLOAD/TMOV/TMATMUL_ACC/TSTORE`；
- 保留结构化控制流（`scf.for/scf.if`）为 C++ 的 `for/if`；
- 生成一个固定入口（本例为）`extern "C" void pto_kernel_cpu(void* arg0, void* arg1, void* arg2)`，方便被 Python `ctypes` 调用。

### 如何生成

```bash
$PTOAS "$OUT/gemm256.pto" --target cpu -o "$OUT/gemm256.cpu.cpp" --repo-root "$PWD"
```

### 输出（`$OUT/gemm256.cpu.cpp`，完整内容）

<details>
<summary>点击展开：完整 gemm256.cpu.cpp</summary>

```cpp
// Generated by ptoas (CPU simulator)
#define __CPU_SIM
#include <pto/pto-inst.hpp>
#include <cstdint>
using namespace pto;

extern "C" void pto_kernel_cpu(void* arg0, void* arg1, void* arg2) {
  auto* arg0_ptr = (half*)arg0;
  using arg0_Shape = Shape<1, 1, 1, 256, 256>;
  using arg0_Stride = Stride<1, 1, 1, 256, 1>;
  using arg0_Tensor = GlobalTensor<half, arg0_Shape, arg0_Stride, Layout::ND>;
  arg0_Tensor g_arg0(arg0_ptr);

  auto* arg1_ptr = (half*)arg1;
  using arg1_Shape = Shape<1, 1, 1, 256, 256>;
  using arg1_Stride = Stride<1, 1, 1, 256, 1>;
  using arg1_Tensor = GlobalTensor<half, arg1_Shape, arg1_Stride, Layout::ND>;
  arg1_Tensor g_arg1(arg1_ptr);

  auto* arg2_ptr = (float*)arg2;
  using arg2_Shape = Shape<1, 1, 1, 256, 256>;
  using arg2_Stride = Stride<1, 1, 1, 256, 1>;
  using arg2_Tensor = GlobalTensor<float, arg2_Shape, arg2_Stride, Layout::ND>;
  arg2_Tensor g_arg2(arg2_ptr);

  using a_mat_Tile = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null>;
  a_mat_Tile t_a_mat;
  using b_mat_Tile = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null>;
  b_mat_Tile t_b_mat;
  using a_left_0_Tile = Tile<TileType::Left, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null>;
  a_left_0_Tile t_a_left_0;
  using a_left_1_Tile = Tile<TileType::Left, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null>;
  a_left_1_Tile t_a_left_1;
  using b_right_0_Tile = Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null>;
  b_right_0_Tile t_b_right_0;
  using b_right_1_Tile = Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null>;
  b_right_1_Tile t_b_right_1;
  using c_acc_Tile = Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024, PadValue::Null>;
  c_acc_Tile t_c_acc;

  TASSIGN(t_a_mat, 0x0);
  TASSIGN(t_b_mat, 0x20000);
  TASSIGN(t_a_left_0, 0x0);
  TASSIGN(t_a_left_1, 0x8000);
  TASSIGN(t_b_right_0, 0x0);
  TASSIGN(t_b_right_1, 0x8000);
  TASSIGN(t_c_acc, 0x0);

  // prologue
  for (int64_t mi = 0; mi < 256; mi += 16) {
    for (int64_t nj = 0; nj < 256; nj += 16) {
      for (int64_t kk = 0; kk < 256; kk += 16) {
        // NOTE: tload with indices uses a tile-shaped GlobalTensor view for conversion correctness.
        {
          auto* g_arg0_ptr = g_arg0.data();
          auto g_arg0_off = (mi) * g_arg0.GetStride(GlobalTensorDim::DIM_3) + (kk) * g_arg0.GetStride(GlobalTensorDim::DIM_4);
          using TloadShape = Shape<1, 1, 1, 16, 16>;
          using TloadTensor = GlobalTensor<half, TloadShape, arg0_Stride, Layout::ND>;
          TloadTensor g_arg0_view(g_arg0_ptr);
          TASSIGN(g_arg0_view, g_arg0_ptr + g_arg0_off);
          TLOAD(t_a_mat, g_arg0_view);
        }
        // NOTE: tload with indices uses a tile-shaped GlobalTensor view for conversion correctness.
        {
          auto* g_arg1_ptr = g_arg1.data();
          auto g_arg1_off = (kk) * g_arg1.GetStride(GlobalTensorDim::DIM_3) + (nj) * g_arg1.GetStride(GlobalTensorDim::DIM_4);
          using TloadShape = Shape<1, 1, 1, 16, 16>;
          using TloadTensor = GlobalTensor<half, TloadShape, arg1_Stride, Layout::ND>;
          TloadTensor g_arg1_view(g_arg1_ptr);
          TASSIGN(g_arg1_view, g_arg1_ptr + g_arg1_off);
          TLOAD(t_b_mat, g_arg1_view);
        }
        auto it0 = (kk) / (16);
        auto lane = (it0) % (2);
        auto t1 = (lane) == (0);
        if (t1) {
          TMOV(t_a_left_0, t_a_mat);
          TMOV(t_b_right_0, t_b_mat);
          auto t2 = (kk) == (0);
          if (t2) {
            TMATMUL(t_c_acc, t_a_left_0, t_b_right_0);
          } else {
            TMATMUL_ACC(t_c_acc, t_c_acc, t_a_left_0, t_b_right_0);
          }
        } else {
          TMOV(t_a_left_1, t_a_mat);
          TMOV(t_b_right_1, t_b_mat);
          TMATMUL_ACC(t_c_acc, t_c_acc, t_a_left_1, t_b_right_1);
        }
      }
      // NOTE: tstore with indices uses a tile-shaped GlobalTensor view for conversion correctness.
      {
        auto* g_arg2_ptr = g_arg2.data();
        auto g_arg2_off = (mi) * g_arg2.GetStride(GlobalTensorDim::DIM_3) + (nj) * g_arg2.GetStride(GlobalTensorDim::DIM_4);
        using TstoreShape = Shape<1, 1, 1, 16, 16>;
        using TstoreTensor = GlobalTensor<float, TstoreShape, arg2_Stride, Layout::ND>;
        TstoreTensor g_arg2_view(g_arg2_ptr);
        TASSIGN(g_arg2_view, g_arg2_ptr + g_arg2_off);
        TSTORE(g_arg2_view, t_c_acc);
      }
    }
  }
  // epilogue
}
```

</details>

## 阶段 E：PTO-AS → CCE C++（NPU 版本：`.cce.cpp`）

### 这一步工具做了什么

`ptoas --target npu` 会生成可用 `bisheng -xcce` 编译的 CCE C++ 源码：

- 入口形如（本例为）`extern "C" __global__ AICORE void pto_kernel_gemm256(GM_ADDR arg0, GM_ADDR arg1, GM_ADDR arg2)`
- 类型使用 `__gm__` / `GM_ADDR` / `kernel_operator.h`
- 同样把 `.pto` 的 tile 指令映射为 `TLOAD/TMOV/TMATMUL_ACC/TSTORE`

注意：生成 `.cce.cpp` 本身不要求你机器上真的有 Ascend toolkit（只要不启用 `--emit-bin`）。但后续要编译/运行 NPU，则需要 toolkit。

### 如何生成

```bash
$PTOAS "$OUT/gemm256.pto" \
  --target npu \
  -o "$OUT/gemm256.cce.cpp" \
  --kernel-name pto_kernel_gemm256 \
  --arch dav-c220-cube \
  --memory-model MEMORY_BASE \
  --repo-root "$PWD"
```

### 输出（`$OUT/gemm256.cce.cpp`，完整内容）

<details>
<summary>点击展开：完整 gemm256.cce.cpp</summary>

```cpp
// Generated by ptoas (mlir prototype)
#if defined(__CCE_AICORE__)
#define MEMORY_BASE
#include "kernel_operator.h"
#include <pto/pto-inst.hpp>
#include <cstdint>
using namespace pto;

extern "C" __global__ AICORE void pto_kernel_gemm256(GM_ADDR arg0, GM_ADDR arg1, GM_ADDR arg2) {
  using arg0_Shape = Shape<1, 1, 1, 256, 256>;
  using arg0_Stride = Stride<1, 1, 1, 256, 1>;
  using arg0_Tensor = GlobalTensor<half, arg0_Shape, arg0_Stride, Layout::ND>;
  arg0_Tensor g_arg0((__gm__ half*)arg0);

  using arg1_Shape = Shape<1, 1, 1, 256, 256>;
  using arg1_Stride = Stride<1, 1, 1, 256, 1>;
  using arg1_Tensor = GlobalTensor<half, arg1_Shape, arg1_Stride, Layout::ND>;
  arg1_Tensor g_arg1((__gm__ half*)arg1);

  using arg2_Shape = Shape<1, 1, 1, 256, 256>;
  using arg2_Stride = Stride<1, 1, 1, 256, 1>;
  using arg2_Tensor = GlobalTensor<float, arg2_Shape, arg2_Stride, Layout::ND>;
  arg2_Tensor g_arg2((__gm__ float*)arg2);

  using a_mat_Tile = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null>;
  a_mat_Tile t_a_mat;
  using b_mat_Tile = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null>;
  b_mat_Tile t_b_mat;
  using a_left_0_Tile = Tile<TileType::Left, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null>;
  a_left_0_Tile t_a_left_0;
  using a_left_1_Tile = Tile<TileType::Left, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null>;
  a_left_1_Tile t_a_left_1;
  using b_right_0_Tile = Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null>;
  b_right_0_Tile t_b_right_0;
  using b_right_1_Tile = Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null>;
  b_right_1_Tile t_b_right_1;
  using c_acc_Tile = Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024, PadValue::Null>;
  c_acc_Tile t_c_acc;

  TASSIGN(t_a_mat, 0x0);
  TASSIGN(t_b_mat, 0x20000);
  TASSIGN(t_a_left_0, 0x0);
  TASSIGN(t_a_left_1, 0x8000);
  TASSIGN(t_b_right_0, 0x0);
  TASSIGN(t_b_right_1, 0x8000);
  TASSIGN(t_c_acc, 0x0);

  // prologue
  for (int64_t mi = 0; mi < 256; mi += 16) {
    for (int64_t nj = 0; nj < 256; nj += 16) {
      for (int64_t kk = 0; kk < 256; kk += 16) {
        // NOTE: tload with indices uses a tile-shaped GlobalTensor view for conversion correctness.
        {
          auto* g_arg0_ptr = g_arg0.data();
          auto g_arg0_off = (mi) * g_arg0.GetStride(GlobalTensorDim::DIM_3) + (kk) * g_arg0.GetStride(GlobalTensorDim::DIM_4);
          using TloadShape = Shape<1, 1, 1, 16, 16>;
          using TloadTensor = GlobalTensor<half, TloadShape, arg0_Stride, Layout::ND>;
          TloadTensor g_arg0_view(g_arg0_ptr);
          TASSIGN(g_arg0_view, g_arg0_ptr + g_arg0_off);
          TLOAD(t_a_mat, g_arg0_view);
        }
        // NOTE: tload with indices uses a tile-shaped GlobalTensor view for conversion correctness.
        {
          auto* g_arg1_ptr = g_arg1.data();
          auto g_arg1_off = (kk) * g_arg1.GetStride(GlobalTensorDim::DIM_3) + (nj) * g_arg1.GetStride(GlobalTensorDim::DIM_4);
          using TloadShape = Shape<1, 1, 1, 16, 16>;
          using TloadTensor = GlobalTensor<half, TloadShape, arg1_Stride, Layout::ND>;
          TloadTensor g_arg1_view(g_arg1_ptr);
          TASSIGN(g_arg1_view, g_arg1_ptr + g_arg1_off);
          TLOAD(t_b_mat, g_arg1_view);
        }
        auto it0 = (kk) / (16);
        auto lane = (it0) % (2);
        auto t1 = (lane) == (0);
        if (t1) {
          TMOV(t_a_left_0, t_a_mat);
          TMOV(t_b_right_0, t_b_mat);
          auto t2 = (kk) == (0);
          if (t2) {
            TMATMUL(t_c_acc, t_a_left_0, t_b_right_0);
          } else {
            TMATMUL_ACC(t_c_acc, t_c_acc, t_a_left_0, t_b_right_0);
          }
        } else {
          TMOV(t_a_left_1, t_a_mat);
          TMOV(t_b_right_1, t_b_mat);
          TMATMUL_ACC(t_c_acc, t_c_acc, t_a_left_1, t_b_right_1);
        }
      }
      // NOTE: tstore with indices uses a tile-shaped GlobalTensor view for conversion correctness.
      {
        auto* g_arg2_ptr = g_arg2.data();
        auto g_arg2_off = (mi) * g_arg2.GetStride(GlobalTensorDim::DIM_3) + (nj) * g_arg2.GetStride(GlobalTensorDim::DIM_4);
        using TstoreShape = Shape<1, 1, 1, 16, 16>;
        using TstoreTensor = GlobalTensor<float, TstoreShape, arg2_Stride, Layout::ND>;
        TstoreTensor g_arg2_view(g_arg2_ptr);
        TASSIGN(g_arg2_view, g_arg2_ptr + g_arg2_off);
        TSTORE(g_arg2_view, t_c_acc);
      }
    }
  }
  // epilogue
}
#endif
```

</details>

## 阶段 F：生成 `.bin`（可选）—— `ptoas --emit-bin` 调用 `bisheng`

如果你希望 `ptoas` 在生成 `.cce.cpp` 的同时顺带编译出 `.bin`，用 `--emit-bin`：

```bash
$PTOAS "$OUT/gemm256.pto" \
  --target npu \
  -o "$OUT/gemm256.cce.cpp" \
  --kernel-name pto_kernel_gemm256 \
  --arch dav-c220-cube \
  --memory-model MEMORY_BASE \
  --repo-root "$PWD" \
  --ascend-home "$ASCEND_HOME_PATH" \
  --emit-bin="$OUT/gemm256.bin"
```

这一步会在 `ptoas_main.cpp` 里调用 `compileCceToBin`（驱动 `bisheng`），因此需要你本机安装好 toolkit 并且 `ASCEND_HOME_PATH` 正确。

## 阶段 G：生成 fatobj `.so`（用于 Python/ACL 统一加载与 launch）

### 这一步工具做了什么

为了让 Python 能用 `ctypes` 统一调用，我们会把 kernel 包进一个 `.so`，并提供统一入口：

- `extern "C" void ptoas_launch(void *stream, uint32_t blockDim, void *arg0, void *arg1, void *arg2)`

`ptoas.python.pipeline.build_fatobj_so_from_cce`（源码在 `binding/python/ptoas/python/pipeline.py`）会：

1) 读取 `gemm256.cce.cpp`
2) 生成一个 wrapper `combined.cpp`，包含 `ptoas_launch`，并在里面 launch 你生成的 kernel
3) 用 `bisheng -xcce` 编译
4) 用 `bisheng --cce-fatobj-link` 链接成 `.so`

### 如何生成

```bash
python3 - <<'PY'
import os
from pathlib import Path
from ptoas.python import pipeline

outdir = Path(os.environ["OUT"])
ascend_home = Path(os.environ["ASCEND_HOME_PATH"])

pipeline.build_fatobj_so_from_cce(
    cce_path=outdir / "gemm256.cce.cpp",
    out_so=outdir / "libgemm256_npu.so",
    arch="dav-c220-cube",
    ascend_home=ascend_home,
)
print("wrote:", outdir / "libgemm256_npu.so")
PY
```

### Wrapper 代码（`combined.cpp`，完整内容）

`build_fatobj_so_from_cce` 会把 “kernel 源码” 写到 `kernel.cpp`，同时生成一个固定入口 `ptoas_launch` 的 wrapper（这里展示其内容形状；对本例来说 kernel 名称和参数个数是确定的）：

```cpp
#include "kernel.cpp"
#include <cstdint>

extern "C" void ptoas_launch(void *stream, uint32_t blockDim, void *arg0, void *arg1, void *arg2)
{
    pto_kernel_gemm256<<<blockDim, nullptr, stream>>>((GM_ADDR)arg0, (GM_ADDR)arg1, (GM_ADDR)arg2);
}
```

## 阶段 H：运行（NPU 或模拟器）

推荐直接用 end-to-end runner：它会自动生成中间产物，并做 CPU vs NPU 结果对比。

NPU：

```bash
python3 kernels/python/run_gemm256.py \
  --run-mode npu \
  --ascend-home "$ASCEND_HOME_PATH" \
  --ptoas "$PTOAS" \
  --outdir "$OUT" \
  --no-insert-events
```

模拟器：

```bash
python3 kernels/python/run_gemm256.py \
  --run-mode sim \
  --soc a5 \
  --ascend-home "$ASCEND_HOME_PATH" \
  --ptoas "$PTOAS" \
  --outdir "$OUT" \
  --no-insert-events
```

如果你只想“手动跑 `.so`”：

```bash
python3 - <<'PY'
import os
from pathlib import Path
from ptoas.python import pipeline

outdir = Path(os.environ["OUT"])
pto_text = (outdir / "gemm256.pto").read_text(encoding="utf-8")
host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
host_arrays = pipeline.make_host_arrays(host_spec)

res = pipeline.run_npu_kernel_from_so(
    so_path=outdir / "libgemm256_npu.so",
    host_spec=host_spec,
    host_arrays=host_arrays,
    device_id=0,
    block_dim=1,
)
out = res.outputs
print("output:", out[0].shape, out[0].dtype)
PY
```

## 附录：end-to-end 脚本（`kernels/python/run_gemm256.py`，完整内容）

<details>
<summary>点击展开：完整 run_gemm256.py</summary>

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import binding, pipeline  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402


def _default_ptoas(repo: Path) -> Path:
    for p in (
        repo / "ptoas/mlir/build-macos/bin/ptoas",
        repo / "ptoas/mlir/build/bin/ptoas",
    ):
        if p.exists():
            return p
    return repo / "ptoas/mlir/build/bin/ptoas"


def _soc_from_alias(alias: str) -> str:
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="Run GEMM 256x256x256 (tiled 16x16) end-to-end.")
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC alias when --run-mode=sim (a3|a5|other)")
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas(repo))
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_gemm256"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    ap.add_argument("--verbose-build", action="store_true", help="Print compiler commands/warnings")
    args = ap.parse_args()

    if not args.verbose_build:
        os.environ.setdefault("PTOAS_QUIET", "1")

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    py = Path(__file__).resolve().with_name("gemm256.py")
    spec = binding.compile_file(py, kernel="gemm256")
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))

    args.outdir.mkdir(parents=True, exist_ok=True)
    pto_path = args.outdir / f"{spec.name}.pto"
    pto_path.write_text(pto_text, encoding="utf-8")

    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=args.block_dim, kernel_name=host_spec.kernel_name
    )
    base_arrays = pipeline.make_host_arrays(host_spec)

    # CPU reference.
    cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
    cpu_so = args.outdir / f"lib{spec.name}_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
    cpu_arrays = [a.copy() for a in base_arrays]
    cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

    # NPU run (sim or real).
    if args.run_mode == "sim":
        pipeline.configure_ascend_sim_env(ascend_home=args.ascend_home, soc=_soc_from_alias(args.soc))

    cfg = pipeline.CompileConfig(
        ptoas=args.ptoas,
        ascend_home=args.ascend_home,
        arch="dav-c220-cube",
        memory_model=args.memory_model,
        insert_events=args.insert_events,
    )
    cce_cpp, _bin = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg)
    npu_so = args.outdir / f"lib{spec.name}_{args.run_mode}.so"
    pipeline.build_fatobj_so_from_cce(cce_path=cce_cpp, out_so=npu_so, arch=cfg.arch, ascend_home=cfg.ascend_home)

    npu_arrays = [a.copy() for a in base_arrays]
    npu_res = pipeline.run_npu_kernel_from_so(
        so_path=npu_so, host_spec=host_spec, host_arrays=npu_arrays, device_id=args.device, block_dim=args.block_dim
    )
    npu_out = npu_res.outputs

    out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
    pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)
    for a in npu_out:
        if a.dtype in (np.float16, np.float32):
            print("OK (max abs):", float(np.max(np.abs(a))))
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

</details>

## 一眼看到所有中间产物

```bash
ls -lh "$OUT"
```

你应该至少能看到（视你执行了哪些步骤而定）：

- `gemm256.pto`
- `gemm256.cpu.cpp`
- `gemm256.cce.cpp`
- `gemm256.bin`
- `libgemm256_npu.so`
