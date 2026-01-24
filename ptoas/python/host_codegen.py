from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    # `dtype` is the PTO-AS element spelling ("f16", "f32", "i32", ...).
    dtype: str
    shape: tuple[int, int]

    @property
    def nbytes(self) -> int:
        h, w = self.shape
        if self.dtype == "f16":
            elem = 2
        elif self.dtype == "f32":
            elem = 4
        elif self.dtype == "i32" or self.dtype == "u32":
            elem = 4
        else:
            raise ValueError(f"unsupported dtype for host codegen: {self.dtype}")
        return int(h) * int(w) * elem


def emit_acl_host_cpp(*, so_basename: str, args: list[TensorSpec]) -> str:
    """
    Emit a small C++ host launcher that:
      - loads `so_basename` via dlopen()
      - allocates device buffers for each arg
      - launches `ptoas_launch(stream, blockDim, arg0, arg1, ...)`
      - copies back the last argument buffer

    This is meant to be compiled on a Linux machine with Ascend CANN installed.
    """
    if not args:
        raise ValueError("args must be non-empty")

    n = len(args)
    arg_params = ", ".join(["void *stream", "uint32_t blockDim"] + [f"void *arg{i}" for i in range(n)])
    dlsym_cast = f"reinterpret_cast<launch_fn_t>(dlsym(handle, \"ptoas_launch\"))"
    dev_decls = "\n".join([f"  void* d{i} = nullptr;" for i in range(n)])
    host_decls = "\n".join([f"  void* h{i} = nullptr;" for i in range(n)])
    sizes = [a.nbytes for a in args]
    size_consts = "\n".join([f"  constexpr size_t kArg{i}Bytes = {sizes[i]};" for i in range(n)])

    malloc_host = "\n".join([f"  ACL_CHECK(aclrtMallocHost(&h{i}, kArg{i}Bytes));" for i in range(n)])
    malloc_dev = "\n".join(
        [f"  ACL_CHECK(aclrtMalloc(&d{i}, kArg{i}Bytes, ACL_MEM_MALLOC_HUGE_FIRST));" for i in range(n)]
    )
    # Initialize inputs (all but the last arg).
    init_inputs = "\n".join(
        [f"  Fill(h{i}, kArg{i}Bytes, {i + 1});" for i in range(n - 1)]
        + [f"  std::memset(h{n - 1}, 0, kArg{n - 1}Bytes);"]
    )
    h2d = "\n".join(
        [f"  ACL_CHECK(aclrtMemcpy(d{i}, kArg{i}Bytes, h{i}, kArg{i}Bytes, ACL_MEMCPY_HOST_TO_DEVICE));" for i in range(n)]
    )
    launch_args = ", ".join(["stream", "blockDim"] + [f"d{i}" for i in range(n)])
    d2h = f"  ACL_CHECK(aclrtMemcpy(h{n - 1}, kArg{n - 1}Bytes, d{n - 1}, kArg{n - 1}Bytes, ACL_MEMCPY_DEVICE_TO_HOST));"
    free_dev = "\n".join([f"  (void)aclrtFree(d{i});" for i in range(n)])
    free_host = "\n".join([f"  (void)aclrtFreeHost(h{i});" for i in range(n)])

    return f"""
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>

static void Die(const char* msg, int32_t code) {{
  std::fprintf(stderr, "error: %s (ret=%d)\\n", msg, code);
  std::exit(2);
}}

#define ACL_CHECK(x) do {{ \\
  int32_t _ret = (x); \\
  if (_ret != 0) Die(#x, _ret); \\
}} while (0)

static void Fill(void* p, size_t n, uint8_t seed) {{
  auto* b = static_cast<uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) {{
    seed = static_cast<uint8_t>(seed * 131u + 7u);
    b[i] = seed;
  }}
}}

int main(int argc, char** argv) {{
  int device = 0;
  uint32_t blockDim = 1;
  std::string soPath = "{so_basename}";

  for (int i = 1; i < argc; ++i) {{
    if (!std::strcmp(argv[i], "--device") && i + 1 < argc) {{
      device = std::atoi(argv[++i]);
      continue;
    }}
    if (!std::strcmp(argv[i], "--block-dim") && i + 1 < argc) {{
      blockDim = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
      continue;
    }}
    if (!std::strcmp(argv[i], "--so") && i + 1 < argc) {{
      soPath = argv[++i];
      continue;
    }}
    std::fprintf(stderr, "usage: %s [--so libfoo.so] [--device 0] [--block-dim 1]\\n", argv[0]);
    return 2;
  }}

  using launch_fn_t = void(*)(void*, uint32_t{''.join([', void*' for _ in range(n)])});

  void* handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {{
    std::fprintf(stderr, "error: dlopen(%s): %s\\n", soPath.c_str(), dlerror());
    return 2;
  }}

  auto launch = {dlsym_cast};
  if (!launch) {{
    std::fprintf(stderr, "error: dlsym(ptoas_launch): %s\\n", dlerror());
    return 2;
  }}

{size_consts}

{host_decls}
{dev_decls}

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(device));
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

{malloc_host}
{malloc_dev}
{init_inputs}
{h2d}

  launch({launch_args});
  ACL_CHECK(aclrtSynchronizeStream(stream));
{d2h}

  std::printf("OK: launched kernel (copied back %zu bytes from last arg)\\n", kArg{n - 1}Bytes);

{free_dev}
{free_host}
  (void)aclrtDestroyStream(stream);
  (void)aclrtResetDevice(device);
  (void)aclFinalize();
  (void)dlclose(handle);
  return 0;
}}
""".lstrip()


def emit_acl_host_cpp_static(*, args: list[TensorSpec]) -> str:
    """
    Emit a small C++ host launcher that:
      - links against a fatobj object that provides `ptoas_launch`
      - allocates device buffers for each arg
      - launches `ptoas_launch(stream, blockDim, arg0, arg1, ...)`
      - optionally times via ACL events
      - copies back the last argument buffer

    This is meant to be compiled on a Linux machine with Ascend CANN installed.
    """
    if not args:
        raise ValueError("args must be non-empty")

    n = len(args)
    dev_decls = "\n".join([f"  void* d{i} = nullptr;" for i in range(n)])
    host_decls = "\n".join([f"  void* h{i} = nullptr;" for i in range(n)])
    sizes = [a.nbytes for a in args]
    size_consts = "\n".join([f"  constexpr size_t kArg{i}Bytes = {sizes[i]};" for i in range(n)])

    malloc_host = "\n".join([f"  ACL_CHECK(aclrtMallocHost(&h{i}, kArg{i}Bytes));" for i in range(n)])
    malloc_dev = "\n".join(
        [f"  ACL_CHECK(aclrtMalloc(&d{i}, kArg{i}Bytes, ACL_MEM_MALLOC_HUGE_FIRST));" for i in range(n)]
    )
    init_inputs = "\n".join(
        [f"  Fill(h{i}, kArg{i}Bytes, {i + 1});" for i in range(n - 1)]
        + [f"  std::memset(h{n - 1}, 0, kArg{n - 1}Bytes);"]
    )
    h2d = "\n".join(
        [f"  ACL_CHECK(aclrtMemcpy(d{i}, kArg{i}Bytes, h{i}, kArg{i}Bytes, ACL_MEMCPY_HOST_TO_DEVICE));" for i in range(n)]
    )
    launch_args = ", ".join(["stream", "blockDim"] + [f"d{i}" for i in range(n)])
    d2h = f"  ACL_CHECK(aclrtMemcpy(h{n - 1}, kArg{n - 1}Bytes, d{n - 1}, kArg{n - 1}Bytes, ACL_MEMCPY_DEVICE_TO_HOST));"
    free_dev = "\n".join([f"  (void)aclrtFree(d{i});" for i in range(n)])
    free_host = "\n".join([f"  (void)aclrtFreeHost(h{i});" for i in range(n)])

    return f"""
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void ptoas_launch(void* stream, uint32_t blockDim{''.join([', void*' for _ in range(n)])});

static void Die(const char* msg, int32_t code) {{
  std::fprintf(stderr, "error: %s (ret=%d)\\n", msg, code);
  std::exit(2);
}}

#define ACL_CHECK(x) do {{ \\
  int32_t _ret = (x); \\
  if (_ret != 0) Die(#x, _ret); \\
}} while (0)

static void Fill(void* p, size_t n, uint8_t seed) {{
  auto* b = static_cast<uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) {{
    seed = static_cast<uint8_t>(seed * 131u + 7u);
    b[i] = seed;
  }}
}}

int main(int argc, char** argv) {{
  int device = 0;
  uint32_t blockDim = 1;
  int iters = 1;
  int warmup = 0;

  for (int i = 1; i < argc; ++i) {{
    if (!std::strcmp(argv[i], "--device") && i + 1 < argc) {{
      device = std::atoi(argv[++i]);
      continue;
    }}
    if (!std::strcmp(argv[i], "--block-dim") && i + 1 < argc) {{
      blockDim = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
      continue;
    }}
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {{
      iters = std::atoi(argv[++i]);
      continue;
    }}
    if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {{
      warmup = std::atoi(argv[++i]);
      continue;
    }}
    std::fprintf(stderr, "usage: %s [--device 0] [--block-dim 1] [--warmup 0] [--iters 1]\\n", argv[0]);
    return 2;
  }}

{size_consts}

{host_decls}
{dev_decls}

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(device));
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

{malloc_host}
{malloc_dev}
{init_inputs}
{h2d}

  for (int i = 0; i < warmup; ++i) {{
    ptoas_launch({launch_args});
  }}
  ACL_CHECK(aclrtSynchronizeStream(stream));

  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  ACL_CHECK(aclrtCreateEvent(&start));
  ACL_CHECK(aclrtCreateEvent(&end));
  ACL_CHECK(aclrtRecordEvent(start, stream));
  for (int i = 0; i < iters; ++i) {{
    ptoas_launch({launch_args});
  }}
  ACL_CHECK(aclrtRecordEvent(end, stream));
  ACL_CHECK(aclrtSynchronizeEvent(end));
  float elapsedMs = 0.0f;
  ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, start, end));
  float avgMs = elapsedMs / static_cast<float>(iters > 0 ? iters : 1);

  ACL_CHECK(aclrtDestroyEvent(start));
  ACL_CHECK(aclrtDestroyEvent(end));

{d2h}

  std::printf("avg_time_ms: %.4f\\n", static_cast<double>(avgMs));
  std::printf("OK: launched kernel (copied back %zu bytes from last arg)\\n", kArg{n - 1}Bytes);

{free_dev}
{free_host}
  (void)aclrtDestroyStream(stream);
  (void)aclrtResetDevice(device);
  (void)aclFinalize();
  return 0;
}}
""".lstrip()
