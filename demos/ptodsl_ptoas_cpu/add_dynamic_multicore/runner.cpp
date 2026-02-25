#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <string>
#include <algorithm>

#include "pto/common/cpu_stub.hpp"

extern void vec_add_1d_dynamic(float *v1, float *v2, float *v3, int32_t n);

static bool almost_equal(float a, float b, float eps = 1e-5f) {
  float diff = std::fabs(a - b);
  float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
  return diff <= eps * scale;
}

int main() {
  // Default launch topology (matches PTODSL multicore examples style).
  const int64_t blocks = 20;
  const int64_t subblocks = 2;

  // Pick a length that is NOT a multiple of tile_length(1024).
  // NOTE: this PTODSL example uses full-tile TLOAD/TSTORE (1024 elems) even for the last tile.
  // On real hardware, tensors are typically padded/allocated with enough tail space.
  // For CPU simulator we allocate padded buffers to avoid OOB.
  const int32_t n = 1024 * 123 + 99;
  const int32_t padded_n = ((n + 1024 - 1) / 1024) * 1024;

  std::vector<float> x(padded_n), y(padded_n), z(padded_n, 0.0f), ref(n);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < padded_n; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
  }
  for (int i = 0; i < n; ++i) {
    ref[i] = x[i] + y[i];
  }

  const char* launch_mode = std::getenv("PTO_CPU_LAUNCH");
  const char* max_threads_env = std::getenv("PTO_CPU_MAX_THREADS");
  int max_threads = 0;
  if (max_threads_env && *max_threads_env) {
    max_threads = std::atoi(max_threads_env);
  }

  auto kernel = +[](float* a, float* b, float* c, int32_t n) {
    vec_add_1d_dynamic(a, b, c, n);
  };

  if (launch_mode && std::string(launch_mode) == "sequential") {
    pto::cpu_sim::launch_sequential(blocks, subblocks, kernel, x.data(), y.data(), z.data(), n);
  } else {
    pto::cpu_sim::launch_parallel(blocks, subblocks, kernel, max_threads, x.data(), y.data(), z.data(), n);
  }

  for (int i = 0; i < n; ++i) {
    if (!almost_equal(z[i], ref[i])) {
      std::fprintf(stderr, "Mismatch at %d: got=%f ref=%f\n", i, z[i], ref[i]);
      return 1;
    }
  }

  std::puts("PASS: CPU-sim vec_add_1d_dynamic (multicore)");
  return 0;
}
