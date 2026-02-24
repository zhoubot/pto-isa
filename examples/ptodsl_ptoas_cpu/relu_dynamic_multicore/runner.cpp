#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <string>
#include <algorithm>

#include "pto/common/cpu_stub.hpp"

extern void sync_kernel_dyn(float *in, float *out, int32_t n);

static bool almost_equal(float a, float b, float eps = 1e-5f) {
  float diff = std::fabs(a - b);
  float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
  return diff <= eps * scale;
}

int main() {
  const int64_t blocks = 8;
  const int64_t subblocks = 1;

  const int32_t n = 32 * 100 + 7;

  std::vector<float> x(n), y(n, 0.0f), ref(n);
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (int i = 0; i < n; ++i) {
    x[i] = dist(rng);
    ref[i] = std::max(0.0f, x[i]);
  }

  const char* launch_mode = std::getenv("PTO_CPU_LAUNCH");
  const char* max_threads_env = std::getenv("PTO_CPU_MAX_THREADS");
  int max_threads = 0;
  if (max_threads_env && *max_threads_env) {
    max_threads = std::atoi(max_threads_env);
  }

  auto kernel = +[](float* in, float* out, int32_t n) {
    sync_kernel_dyn(in, out, n);
  };

  if (launch_mode && std::string(launch_mode) == "sequential") {
    pto::cpu_sim::launch_sequential(blocks, subblocks, kernel, x.data(), y.data(), n);
  } else {
    pto::cpu_sim::launch_parallel(blocks, subblocks, kernel, max_threads, x.data(), y.data(), n);
  }

  for (int i = 0; i < n; ++i) {
    if (!almost_equal(y[i], ref[i])) {
      std::fprintf(stderr, "Mismatch at %d: got=%f ref=%f x=%f\n", i, y[i], ref[i], x[i]);
      return 1;
    }
  }

  std::puts("PASS: CPU-sim sync_kernel_dyn (relu, multicore)");
  return 0;
}
