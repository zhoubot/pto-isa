#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <string>

#include "pto/common/cpu_stub.hpp"

extern void RunTMATMULSplitK(float *out, float *a, float *b, float *bias, bool isBias, int32_t batch);

static bool almost_equal(float a, float b, float eps = 1e-3f) {
  float diff = std::fabs(a - b);
  float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
  return diff <= eps * scale;
}

int main() {
  constexpr int M = 32;
  constexpr int K = 64;
  constexpr int N = 32;
  const int32_t batch = 8;

  const int64_t blocks = 4;
  const int64_t subblocks = 1;

  std::vector<float> A(batch * M * K);
  std::vector<float> B(K * N);
  std::vector<float> bias(N);
  std::vector<float> C(batch * M * N, 0.0f);
  std::vector<float> ref(batch * M * N, 0.0f);

  std::mt19937 rng(4);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto &v : A) v = dist(rng);
  for (auto &v : B) v = dist(rng);
  for (auto &v : bias) v = dist(rng);

  for (int b = 0; b < batch; ++b) {
    const float* Ab = A.data() + b * M * K;
    float* Cb = ref.data() + b * M * N;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float acc = bias[j];
        for (int k = 0; k < K; ++k) {
          acc += Ab[i * K + k] * B[k * N + j];
        }
        Cb[i * N + j] = acc;
      }
    }
  }

  const char* launch_mode = std::getenv("PTO_CPU_LAUNCH");
  const char* max_threads_env = std::getenv("PTO_CPU_MAX_THREADS");
  int max_threads = 0;
  if (max_threads_env && *max_threads_env) {
    max_threads = std::atoi(max_threads_env);
  }

  auto kernel = +[](float* out, float* a, float* b, float* bias, bool isBias, int32_t batch) {
    RunTMATMULSplitK(out, a, b, bias, isBias, batch);
  };

  if (launch_mode && std::string(launch_mode) == "sequential") {
    pto::cpu_sim::launch_sequential(blocks, subblocks, kernel, C.data(), A.data(), B.data(), bias.data(), true, batch);
  } else {
    pto::cpu_sim::launch_parallel(blocks, subblocks, kernel, max_threads, C.data(), A.data(), B.data(), bias.data(), true, batch);
  }

  for (int i = 0; i < batch * M * N; ++i) {
    if (!almost_equal(C[i], ref[i])) {
      std::fprintf(stderr, "Mismatch at %d: got=%f ref=%f\n", i, C[i], ref[i]);
      return 1;
    }
  }

  std::puts("PASS: CPU-sim RunTMATMULSplitK (dynbatch, multicore)");
  return 0;
}
