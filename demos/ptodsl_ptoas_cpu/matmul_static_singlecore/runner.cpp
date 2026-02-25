#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "pto/common/cpu_stub.hpp"

extern void RunTMATMULSplitK(float *out, float *a, float *b, float *bias, bool isBias);

static bool almost_equal(float a, float b, float eps = 1e-3f) {
  float diff = std::fabs(a - b);
  float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
  return diff <= eps * scale;
}

int main() {
  constexpr int M = 32;
  constexpr int K = 256;
  constexpr int N = 32;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> bias(N);
  std::vector<float> C(M * N, 0.0f);
  std::vector<float> ref(M * N, 0.0f);

  std::mt19937 rng(3);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto &v : A) v = dist(rng);
  for (auto &v : B) v = dist(rng);
  for (auto &v : bias) v = dist(rng);

  // Reference: C = A*B + bias (broadcast on rows)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = bias[j];
      for (int k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      ref[i * N + j] = acc;
    }
  }

  // Single-core invocation.
  pto::cpu_sim::set_grid_dim(1, 1);
  pto::cpu_sim::set_launch_context(0, 0, 1);
  RunTMATMULSplitK(C.data(), A.data(), B.data(), bias.data(), true);

  for (int i = 0; i < M * N; ++i) {
    if (!almost_equal(C[i], ref[i])) {
      std::fprintf(stderr, "Mismatch at %d: got=%f ref=%f\n", i, C[i], ref[i]);
      return 1;
    }
  }

  std::puts("PASS: CPU-sim RunTMATMULSplitK (static, singlecore)");
  return 0;
}
