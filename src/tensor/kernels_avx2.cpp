// The AVX2 kernels.
//
// This file is compiled with AVX2 enabled, and is only ever called after
// the run-time check in kernels.cpp has confirmed the CPU supports it.
#include <immintrin.h>

#include "tensor/kernels.hpp"

namespace milo::tensor::kernels {
namespace {

void addF32(const float* a, const float* b, float* out, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
  }
  for (; i < n; ++i) out[i] = a[i] + b[i];
}

void subF32(const float* a, const float* b, float* out, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
  }
  for (; i < n; ++i) out[i] = a[i] - b[i];
}

void mulF32(const float* a, const float* b, float* out, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
  }
  for (; i < n; ++i) out[i] = a[i] * b[i];
}

void divF32(const float* a, const float* b, float* out, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
  }
  for (; i < n; ++i) out[i] = a[i] / b[i];
}

void scaleF32(const float* a, float scalar, float* out, std::size_t n) {
  const __m256 vscalar = _mm256_set1_ps(scalar);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vscalar));
  }
  for (; i < n; ++i) out[i] = a[i] * scalar;
}

void axpyF32(float scalar, const float* a, float* out, std::size_t n) {
  const __m256 vscalar = _mm256_set1_ps(scalar);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 contribution = _mm256_mul_ps(_mm256_loadu_ps(a + i), vscalar);
    _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(out + i), contribution));
  }
  for (; i < n; ++i) out[i] += scalar * a[i];
}

float dotF32(const float* a, const float* b, std::size_t n) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();

  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    acc1 = _mm256_add_ps(
        acc1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
  }
  for (; i + 8 <= n; i += 8) {
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }

  __m256 sum = _mm256_add_ps(acc0, acc1);
  __m128 low = _mm256_castps256_ps128(sum);
  __m128 high = _mm256_extractf128_ps(sum, 1);
  low = _mm_add_ps(low, high);
  low = _mm_hadd_ps(low, low);
  low = _mm_hadd_ps(low, low);
  float total = _mm_cvtss_f32(low);

  for (; i < n; ++i) total += a[i] * b[i];
  return total;
}

}  // namespace

const Table& avx2Table() {
  static const Table table = {&addF32, &subF32, &mulF32, &divF32, &scaleF32, &axpyF32, &dotF32};
  return table;
}

}  // namespace milo::tensor::kernels
