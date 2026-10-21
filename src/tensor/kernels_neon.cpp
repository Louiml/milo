// The NEON kernels (aarch64).
//
// This file is compiled with NEON enabled, and is only ever called after
// the run-time check in kernels.cpp has confirmed the CPU supports it.
#include <arm_neon.h>

#include "tensor/kernels.hpp"

namespace milo::tensor::kernels {
namespace {

void addF32(const float* a, const float* b, float* out, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] + b[i];
}

void subF32(const float* a, const float* b, float* out, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] - b[i];
}

void mulF32(const float* a, const float* b, float* out, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] * b[i];
}

void divF32(const float* a, const float* b, float* out, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vdivq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] / b[i];
}

void scaleF32(const float* a, float scalar, float* out, std::size_t n) {
  const float32x4_t vscalar = vdupq_n_f32(scalar);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vscalar));
  }
  for (; i < n; ++i) out[i] = a[i] * scalar;
}

void axpyF32(float scalar, const float* a, float* out, std::size_t n) {
  const float32x4_t vscalar = vdupq_n_f32(scalar);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t contribution = vmulq_f32(vld1q_f32(a + i), vscalar);
    vst1q_f32(out + i, vaddq_f32(vld1q_f32(out + i), contribution));
  }
  for (; i < n; ++i) out[i] += scalar * a[i];
}

float dotF32(const float* a, const float* b, std::size_t n) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);

  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  for (; i + 4 <= n; i += 4) {
    acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
  }

  float32x2_t sum = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
  sum = vadd_f32(sum, vget_low_f32(acc1));
  sum = vadd_f32(sum, vget_high_f32(acc1));
  float total = vget_lane_f32(vpadd_f32(sum, sum), 0);

  for (; i < n; ++i) total += a[i] * b[i];
  return total;
}

}  // namespace

const Table& neonTable() {
  static const Table table = {&addF32, &subF32, &mulF32, &divF32, &scaleF32, &axpyF32, &dotF32};
  return table;
}

}  // namespace milo::tensor::kernels
