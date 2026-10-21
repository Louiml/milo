// The scalar kernels: the reference every other path is checked against.
#include "tensor/kernels.hpp"

namespace milo::tensor::kernels {
namespace {

void addF32(const float* a, const float* b, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void subF32(const float* a, const float* b, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

void mulF32(const float* a, const float* b, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

void divF32(const float* a, const float* b, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
}

void scaleF32(const float* a, float scalar, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * scalar;
}

void axpyF32(float scalar, const float* a, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] += scalar * a[i];
}

float dotF32(const float* a, const float* b, std::size_t n) {
  float total = 0.0f;
  for (std::size_t i = 0; i < n; ++i) total += a[i] * b[i];
  return total;
}

}  // namespace

const Table& scalarTable() {
  static const Table table = {&addF32, &subF32, &mulF32, &divF32, &scaleF32, &axpyF32, &dotF32};
  return table;
}

}  // namespace milo::tensor::kernels
