// CPU kernels.
//
// Every kernel has a scalar implementation and, where it is worth it, a
// vectorised one. The vectorised versions live in their own translation
// units, compiled with the right instruction set, and are chosen at run
// time — so a binary built on one machine still runs on an older one.
#ifndef MILO_TENSOR_KERNELS_HPP
#define MILO_TENSOR_KERNELS_HPP

#include <cstddef>
#include <cstdint>

namespace milo::tensor::kernels {

/// Which instruction set the running CPU supports.
enum class Isa {
  Scalar,
  Avx2,
  Neon,
};

/// The name of the active instruction set, for diagnostics.
const char* isaName(Isa isa);

/// The instruction set this process is using.
Isa activeIsa();

/// Is AVX2 available on this machine?
bool hasAvx2();

/// Is NEON available on this machine?
bool hasNeon();

/// The functions the tensor engine calls.
struct Table {
  void (*addF32)(const float* a, const float* b, float* out, std::size_t n);
  void (*subF32)(const float* a, const float* b, float* out, std::size_t n);
  void (*mulF32)(const float* a, const float* b, float* out, std::size_t n);
  void (*divF32)(const float* a, const float* b, float* out, std::size_t n);
  void (*scaleF32)(const float* a, float scalar, float* out, std::size_t n);
  void (*axpyF32)(float scalar, const float* a, float* out, std::size_t n);
  float (*dotF32)(const float* a, const float* b, std::size_t n);
};

/// The kernel table for this machine.
const Table& table();

}  // namespace milo::tensor::kernels

#endif  // MILO_TENSOR_KERNELS_HPP
