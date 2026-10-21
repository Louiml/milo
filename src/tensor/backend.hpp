// The backend interface.
//
// The tensor engine does not know how a tensor's memory is stored. It asks
// a backend, and the backend answers. The CPU is one backend; CUDA is
// another; a Metal or Vulkan backend would be a third.
//
// Everything here is deliberately narrow: the interface is small enough
// that a new backend is a day's work, and the operations are the ones
// that actually cost something.
#ifndef MILO_TENSOR_BACKEND_HPP
#define MILO_TENSOR_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensor/types.hpp"

namespace milo::tensor {

/// The operations a backend has to provide.
enum class BackendOp {
  Add,
  Sub,
  Mul,
  Div,
};

/// A place where tensors can live, and the things that can be done there.
class Backend {
 public:
  virtual ~Backend() = default;

  /// The backend's name, for diagnostics: "cpu", "cuda".
  virtual const char* name() const = 0;

  /// The device this backend serves.
  virtual Device device() const = 0;

  /// Can this backend hold this element type?
  virtual bool supports(DType dtype) const = 0;

  /// Allocate memory on the device. Returns nullptr when it cannot.
  virtual void* allocate(std::size_t bytes) = 0;

  /// Give memory back.
  virtual void deallocate(void* data) = 0;

  /// Copy from host memory to the device.
  virtual void copyToDevice(void* destination, const void* source, std::size_t bytes) = 0;

  /// Copy from the device to host memory.
  virtual void copyToHost(void* destination, const void* source, std::size_t bytes) = 0;

  /// Copy from one device to another.
  virtual void copyBetween(void* destination, const void* source, std::size_t bytes) = 0;

  /// Wait for everything queued to finish.
  virtual void synchronize() = 0;

  /// How much memory is free on the device, in bytes.
  virtual std::uint64_t freeMemory() const = 0;

  /// How much memory the device has, in bytes.
  virtual std::uint64_t totalMemory() const = 0;

  // --- the kernels -------------------------------------------------------

  /// Fill a buffer with a number.
  virtual void fill(float* data, std::size_t count, float value) = 0;

  /// An elementwise operation on two f32 buffers.
  virtual void elementwise(BackendOp op, const float* a, const float* b, float* out,
                           std::size_t count) = 0;

  /// A matrix product: `[m, k] x [k, n] -> [m, n]`.
  virtual void matmul(const float* a, const float* b, float* out, std::int64_t m, std::int64_t k,
                      std::int64_t n) = 0;

  /// The sum of a buffer.
  virtual double sum(const float* data, std::size_t count) = 0;
};

/// The backend for a device.
///
/// The CPU backend is always there. The CUDA backend is registered when the
/// machine can reach a GPU.
Backend* backendFor(Device device);

/// The backends this build knows about, for diagnostics.
std::vector<Backend*> availableBackends();

}  // namespace milo::tensor

#endif  // MILO_TENSOR_BACKEND_HPP
