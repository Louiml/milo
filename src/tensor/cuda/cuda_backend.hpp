// The CUDA backend's interface.
#ifndef MILO_TENSOR_CUDA_CUDA_BACKEND_HPP
#define MILO_TENSOR_CUDA_CUDA_BACKEND_HPP

#include <cstdint>
#include <string>

#include "tensor/backend.hpp"

namespace milo::tensor::cuda {

/// The kernels, as CUDA code.
//
// They are declared here so that the backend can call them without the
// kernel file having to know about C++ classes.
void cudaFillKernel(float* data, std::size_t count, float value);
void cudaAddKernel(const float* a, const float* b, float* out, std::size_t count);
void cudaSubKernel(const float* a, const float* b, float* out, std::size_t count);
void cudaMulKernel(const float* a, const float* b, float* out, std::size_t count);
void cudaDivKernel(const float* a, const float* b, float* out, std::size_t count);
void cudaMatmulKernel(const float* a, const float* b, float* out, int m, int k, int n);
void cudaSumKernel(const float* data, float* partials, std::size_t count);
void cudaSumFinalKernel(const float* partials, float* total, std::size_t count);

/// A GPU that Milo can put tensors on.
class CudaBackend : public Backend {
 public:
  /// Open a device. Throws when it cannot be reached.
  explicit CudaBackend(int device = 0);

  const char* name() const override;
  Device device() const override;
  bool supports(DType dtype) const override;

  void* allocate(std::size_t bytes) override;
  void deallocate(void* data) override;
  void copyToDevice(void* destination, const void* source, std::size_t bytes) override;
  void copyToHost(void* destination, const void* source, std::size_t bytes) override;
  void copyBetween(void* destination, const void* source, std::size_t bytes) override;
  void synchronize() override;
  std::uint64_t freeMemory() const override;
  std::uint64_t totalMemory() const override;

  void fill(float* data, std::size_t count, float value) override;
  void elementwise(BackendOp op, const float* a, const float* b, float* out,
                   std::size_t count) override;
  void matmul(const float* a, const float* b, float* out, std::int64_t m, std::int64_t k,
              std::int64_t n) override;
  double sum(const float* data, std::size_t count) override;

  /// The device's compute capability, e.g. "8.6".
  const std::string& computeCapability() const;

 private:
  int device_ = 0;
  std::string name_;
  std::string computeCapability_;
  std::uint64_t totalMemory_ = 0;
};

/// How many GPUs this machine has.
int deviceCount();

/// Is there a GPU to talk to?
bool isAvailable();

}  // namespace milo::tensor::cuda

#endif  // MILO_TENSOR_CUDA_CUDA_BACKEND_HPP
