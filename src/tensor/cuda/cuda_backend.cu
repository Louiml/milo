// The CUDA backend.
//
// It talks to the GPU through the CUDA runtime. Everything that can fail
// reports itself rather than crashing: a missing driver, a device that is
// busy, an allocation that does not fit.
#include "tensor/cuda/cuda_backend.hpp"

#include "tensor/cuda/cuda_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace milo::tensor::cuda {

namespace {

/// The number of threads in a block.
constexpr int kBlockSize = 256;

/// The number of threads along each axis of a tile.
constexpr int kTileSize = 16;

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(status));
  }
}

/// Is the CUDA runtime library on this machine?
///
/// The library is loaded on demand, so a machine without CUDA can still run
/// Milo. Asking the question first keeps the answer honest: without the
/// library there is no GPU to talk to, whatever the driver says.
bool runtimeLibraryPresent() {
#ifdef _WIN32
  HMODULE module = LoadLibraryA(MILO_CUDART_DLL);
  if (module == nullptr) return false;
  FreeLibrary(module);
  return true;
#else
  return true;
#endif
}

}  // namespace

int deviceCount() {
  if (!runtimeLibraryPresent()) return 0;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return count;
}

bool isAvailable() { return deviceCount() > 0; }

CudaBackend::CudaBackend(int device) : device_(device) {
  if (!runtimeLibraryPresent()) {
    throw std::runtime_error(std::string("CUDA: the CUDA runtime library (") + MILO_CUDART_DLL +
                             ") is not on this machine");
  }
  cudaDeviceProp properties = {};
  const cudaError_t status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA: no device ") + std::to_string(device) + ": " +
                             cudaGetErrorString(status));
  }
  name_ = std::string("cuda:") + std::to_string(device) + " (" + properties.name + ")";
  computeCapability_ = std::to_string(properties.major) + "." + std::to_string(properties.minor);
  totalMemory_ = properties.totalGlobalMem;
}

const char* CudaBackend::name() const { return name_.c_str(); }

Device CudaBackend::device() const { return Device::Cuda; }

bool CudaBackend::supports(DType dtype) const {
  // The kernels are written for f32. Everything else stays on the host.
  return dtype == DType::F32;
}

void* CudaBackend::allocate(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* data = nullptr;
  const cudaError_t status = cudaMalloc(&data, bytes);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  return data;
}

void CudaBackend::deallocate(void* data) {
  if (data != nullptr) {
    cudaFree(data);
  }
}

void CudaBackend::copyToDevice(void* destination, const void* source, std::size_t bytes) {
  if (bytes == 0) return;
  check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice), "copying to the device");
}

void CudaBackend::copyToHost(void* destination, const void* source, std::size_t bytes) {
  if (bytes == 0) return;
  check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost), "copying to the host");
}

void CudaBackend::copyBetween(void* destination, const void* source, std::size_t bytes) {
  if (bytes == 0) return;
  check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice),
        "copying between devices");
}

void CudaBackend::synchronize() { check(cudaDeviceSynchronize(), "waiting for the device"); }

std::uint64_t CudaBackend::freeMemory() const {
  std::size_t freeBytes = 0;
  std::size_t totalBytes = 0;
  if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return freeBytes;
}

std::uint64_t CudaBackend::totalMemory() const { return totalMemory_; }

const std::string& CudaBackend::computeCapability() const { return computeCapability_; }

void CudaBackend::fill(float* data, std::size_t count, float value) {
  if (count == 0) return;
  const std::size_t blocks = (count + kBlockSize - 1) / kBlockSize;
  fillKernel<<<static_cast<unsigned int>(blocks), kBlockSize>>>(data, count, value);
}

void CudaBackend::elementwise(BackendOp op, const float* a, const float* b, float* out,
                              std::size_t count) {
  if (count == 0) return;
  const std::size_t blocks = (count + kBlockSize - 1) / kBlockSize;
  const unsigned int grid = static_cast<unsigned int>(blocks);
  switch (op) {
    case BackendOp::Add:
      addKernel<<<grid, kBlockSize>>>(a, b, out, count);
      break;
    case BackendOp::Sub:
      subKernel<<<grid, kBlockSize>>>(a, b, out, count);
      break;
    case BackendOp::Mul:
      mulKernel<<<grid, kBlockSize>>>(a, b, out, count);
      break;
    case BackendOp::Div:
      divKernel<<<grid, kBlockSize>>>(a, b, out, count);
      break;
  }
}

void CudaBackend::matmul(const float* a, const float* b, float* out, std::int64_t m, std::int64_t k,
                         std::int64_t n) {
  if (m == 0 || k == 0 || n == 0) return;
  const dim3 blocks(static_cast<unsigned int>((n + kTileSize - 1) / kTileSize),
                    static_cast<unsigned int>((m + kTileSize - 1) / kTileSize));
  const dim3 threads(kTileSize, kTileSize);
  matmulKernel<<<blocks, threads>>>(a, b, out, static_cast<int>(m), static_cast<int>(k),
                                       static_cast<int>(n));
}

double CudaBackend::sum(const float* data, std::size_t count) {
  if (count == 0) return 0.0;

  const std::size_t blocks = (count + kBlockSize - 1) / kBlockSize;
  float* partials = nullptr;
  if (cudaMalloc(&partials, blocks * sizeof(float)) != cudaSuccess) {
    cudaGetLastError();
    return 0.0;
  }

  sumKernel<<<static_cast<unsigned int>(blocks), kBlockSize>>>(data, partials, count);

  float* total = nullptr;
  if (cudaMalloc(&total, sizeof(float)) != cudaSuccess) {
    cudaGetLastError();
    cudaFree(partials);
    return 0.0;
  }
  sumFinalKernel<<<1, 1>>>(partials, total, blocks);

  float result = 0.0f;
  cudaMemcpy(&result, total, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(partials);
  cudaFree(total);
  return static_cast<double>(result);
}

}  // namespace milo::tensor::cuda
