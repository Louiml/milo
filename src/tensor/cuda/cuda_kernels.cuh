// The CUDA kernels.
//
// These are the operations that are worth moving to a GPU: the elementwise
// maths and the matrix product. Everything else is left to the CPU.
#include <cuda_runtime.h>

namespace milo::tensor::cuda {

// These kernels are in a header rather than a translation unit of their
// own, because a kernel can only be launched from the file that defines it.


__global__ void fillKernel(float* data, std::size_t count, float value) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    data[index] = value;
  }
}

__global__ void addKernel(const float* a, const float* b, float* out, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    out[index] = a[index] + b[index];
  }
}

__global__ void subKernel(const float* a, const float* b, float* out, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    out[index] = a[index] - b[index];
  }
}

__global__ void mulKernel(const float* a, const float* b, float* out, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    out[index] = a[index] * b[index];
  }
}

__global__ void divKernel(const float* a, const float* b, float* out, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    out[index] = a[index] / b[index];
  }
}

// A tiled matrix product.
//
// Each block computes one 16x16 tile of the result. The tiles are staged
// through shared memory, so the inner loop reads the inputs once per tile
// rather than once per element.
constexpr int kTile = 16;

__global__ void matmulKernel(const float* a, const float* b, float* out, int m, int k, int n) {
  __shared__ float aTile[kTile][kTile];
  __shared__ float bTile[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int column = blockIdx.x * kTile + threadIdx.x;

  float total = 0.0f;
  for (int tile = 0; tile < (k + kTile - 1) / kTile; ++tile) {
    const int aColumn = tile * kTile + threadIdx.x;
    const int bRow = tile * kTile + threadIdx.y;

    aTile[threadIdx.y][threadIdx.x] =
        (row < m && aColumn < k) ? a[row * k + aColumn] : 0.0f;
    bTile[threadIdx.y][threadIdx.x] =
        (bRow < k && column < n) ? b[bRow * n + column] : 0.0f;

    __syncthreads();

    for (int i = 0; i < kTile; ++i) {
      total += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && column < n) {
    out[row * n + column] = total;
  }
}

// A reduction over a block, then one value per block.
__global__ void sumKernel(const float* data, float* partials, std::size_t count) {
  __shared__ float shared[256];
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  float value = 0.0f;
  if (index < count) {
    value = data[index];
  }
  shared[threadIdx.x] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void sumFinalKernel(const float* partials, float* total, std::size_t count) {
  float value = 0.0f;
  for (std::size_t i = 0; i < count; ++i) {
    value += partials[i];
  }
  *total = value;
}

}  // namespace milo::tensor::cuda
