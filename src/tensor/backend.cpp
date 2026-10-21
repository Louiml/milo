// The backends, and the operations that run on them.
//
// The tensor engine asks a backend where a tensor's memory is and what can
// be done with it. The CPU backend is always there. A GPU backend is
// registered when the machine can reach one.
#include "tensor/backend.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

#if defined(MILO_HAVE_CUDA)
#include "tensor/cuda/cuda_backend.hpp"
#endif

namespace milo::tensor {
namespace {

/// The backend that does everything on the host.
class CpuBackend : public Backend {
 public:
  const char* name() const override { return "cpu"; }
  Device device() const override { return Device::Cpu; }
  bool supports(DType) const override { return true; }

  void* allocate(std::size_t bytes) override {
    if (bytes == 0) return nullptr;
    return ::operator new(bytes, std::align_val_t(64));
  }

  void deallocate(void* data) override {
    if (data != nullptr) {
      ::operator delete(data, std::align_val_t(64));
    }
  }

  void copyToDevice(void* destination, const void* source, std::size_t bytes) override {
    std::memcpy(destination, source, bytes);
  }

  void copyToHost(void* destination, const void* source, std::size_t bytes) override {
    std::memcpy(destination, source, bytes);
  }

  void copyBetween(void* destination, const void* source, std::size_t bytes) override {
    std::memcpy(destination, source, bytes);
  }

  void synchronize() override {}
  std::uint64_t freeMemory() const override { return 0; }
  std::uint64_t totalMemory() const override { return 0; }

  void fill(float* data, std::size_t count, float value) override {
    std::fill(data, data + count, value);
  }

  void elementwise(BackendOp op, const float* a, const float* b, float* out,
                   std::size_t count) override {
    for (std::size_t i = 0; i < count; ++i) {
      switch (op) {
        case BackendOp::Add:
          out[i] = a[i] + b[i];
          break;
        case BackendOp::Sub:
          out[i] = a[i] - b[i];
          break;
        case BackendOp::Mul:
          out[i] = a[i] * b[i];
          break;
        case BackendOp::Div:
          out[i] = a[i] / b[i];
          break;
      }
    }
  }

  void matmul(const float* a, const float* b, float* out, std::int64_t m, std::int64_t k,
              std::int64_t n) override {
    for (std::int64_t row = 0; row < m; ++row) {
      for (std::int64_t column = 0; column < n; ++column) {
        float total = 0.0f;
        for (std::int64_t inner = 0; inner < k; ++inner) {
          total += a[row * k + inner] * b[inner * n + column];
        }
        out[row * n + column] = total;
      }
    }
  }

  double sum(const float* data, std::size_t count) override {
    double total = 0.0;
    for (std::size_t i = 0; i < count; ++i) total += data[i];
    return total;
  }
};

/// The registry of backends.
class Registry {
 public:
  static Registry& instance() {
    static Registry registry;
    return registry;
  }

  Backend* forDevice(Device device) {
    switch (device) {
      case Device::Cuda: {
#if defined(MILO_HAVE_CUDA)
        std::lock_guard<std::mutex> lock(mutex_);
        if (cuda_ == nullptr) {
          try {
            cuda_ = std::make_unique<cuda::CudaBackend>(0);
          } catch (const std::exception&) {
            return nullptr;
          }
        }
        return cuda_.get();
#else
        return nullptr;
#endif
      }
      case Device::Cpu:
      default:
        return cpu();
    }
  }

  std::vector<Backend*> all() {
    std::vector<Backend*> backends;
    backends.push_back(cpu());
#if defined(MILO_HAVE_CUDA)
    Backend* cuda = forDevice(Device::Cuda);
    if (cuda != nullptr) backends.push_back(cuda);
#endif
    return backends;
  }

 private:
  Backend* cpu() {
    static CpuBackend backend;
    return &backend;
  }

  std::mutex mutex_;
#if defined(MILO_HAVE_CUDA)
  std::unique_ptr<cuda::CudaBackend> cuda_;
#endif
};

}  // namespace

Backend* backendFor(Device device) { return Registry::instance().forDevice(device); }

std::vector<Backend*> availableBackends() { return Registry::instance().all(); }

// ---------------------------------------------------------------------------
// The operations that run on a device
// ---------------------------------------------------------------------------

namespace {

/// Copy a tensor to the host, so that an operation the device cannot do
/// still works.
Tensor toHost(const Tensor& tensor) { return tensor.hostCopy(); }

/// The bytes of a contiguous f32 tensor.
const float* f32Data(const Tensor& tensor) {
  return reinterpret_cast<const float*>(static_cast<const std::byte*>(tensor.storage()->data()) +
                                     tensor.byteOffset());
}

float* f32DataMutable(Tensor& tensor) {
  return reinterpret_cast<float*>(static_cast<std::byte*>(tensor.storage()->data()) +
                                  tensor.byteOffset());
}

}  // namespace

Tensor deviceBinaryOp(BinaryOp op, const Tensor& a, const Tensor& b) {
  Backend* backend = backendFor(a.storage()->device() != Device::Cpu ? a.storage()->device()
                                                                     : b.storage()->device());
  if (backend == nullptr) {
    throw std::runtime_error("there is no backend for this device");
  }

  // The kernels are written for contiguous f32 tensors. Anything else comes
  // back to the host first, and goes out again afterwards.
  const bool direct = a.dtype() == DType::F32 && b.dtype() == DType::F32 && a.isContiguous() &&
                      b.isContiguous() && a.storage()->device() == backend->device() &&
                      b.storage()->device() == backend->device();

  if (!direct) {
    const Tensor hostA = toHost(a);
    const Tensor hostB = toHost(b);
    const Tensor result = binaryOp(op, hostA, hostB);
    Tensor out = Tensor::create(result.dtype(), result.shape(), backend->device());
    out.copyFrom(result);
    return out;
  }

  const Shape shape = broadcastShapes(a.shape(), b.shape());
  Tensor out = Tensor::create(DType::F32, shape, backend->device());

  backend->elementwise(op == BinaryOp::Add    ? BackendOp::Add
                        : op == BinaryOp::Sub  ? BackendOp::Sub
                        : op == BinaryOp::Mul  ? BackendOp::Mul
                                              : BackendOp::Div,
                        f32Data(a), f32Data(b), f32DataMutable(out),
                        static_cast<std::size_t>(out.numel()));
  backend->synchronize();
  return out;
}

Tensor deviceMatmul(const Tensor& a, const Tensor& b) {
  const Device device = a.storage()->device() != Device::Cpu ? a.storage()->device()
                                                             : b.storage()->device();
  Backend* backend = backendFor(device);
  if (backend == nullptr) {
    throw std::runtime_error("there is no backend for this device");
  }

  const bool direct = a.dtype() == DType::F32 && b.dtype() == DType::F32 && a.isContiguous() &&
                      b.isContiguous() && a.storage()->device() == device &&
                      b.storage()->device() == device;

  if (!direct) {
    const Tensor hostA = toHost(a);
    const Tensor hostB = toHost(b);
    const Tensor result = matmul(hostA, hostB);
    Tensor out = Tensor::create(result.dtype(), result.shape(), device);
    out.copyFrom(result);
    return out;
  }

  const bool aIsVector = a.shape().rank() == 1;
  const bool bIsVector = b.shape().rank() == 1;
  if (aIsVector || bIsVector) {
    // The kernel does matrices; the vector cases come back to the host.
    const Tensor result = matmul(a.hostCopy(), b.hostCopy());
    Tensor out = Tensor::create(result.dtype(), result.shape(), device);
    out.copyFrom(result);
    return out;
  }

  const std::int64_t m = a.shape().dim(0);
  const std::int64_t k = a.shape().dim(1);
  const std::int64_t n = b.shape().dim(1);

  Tensor out = Tensor::create(DType::F32, Shape{std::vector<std::int64_t>{m, n}}, device);
  backend->matmul(f32Data(a), f32Data(b), f32DataMutable(out), m, k, n);
  backend->synchronize();
  return out;
}

Tensor deviceSum(const Tensor& a) {
  Backend* backend = backendFor(a.storage()->device());
  if (backend == nullptr) {
    throw std::runtime_error("there is no backend for this device");
  }
  if (a.dtype() != DType::F32 || !a.isContiguous()) {
    const Tensor result = sum(a.hostCopy());
    Tensor out = Tensor::create(result.dtype(), result.shape(), a.storage()->device());
    out.copyFrom(result);
    return out;
  }
  const double total = backend->sum(f32Data(a), static_cast<std::size_t>(a.numel()));
  return Tensor::full(DType::F32, Shape::scalar(), total, a.storage()->device());
}

Tensor deviceUnaryOp(UnaryOp op, const Tensor& a) {
  Backend* backend = backendFor(a.storage()->device());
  if (backend == nullptr) {
    throw std::runtime_error("there is no backend for this device");
  }
  // The GPU has no transcendental functions yet, so the unary operations
  // come back to the host. The answer is right; it is just not fast.
  const Tensor result = unaryOp(op, a.hostCopy());
  Tensor out = Tensor::create(result.dtype(), result.shape(), a.storage()->device());
  out.copyFrom(result);
  return out;
}

}  // namespace milo::tensor
