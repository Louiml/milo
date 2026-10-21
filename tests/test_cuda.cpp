#include <cmath>
#include <vector>

#include "tensor/backend.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "test_framework.hpp"

#if defined(MILO_HAVE_CUDA)
#include "tensor/cuda/cuda_backend.hpp"
#endif

using namespace milo;
using namespace milo::tensor;

namespace {

bool closeEnough(double a, double b, double tolerance = 1e-4) {
  return std::fabs(a - b) <= tolerance * (1.0 + std::fabs(a) + std::fabs(b));
}

}  // namespace

TEST(theCpuBackendIsAlwaysThere) {
  Backend* cpu = backendFor(Device::Cpu);
  CHECK(cpu != nullptr);
  CHECK_EQ(std::string(cpu->name()), std::string("cpu"));
  CHECK(cpu->supports(DType::F32));
  CHECK(cpu->supports(DType::BF16));
}

#if defined(MILO_HAVE_CUDA)

TEST(theCudaBackendIsFoundWhenThereIsAGpu) {
  if (!cuda::isAvailable()) {
    // There is no GPU on this machine, so there is nothing to check.
    return;
  }
  Backend* backend = backendFor(Device::Cuda);
  CHECK(backend != nullptr);
  if (backend != nullptr) {
    CHECK_EQ(backend->device(), Device::Cuda);
    CHECK(backend->totalMemory() > 0);
  }
}

TEST(aGpuTensorHoldsItsValues) {
  if (!cuda::isAvailable()) return;

  const Tensor host = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 3}},
                                          {1, 2, 3, 4, 5, 6});
  const Tensor device = host.toDevice(Device::Cuda);
  CHECK_EQ(device.storage()->device(), Device::Cuda);
  CHECK_EQ(device.numel(), host.numel());

  // Bring it back and check it survived the journey.
  const Tensor back = device.hostCopy();
  CHECK_EQ(back.storage()->device(), Device::Cpu);
  for (std::int64_t i = 0; i < host.numel(); ++i) {
    CHECK_EQ(back.getFlat(i), host.getFlat(i));
  }
}

TEST(theGpuAndTheCpuAgreeOnAMatrixProduct) {
  if (!cuda::isAvailable()) return;

  const std::int64_t m = 33;
  const std::int64_t k = 17;
  const std::int64_t n = 19;

  std::vector<double> aValues;
  std::vector<double> bValues;
  for (std::int64_t i = 0; i < m * k; ++i) aValues.push_back(std::sin(static_cast<double>(i) * 0.5));
  for (std::int64_t i = 0; i < k * n; ++i) bValues.push_back(std::cos(static_cast<double>(i) * 0.25));

  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{m, k}}, aValues);
  const Tensor b = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{k, n}}, bValues);

  const Tensor onCpu = matmul(a, b);
  const Tensor onGpu = matmul(a.toDevice(Device::Cuda), b.toDevice(Device::Cuda));

  CHECK_EQ(onGpu.shape().toString(), onCpu.shape().toString());
  const Tensor back = onGpu.hostCopy();
  for (std::int64_t i = 0; i < onCpu.numel(); ++i) {
    CHECK(closeEnough(back.getFlat(i), onCpu.getFlat(i), 1e-4));
  }
}

TEST(theGpuAndTheCpuAgreeOnElementwiseMaths) {
  if (!cuda::isAvailable()) return;

  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{4}}, {1, 2, 3, 4});
  const Tensor b = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{4}}, {10, 20, 30, 40});

  const Tensor aDevice = a.toDevice(Device::Cuda);
  const Tensor bDevice = b.toDevice(Device::Cuda);

  const Tensor sum = binaryOp(BinaryOp::Add, aDevice, bDevice);
  const Tensor sumCpu = binaryOp(BinaryOp::Add, a, b);
  for (std::int64_t i = 0; i < sumCpu.numel(); ++i) {
    CHECK(closeEnough(sum.hostCopy().getFlat(i), sumCpu.getFlat(i)));
  }

  const Tensor product = binaryOp(BinaryOp::Mul, aDevice, bDevice);
  const Tensor productCpu = binaryOp(BinaryOp::Mul, a, b);
  for (std::int64_t i = 0; i < productCpu.numel(); ++i) {
    CHECK(closeEnough(product.hostCopy().getFlat(i), productCpu.getFlat(i)));
  }
}

TEST(theGpuAndTheCpuAgreeOnASum) {
  if (!cuda::isAvailable()) return;

  std::vector<double> values;
  for (std::int64_t i = 0; i < 1000; ++i) values.push_back(static_cast<double>(i) * 0.25);
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{1000}}, values);

  const double onCpu = sum(a).getFlat(0);
  const double onGpu = sum(a.toDevice(Device::Cuda)).hostCopy().getFlat(0);

  CHECK(closeEnough(onGpu, onCpu, 1e-5));
}

TEST(aGpuTensorCannotBeChangedInPlace) {
  if (!cuda::isAvailable()) return;

  const Tensor device = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2}}, {1, 2})
                             .toDevice(Device::Cuda);

  bool threw = false;
  try {
    Tensor copy = device;
    copy.setFlat(0, 5.0);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

#else

TEST(theCudaBackendIsAbsentWithoutTheToolkit) {
  // The build has no CUDA in it, so there is no GPU to talk to.
  CHECK(cuda::isAvailable() == false);
}

#endif
