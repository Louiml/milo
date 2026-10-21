#include <cmath>
#include <vector>

#include "tensor/convert.hpp"
#include "tensor/kernels.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "test_framework.hpp"

using namespace milo;
using namespace milo::tensor;

namespace {

/// A deliberately naive matrix product, used to check the real one.
std::vector<double> referenceMatmul(const std::vector<double>& a, std::int64_t m, std::int64_t k,
                                    const std::vector<double>& b, std::int64_t n) {
  std::vector<double> result(static_cast<std::size_t>(m * n), 0.0);
  for (std::int64_t row = 0; row < m; ++row) {
    for (std::int64_t column = 0; column < n; ++column) {
      double total = 0.0;
      for (std::int64_t inner = 0; inner < k; ++inner) {
        total += a[static_cast<std::size_t>(row * k + inner)] *
                 b[static_cast<std::size_t>(inner * n + column)];
      }
      result[static_cast<std::size_t>(row * n + column)] = total;
    }
  }
  return result;
}

bool closeEnough(double a, double b) { return std::fabs(a - b) < 1e-5; }

}  // namespace

TEST(tensorDTypesHaveSizes) {
  CHECK_EQ(dtypeSize(DType::F32), static_cast<std::size_t>(4));
  CHECK_EQ(dtypeSize(DType::F16), static_cast<std::size_t>(2));
  CHECK_EQ(dtypeSize(DType::BF16), static_cast<std::size_t>(2));
  CHECK_EQ(dtypeSize(DType::F64), static_cast<std::size_t>(8));
  CHECK_EQ(dtypeSize(DType::I8), static_cast<std::size_t>(1));
  CHECK_EQ(dtypeSize(DType::U8), static_cast<std::size_t>(1));
  CHECK_EQ(dtypeSize(DType::Bool), static_cast<std::size_t>(1));
  CHECK_EQ(std::string(dtypeName(DType::F32)), std::string("f32"));
  CHECK(dtypeIsFloat(DType::BF16));
  CHECK(dtypeIsInteger(DType::I32));
  CHECK(!dtypeIsSigned(DType::U8));
}

TEST(tensorDTypeNamesParse) {
  DType dtype = DType::F32;
  CHECK(dtypeFromName("bf16", dtype));
  CHECK(dtype == DType::BF16);
  CHECK(!dtypeFromName("nonsense", dtype));
}

TEST(tensorShapesKnowTheirSize) {
  const Shape shape{std::vector<std::int64_t>{2, 3, 4}};
  CHECK_EQ(shape.rank(), static_cast<std::size_t>(3));
  CHECK_EQ(shape.numel(), static_cast<std::int64_t>(24));
  CHECK_EQ(shape.toString(), std::string("[2, 3, 4]"));
  CHECK_EQ(shape.toDisplayString(), std::string("2x3x4"));
  CHECK_EQ(Shape::scalar().numel(), static_cast<std::int64_t>(1));
}

TEST(tensorStridesAreRowMajor) {
  const Strides strides = Strides::contiguous(Shape{std::vector<std::int64_t>{2, 3}});
  CHECK_EQ(strides[0], static_cast<std::int64_t>(3));
  CHECK_EQ(strides[1], static_cast<std::int64_t>(1));
}

TEST(tensorBroadcastsShapes) {
  const Shape a{std::vector<std::int64_t>{2, 3}};
  const Shape b{std::vector<std::int64_t>{3}};
  const Shape result = broadcastShapes(a, b);
  CHECK_EQ(result.toString(), std::string("[2, 3]"));

  const Shape c{std::vector<std::int64_t>{1, 3}};
  CHECK_EQ(broadcastShapes(a, c).toString(), std::string("[2, 3]"));

  const Shape bad{std::vector<std::int64_t>{4}};
  CHECK(!canBroadcast(a, bad));
}

TEST(tensorCreatesZerosAndOnes) {
  const Tensor zeros = Tensor::zeros(DType::F32, Shape{std::vector<std::int64_t>{2, 2}});
  CHECK_EQ(zeros.numel(), static_cast<std::int64_t>(4));
  CHECK_EQ(zeros.getFlat(0), 0.0);
  CHECK(zeros.isContiguous());
  CHECK(zeros.ownsStorage());

  const Tensor ones = Tensor::ones(DType::F32, Shape{std::vector<std::int64_t>{3}});
  CHECK_EQ(ones.getFlat(2), 1.0);
}

TEST(tensorStoresAndReadsElements) {
  Tensor tensor = Tensor::zeros(DType::F32, Shape{std::vector<std::int64_t>{2, 3}});
  tensor.set({1, 2}, 7.5);
  CHECK_EQ(tensor.get({1, 2}), 7.5);
  CHECK_EQ(tensor.get({0, 0}), 0.0);
  CHECK_EQ(tensor.flatIndex({1, 2}), static_cast<std::int64_t>(5));
}

TEST(tensorReshapeMakesAView) {
  const Tensor tensor = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 3}},
                                            {1, 2, 3, 4, 5, 6});
  Tensor reshaped = tensor.reshape(Shape{std::vector<std::int64_t>{3, 2}});
  CHECK_EQ(reshaped.shape().toString(), std::string("[3, 2]"));
  // A view shares the storage, so writing through one is visible in the other.
  reshaped.setFlat(0, 99.0);
  CHECK_EQ(tensor.getFlat(0), 99.0);
  CHECK(!reshaped.ownsStorage());
}

TEST(tensorTransposeMakesAView) {
  const Tensor tensor = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 3}},
                                            {1, 2, 3, 4, 5, 6});
  const Tensor transposed = tensor.transpose();
  CHECK_EQ(transposed.shape().toString(), std::string("[3, 2]"));
  CHECK(!transposed.isContiguous());
  CHECK_EQ(transposed.get({0, 1}), 4.0);
  // Making it contiguous copies the data.
  const Tensor copy = transposed.contiguous();
  CHECK(copy.isContiguous());
  CHECK_EQ(copy.get({0, 1}), 4.0);
}

TEST(tensorSlicesMakeAView) {
  const Tensor tensor = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{5}},
                                            {10, 20, 30, 40, 50});
  const Tensor slice = tensor.slice(0, 1, 4, 1);
  CHECK_EQ(slice.shape().toString(), std::string("[3]"));
  CHECK_EQ(slice.getFlat(0), 20.0);

  const Tensor strided = tensor.slice(0, 0, 5, 2);
  CHECK_EQ(strided.shape().toString(), std::string("[3]"));
  CHECK_EQ(strided.getFlat(1), 30.0);
}

TEST(tensorConvertsBetweenTypes) {
  const Tensor tensor = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2}},
                                            {1.5, -2.25}, DType::F32);
  const Tensor asHalf = tensor.to(DType::F16);
  CHECK(asHalf.dtype() == DType::F16);
  CHECK(closeEnough(asHalf.getFlat(0), 1.5));
  CHECK(closeEnough(asHalf.getFlat(1), -2.25));

  const Tensor asInt = tensor.to(DType::I32);
  CHECK_EQ(asInt.getFlat(0), 1.0);
}

TEST(tensorHalfConversionsRoundTrip) {
  CHECK_EQ(halfToFloat(floatToHalf(1.0f)), 1.0f);
  CHECK_EQ(halfToFloat(floatToHalf(-0.5f)), -0.5f);
  CHECK_EQ(halfToFloat(floatToHalf(0.0f)), 0.0f);
  // Half precision keeps about three decimal digits.
  CHECK(std::fabs(halfToFloat(floatToHalf(3.14159f)) - 3.14159f) < 2e-3f);
}

TEST(tensorBfloat16ConversionsRoundTrip) {
  CHECK_EQ(bfloat16ToFloat(floatToBfloat16(1.0f)), 1.0f);
  CHECK_EQ(bfloat16ToFloat(floatToBfloat16(2.5f)), 2.5f);
  // bfloat16 keeps eight mantissa bits, so the error is about 0.2%.
  CHECK(std::fabs(bfloat16ToFloat(floatToBfloat16(1.0f / 3.0f)) - 1.0f / 3.0f) < 1e-2f);
}

TEST(tensorElementwiseAddMatchesAReference) {
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3}}, {1, 2, 3});
  const Tensor b = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3}}, {10, 20, 30});
  const Tensor sum = binaryOp(BinaryOp::Add, a, b);
  CHECK_EQ(sum.getFlat(0), 11.0);
  CHECK_EQ(sum.getFlat(2), 33.0);

  const Tensor product = binaryOp(BinaryOp::Mul, a, b);
  CHECK_EQ(product.getFlat(1), 40.0);
}

TEST(tensorElementwiseBroadcasts) {
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 3}},
                                       {1, 2, 3, 4, 5, 6});
  const Tensor b = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3}}, {10, 20, 30});
  const Tensor sum = binaryOp(BinaryOp::Add, a, b);
  CHECK_EQ(sum.shape().toString(), std::string("[2, 3]"));
  CHECK_EQ(sum.get({0, 0}), 11.0);
  CHECK_EQ(sum.get({1, 2}), 36.0);
}

TEST(tensorElementwiseWithAScalar) {
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3}}, {1, 2, 3});
  const Tensor scaled = binaryOp(BinaryOp::Mul, a, 3.0);
  CHECK_EQ(scaled.getFlat(0), 3.0);
  CHECK_EQ(scaled.getFlat(2), 9.0);
}

TEST(tensorElementwiseWorksOnNonContiguousTensors) {
  const Tensor tensor = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 3}},
                                            {1, 2, 3, 4, 5, 6});
  const Tensor transposed = tensor.transpose();
  const Tensor doubled = binaryOp(BinaryOp::Mul, transposed, 2.0);
  CHECK_EQ(doubled.shape().toString(), std::string("[3, 2]"));
  CHECK_EQ(doubled.get({0, 1}), 8.0);
}

TEST(tensorUnaryOps) {
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3}}, {-1, 0, 2});
  CHECK_EQ(unaryOp(UnaryOp::Relu, a).getFlat(0), 0.0);
  CHECK_EQ(unaryOp(UnaryOp::Relu, a).getFlat(2), 2.0);
  CHECK_EQ(unaryOp(UnaryOp::Neg, a).getFlat(2), -2.0);
  CHECK(closeEnough(unaryOp(UnaryOp::Exp, a).getFlat(1), 1.0));
}

TEST(tensorReductions) {
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{4}}, {1, 2, 3, 4});
  CHECK_EQ(sum(a).getFlat(0), 10.0);
  CHECK_EQ(mean(a).getFlat(0), 2.5);
  CHECK_EQ(max(a).getFlat(0), 4.0);
  CHECK_EQ(min(a).getFlat(0), 1.0);
  CHECK_EQ(argmax(a).getFlat(0), 3.0);
}

TEST(tensorReductionsWorkOnStridedTensors) {
  // A reduction over a strided tensor must visit every element exactly
  // once.
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 3}},
                                        {1, 2, 3, 4, 5, 6});
  const Tensor transposed = a.transpose();
  CHECK(!transposed.isContiguous());
  CHECK_EQ(sum(transposed).getFlat(0), 21.0);
  CHECK_EQ(mean(transposed).getFlat(0), 3.5);
  CHECK_EQ(max(transposed).getFlat(0), 6.0);
  CHECK_EQ(min(transposed).getFlat(0), 1.0);

  const Tensor strided = a.slice(0, 0, 2, 1);
  CHECK_EQ(sum(strided).getFlat(0), 21.0);
}

TEST(tensorMatmulMatchesAReference) {
  const std::int64_t m = 3;
  const std::int64_t k = 4;
  const std::int64_t n = 2;
  std::vector<double> aValues;
  std::vector<double> bValues;
  for (std::int64_t i = 0; i < m * k; ++i) aValues.push_back(static_cast<double>(i) * 0.5 - 3.0);
  for (std::int64_t i = 0; i < k * n; ++i) bValues.push_back(static_cast<double>(i) * 0.25 + 1.0);

  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{m, k}}, aValues);
  const Tensor b = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{k, n}}, bValues);
  const Tensor product = matmul(a, b);

  CHECK_EQ(product.shape().toString(), std::string("[3, 2]"));
  const std::vector<double> expected = referenceMatmul(aValues, m, k, bValues, n);
  for (std::int64_t row = 0; row < m; ++row) {
    for (std::int64_t column = 0; column < n; ++column) {
      const double actual = product.get({row, column});
      const double want = expected[static_cast<std::size_t>(row * n + column)];
      CHECK(closeEnough(actual, want));
    }
  }
}

TEST(tensorMatmulHandlesVectors) {
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3}}, {1, 2, 3});
  const Tensor b = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3}}, {4, 5, 6});
  const Tensor product = matmul(a, b);
  CHECK(product.shape().isScalar());
  CHECK(closeEnough(product.getFlat(0), 32.0));

  const Tensor matrix = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3, 1}}, {4, 5, 6});
  const Tensor asColumn = matmul(a, matrix);
  CHECK_EQ(asColumn.shape().toString(), std::string("[1]"));
  CHECK(closeEnough(asColumn.getFlat(0), 32.0));
}

TEST(tensorMatmulRejectsMismatchedShapes) {
  const Tensor a = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 3}}, {1, 2, 3, 4, 5, 6});
  const Tensor b = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 2}}, {1, 2, 3, 4});
  bool threw = false;
  try {
    matmul(a, b);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(tensorSoftmaxSumsToOne) {
  const Tensor logits = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 3}},
                                            {1, 2, 3, -1, 0, 1});
  const Tensor result = softmax(logits);
  CHECK_EQ(result.shape().toString(), std::string("[2, 3]"));
  double first = 0.0;
  double second = 0.0;
  for (std::int64_t column = 0; column < 3; ++column) {
    first += result.get({0, column});
    second += result.get({1, column});
  }
  CHECK(closeEnough(first, 1.0));
  CHECK(closeEnough(second, 1.0));
  // The largest logit gets the largest share.
  CHECK(result.get({0, 2}) > result.get({0, 0}));
}

TEST(tensorSoftmaxIsStable) {
  const Tensor logits = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{3}},
                                            {1000.0, 1001.0, 1002.0});
  const Tensor result = softmax(logits);
  double total = 0.0;
  for (std::int64_t i = 0; i < 3; ++i) total += result.getFlat(i);
  CHECK(closeEnough(total, 1.0));
}

TEST(tensorKernelsAreDispatched) {
  // Whatever instruction set is chosen, the results must be the same.
  const kernels::Isa isa = kernels::activeIsa();
  CHECK(isa == kernels::Isa::Scalar || isa == kernels::Isa::Avx2 || isa == kernels::Isa::Neon);
  CHECK(std::string(kernels::isaName(isa)).size() > 0);
}

TEST(tensorKernelsAgreeWithScalar) {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<float> b = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> out(9, 0.0f);
  kernels::table().addF32(a.data(), b.data(), out.data(), a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK_EQ(out[i], a[i] + b[i]);
  }
}
