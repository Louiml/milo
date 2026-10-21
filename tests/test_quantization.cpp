#include <cmath>
#include <vector>

#include "autograd/modules.hpp"
#include "autograd/ops.hpp"
#include "tensor/quantization.hpp"
#include "test_framework.hpp"

using namespace milo;
using namespace milo::tensor;
using autograd::Layer;
using autograd::Ptr;
using autograd::Tape;
using autograd::forward;
using autograd::lora;
using autograd::mergeLora;
using autograd::qlora;
using autograd::sum;

namespace {

/// A weight matrix with a spread of values.
Tensor sampleWeight() {
  return Tensor::fromDoubles(Shape{std::vector<std::int64_t>{4, 8}},
                             {0.5, -1.25, 2.0, 0.0, 0.25, -0.75, 1.5, -0.1,
                              0.3, 0.7, -2.5, 1.1, 0.05, -0.4, 0.9, 0.2,
                              -1.0, 0.6, 0.15, -0.05, 2.2, -1.8, 0.35, 0.8,
                              0.45, -0.6, 1.3, -0.9, 0.05, 0.75, -1.1, 0.25});
}

}  // namespace

TEST(quantizationInt8RoundTrips) {
  const Tensor weight = sampleWeight();
  const QuantizedTensor quantized = quantizeInt8(weight);
  const Tensor restored = dequantize(quantized);

  CHECK_EQ(restored.shape().toString(), weight.shape().toString());
  for (std::int64_t i = 0; i < weight.numel(); ++i) {
    // The error is at most half of one step of the quantisation.
    const double step = quantized.scales[static_cast<std::size_t>(i / 8)];
    CHECK(std::fabs(restored.getFlat(i) - weight.getFlat(i)) <= step * 0.5 + 1e-6);
  }
}

TEST(quantizationNf4RoundTrips) {
  const Tensor weight = sampleWeight();
  const QuantizedTensor quantized = quantizeNf4(weight);
  const Tensor restored = dequantize(quantized);

  CHECK_EQ(restored.shape().toString(), weight.shape().toString());
  CHECK_EQ(quantized.blockSize, static_cast<std::int64_t>(64));

  // The widest gap in the NF4 codebook is 0.304, so the error is at most
  // half of one gap times the block's absolute maximum.
  const double absmax = quantized.absmax.empty() ? 1.0 : quantized.absmax.front();
  for (std::int64_t i = 0; i < weight.numel(); ++i) {
    const double difference = std::fabs(restored.getFlat(i) - weight.getFlat(i));
    CHECK(difference <= 0.16 * absmax);
  }
}

TEST(quantizationShrinksTheWeights) {
  const Tensor weight = sampleWeight();
  const double before = static_cast<double>(weight.numel()) * dtypeSize(weight.dtype());

  const QuantizedTensor int8 = quantizeInt8(weight);
  const QuantizedTensor nf4 = quantizeNf4(weight);

  CHECK(static_cast<double>(int8.byteSize()) < before);
  CHECK(static_cast<double>(nf4.byteSize()) < static_cast<double>(int8.byteSize()));
  // NF4 is about a quarter of the size of f32.
  CHECK(compressionRatio(nf4, weight) > 3.0);
}

TEST(quantizationUsesOneScalePerChannelForInt8) {
  const Tensor weight = sampleWeight();
  const QuantizedTensor quantized = quantizeInt8(weight);
  // One scale per row of the last axis.
  CHECK_EQ(quantized.scales.size(), static_cast<std::size_t>(4));
}

TEST(quantizationUsesOneScalePerBlockForNf4) {
  const Tensor weight = sampleWeight();
  const QuantizedTensor quantized = quantizeNf4(weight);
  // Thirty-two values fit in one block of 64.
  CHECK_EQ(quantized.scales.size(), static_cast<std::size_t>(1));
  CHECK_EQ(quantized.absmax.size(), static_cast<std::size_t>(1));
}

TEST(quantizationTheFusedMatrixProductMatchesTheObviousOne) {
  const Tensor weight = sampleWeight();
  const QuantizedTensor quantized = quantizeNf4(weight);

  const Tensor x = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{2, 4}}, {1, 2, 3, 4, 5, 6, 7, 8});

  const Tensor fused = matmulQuantized(x, quantized);
  const Tensor obvious = tensor::matmul(x, dequantize(quantized));

  CHECK_EQ(fused.shape().toString(), obvious.shape().toString());
  for (std::int64_t i = 0; i < fused.numel(); ++i) {
    CHECK(std::fabs(fused.getFlat(i) - obvious.getFlat(i)) < 1e-5);
  }
}

TEST(quantizationTheFusedMatrixProductIsCloseToTheExactOne) {
  const Tensor weight = sampleWeight();
  const QuantizedTensor quantized = quantizeNf4(weight);
  const Tensor x = Tensor::fromDoubles(Shape{std::vector<std::int64_t>{1, 4}}, {1, 2, 3, 4});

  const Tensor fused = matmulQuantized(x, quantized);
  const Tensor exact = tensor::matmul(x, weight);

  // The quantisation costs a little accuracy, but not much.
  for (std::int64_t i = 0; i < fused.numel(); ++i) {
    CHECK(std::fabs(fused.getFlat(i) - exact.getFlat(i)) < 1.0);
  }
}

TEST(loraTrainsOnlyTheAdapters) {
  const Layer layer = lora(4, 3, 2);
  CHECK_EQ(layer.kind, std::string("lora"));

  // The base is frozen.
  CHECK(!layer.parameters.at("weight")->requiresGrad());
  CHECK(!layer.parameters.at("bias")->requiresGrad());
  // The adapters are trained.
  CHECK(layer.parameters.at("lora_a")->requiresGrad());
  CHECK(layer.parameters.at("lora_b")->requiresGrad());

  CHECK_EQ(layer.parameters.at("lora_a")->shape().toString(), std::string("[4, 2]"));
  CHECK_EQ(layer.parameters.at("lora_b")->shape().toString(), std::string("[2, 3]"));
}

TEST(loraStartsAsTheBase) {
  const Layer layer = lora(4, 3, 2);
  // The second adapter starts at zero, so the update starts at nothing.
  const Tensor& b = *layer.parameters.at("lora_b");
  for (std::int64_t i = 0; i < b.numel(); ++i) {
    CHECK_EQ(b.getFlat(i), 0.0);
  }
}

TEST(loraMergesIntoTheBaseWeight) {
  const Layer layer = lora(4, 3, 2);

  // Give the adapter something to do.
  Tensor& a = *layer.parameters.at("lora_a");
  Tensor& b = *layer.parameters.at("lora_b");
  for (std::int64_t i = 0; i < a.numel(); ++i) a.setFlat(i, 0.1 * (i + 1));
  for (std::int64_t i = 0; i < b.numel(); ++i) b.setFlat(i, 0.2 * (i + 1));

  const auto input = std::make_shared<Tensor>(
      Tensor::fromDoubles(Shape{std::vector<std::int64_t>{1, 4}}, {1, 2, 3, 4}));

  const Layer merged = mergeLora(layer);
  CHECK_EQ(merged.kind, std::string("linear"));

  // The merged layer does the same thing as the unmerged one.
  const Ptr before = forward(layer, input);
  const Ptr after = forward(merged, input);
  CHECK_EQ(before->shape().toString(), after->shape().toString());
  for (std::int64_t i = 0; i < before->numel(); ++i) {
    CHECK(std::fabs(before->getFlat(i) - after->getFlat(i)) < 1e-4);
  }
}

TEST(loraTheGradientsReachTheAdaptersAndNotTheBase) {
  const Layer layer = lora(4, 3, 2);
  const auto input = std::make_shared<Tensor>(
      Tensor::fromDoubles(Shape{std::vector<std::int64_t>{1, 4}}, {1, 2, 3, 4}));
  input->setRequiresGrad(true);

  Tape::current().clear();
  const Ptr result = sum(forward(layer, input));
  Tape::current().backward(result);

  CHECK(layer.parameters.at("lora_a")->grad() != nullptr);
  CHECK(layer.parameters.at("lora_b")->grad() != nullptr);
  // The base is frozen, so it has no gradient.
  CHECK(layer.parameters.at("weight")->grad() == nullptr);
  CHECK(layer.parameters.at("bias")->grad() == nullptr);
}

TEST(qloraKeepsTheBaseQuantised) {
  const Layer layer = qlora(4, 3, 2);
  CHECK_EQ(layer.kind, std::string("qlora"));
  CHECK(layer.quantizedWeight != nullptr);
  CHECK_EQ(layer.quantizedWeight->scheme, std::string("nf4"));

  // The base weight is not held in f32 any more.
  const Tensor& weight = *layer.parameters.at("weight");
  for (std::int64_t i = 0; i < weight.numel(); ++i) {
    CHECK_EQ(weight.getFlat(i), 0.0);
  }

  // But the layer still works.
  const auto input = std::make_shared<Tensor>(
      Tensor::fromDoubles(Shape{std::vector<std::int64_t>{1, 4}}, {1, 2, 3, 4}));
  const Ptr result = forward(layer, input);
  CHECK_EQ(result->shape().toString(), std::string("[1, 3]"));
}
