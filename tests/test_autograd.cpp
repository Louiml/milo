#include <cmath>
#include <cstdio>
#include <vector>

#include "autograd/modules.hpp"
#include "autograd/ops.hpp"
#include "tensor/tensor.hpp"
#include "test_framework.hpp"

using namespace milo;
using namespace milo::autograd;
using tensor::Shape;

namespace {

/// A tensor of the given values, marked as one that takes a gradient.
Ptr leaf(const std::vector<double>& values, const std::vector<std::int64_t>& shape) {
  auto tensor = std::make_shared<Tensor>(Tensor::fromDoubles(Shape(shape), values));
  tensor->setRequiresGrad(true);
  return tensor;
}

/// A deep copy of a tensor: new storage, the same values.
Ptr copyOf(const Ptr& tensor) {
  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(tensor->numel()));
  std::vector<std::int64_t> indices(tensor->shape().rank(), 0);
  for (std::int64_t i = 0; i < tensor->numel(); ++i) {
    values.push_back(tensor->get(indices));
    for (std::size_t axis = tensor->shape().rank(); axis-- > 0;) {
      if (++indices[axis] < tensor->shape().dim(axis)) break;
      indices[axis] = 0;
    }
  }
  auto copy = std::make_shared<Tensor>(Tensor::fromDoubles(tensor->shape(), values));
  copy->setRequiresGrad(tensor->requiresGrad());
  return copy;
}

/// The numeric gradient of `function` with respect to `x`, by finite
/// differences.
std::vector<double> numericGradient(const Ptr& x, const std::function<Ptr(const Ptr&)>& function,
                                    double epsilon = 1e-2) {
  const std::int64_t total = x->numel();
  std::vector<double> gradient(static_cast<std::size_t>(total), 0.0);

  std::vector<std::int64_t> indices(x->shape().rank(), 0);
  for (std::int64_t i = 0; i < total; ++i) {
    const double here = x->get(indices);

    auto copy = copyOf(x);
    copy->set(indices, here + epsilon);
    const double high = function(copy)->getFlat(0);

    copy->set(indices, here - epsilon);
    const double low = function(copy)->getFlat(0);

    gradient[static_cast<std::size_t>(i)] = (high - low) / (2.0 * epsilon);

    for (std::size_t axis = x->shape().rank(); axis-- > 0;) {
      if (++indices[axis] < x->shape().dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return gradient;
}

/// Check that the analytic gradient of `function` matches the numeric one.
void checkGradient(const std::string& name, const Ptr& x,
                   const std::function<Ptr(const Ptr&)>& function, double tolerance = 0.05) {
  Tape::current().clear();

  const Ptr result = function(x);
  Tape::current().backward(result);

  const std::vector<double> expected = numericGradient(x, function);
  const Ptr& gradient = x->grad();
  if (!gradient) {
    ::milo::test::fail(__FILE__, __LINE__, name + ": there is no gradient at all");
    return;
  }

  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double actual = gradient->getFlat(static_cast<std::int64_t>(i));
    if (std::fabs(actual - expected[i]) > tolerance) {
      ::milo::test::fail(__FILE__, __LINE__,
                         name + ": the gradient at " + std::to_string(i) + " is " +
                             std::to_string(actual) + ", but it should be " +
                             std::to_string(expected[i]));
      return;
    }
  }
}

}  // namespace

TEST(autogradChecksTheGradientOfAddition) {
  auto x = leaf({1, 2, 3}, {3});
  auto y = leaf({10, 20, 30}, {3});
  checkGradient("add", x, [y](const Ptr& v) { return sum(add(v, y)); });
}

TEST(autogradChecksTheGradientOfSubtraction) {
  auto x = leaf({1, 2, 3}, {3});
  auto y = leaf({10, 20, 30}, {3});
  checkGradient("sub", x, [y](const Ptr& v) { return sum(sub(v, y)); });
}

TEST(autogradChecksTheGradientOfMultiplication) {
  auto x = leaf({1, 2, 3}, {3});
  auto y = leaf({10, 20, 30}, {3});
  checkGradient("mul", x, [y](const Ptr& v) { return sum(mul(v, y)); });
}

TEST(autogradChecksTheGradientOfDivision) {
  auto x = leaf({1, 2, 3}, {3});
  auto y = leaf({10, 20, 30}, {3});
  checkGradient("div", x, [y](const Ptr& v) { return sum(div(v, y)); });
}

TEST(autogradChecksTheGradientOfMultiplicationByAScalar) {
  auto x = leaf({1, 2, 3}, {3});
  checkGradient("mul by a scalar", x, [](const Ptr& v) { return sum(mul(v, 3.0)); });
}

TEST(autogradChecksTheGradientOfMatmul) {
  auto x = leaf({1, 2, 3, 4, 5, 6}, {2, 3});
  auto weight = leaf({0.5, -0.5, 0.25, 1.0, 0.0, -1.0}, {3, 2});
  checkGradient("matmul", weight, [x](const Ptr& v) { return sum(matmul(x, v)); });
}

TEST(autogradChecksTheGradientOfRelu) {
  auto x = leaf({-1, 0.5, 2}, {3});
  checkGradient("relu", x, [](const Ptr& v) { return sum(relu(v)); });
}

TEST(autogradChecksTheGradientOfSigmoid) {
  auto x = leaf({-1, 0.5, 2}, {3});
  checkGradient("sigmoid", x, [](const Ptr& v) { return sum(sigmoid(v)); });
}

TEST(autogradChecksTheGradientOfTanh) {
  auto x = leaf({-1, 0.5, 2}, {3});
  checkGradient("tanh", x, [](const Ptr& v) { return sum(tanh(v)); });
}

TEST(autogradChecksTheGradientOfGelu) {
  auto x = leaf({-1, 0.5, 2}, {3});
  checkGradient("gelu", x, [](const Ptr& v) { return sum(gelu(v)); });
}

TEST(autogradChecksTheGradientOfExp) {
  auto x = leaf({0.5, 1.5, 2.5}, {3});
  checkGradient("exp", x, [](const Ptr& v) { return sum(exp(v)); });
}

TEST(autogradChecksTheGradientOfLog) {
  auto x = leaf({0.5, 1.5, 2.5}, {3});
  checkGradient("log", x, [](const Ptr& v) { return sum(log(v)); });
}

TEST(autogradChecksTheGradientOfSqrt) {
  auto x = leaf({0.5, 1.5, 2.5}, {3});
  checkGradient("sqrt", x, [](const Ptr& v) { return sum(sqrt(v)); });
}

TEST(autogradChecksTheGradientOfMean) {
  auto x = leaf({1, 2, 3, 4}, {4});
  checkGradient("mean", x, [](const Ptr& v) { return mean(v); });
}

TEST(autogradChecksTheGradientOfSoftmax) {
  auto x = leaf({1, 2, 3}, {3});
  checkGradient("softmax", x, [](const Ptr& v) { return sum(softmax(v)); });
}

TEST(autogradChecksTheGradientOfReshape) {
  auto x = leaf({1, 2, 3, 4, 5, 6}, {2, 3});
  checkGradient("reshape", x, [](const Ptr& v) {
    return sum(reshape(v, Shape{std::vector<std::int64_t>{6}}));
  });
}

TEST(autogradChecksTheGradientOfTranspose) {
  auto x = leaf({1, 2, 3, 4, 5, 6}, {2, 3});
  checkGradient("transpose", x, [](const Ptr& v) { return sum(transpose(v)); });
}

TEST(autogradChecksTheGradientOfBroadcasting) {
  // The input is [1, 3] and is broadcast against a [2, 3] tensor.
  auto x = leaf({1, 2, 3}, {1, 3});
  auto y = leaf({10, 20, 30, 40, 50, 60}, {2, 3});
  checkGradient("broadcast", x, [y](const Ptr& v) { return sum(add(v, y)); });
}

TEST(autogradChecksTheGradientOfAMseLoss) {
  auto x = leaf({1, 2, 3}, {3});
  auto target = leaf({0.5, 0.5, 0.5}, {3});
  checkGradient("mse_loss", x,
                [target](const Ptr& v) { return mseLoss(v, target); });
}

TEST(autogradChecksTheGradientOfALinearLayer) {
  const Layer layer = linear(3, 2);
  auto x = leaf({1, 2, 3, 4, 5, 6}, {2, 3});

  Tape::current().clear();
  const Ptr result = sum(forward(layer, x));
  Tape::current().backward(result);

  const Ptr& weightGradient = layer.parameters.at("weight")->grad();
  CHECK(weightGradient != nullptr);
  if (weightGradient) {
    CHECK(weightGradient->shape() == layer.parameters.at("weight")->shape());
  }
  CHECK(layer.parameters.at("bias")->grad() != nullptr);
}

TEST(autogradTheOptimizerMovesTheParameters) {
  const Layer layer = linear(1, 1);
  auto x = leaf({2.0}, {1, 1});
  auto target = leaf({4.0}, {1, 1});

  Optimizer optimizer = sgd({layer.parameters.at("weight"), layer.parameters.at("bias")}, 0.1);

  const double before = layer.parameters.at("weight")->getFlat(0);

  Tape::current().clear();
  const Ptr prediction = forward(layer, x);
  const Ptr loss = mseLoss(prediction, target);
  Tape::current().backward(loss);
  step(optimizer);

  const double after = layer.parameters.at("weight")->getFlat(0);
  CHECK(std::fabs(after - before) > 1e-6);
}
