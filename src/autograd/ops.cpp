#include "autograd/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace milo::autograd {
namespace {

using tensor::BinaryOp;
using tensor::DType;
using tensor::Shape;
using tensor::UnaryOp;

/// Should this operation be recorded?
bool shouldRecord(const std::vector<Ptr>& inputs) {
  if (!recording()) return false;
  for (const Ptr& input : inputs) {
    if (input && input->requiresGrad()) return true;
  }
  return false;
}


/// The raw (unrecorded) product of two tensors.
Ptr rawMul(const Ptr& a, const Ptr& b) {
  return std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *a, *b));
}

/// The raw difference of two tensors.
Ptr rawSub(const Ptr& a, const Ptr& b) {
  return std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Sub, *a, *b));
}

/// The raw sum of two tensors.
Ptr rawAdd(const Ptr& a, const Ptr& b) {
  return std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Add, *a, *b));
}

/// The raw transpose of a tensor.
Ptr rawTranspose(const Ptr& a) {
  return std::make_shared<Tensor>(a->transpose());
}

/// The raw product of a matrix and the transpose of another: `a @ b^T`.
Ptr rawMatmulTimesTranspose(const Ptr& a, const Ptr& b) {
  return std::make_shared<Tensor>(tensor::matmul(*a, *rawTranspose(b)));
}

/// The raw product of the transpose of a matrix and another: `a^T @ b`.
Ptr rawTransposeTimesMatmul(const Ptr& a, const Ptr& b) {
  return std::make_shared<Tensor>(tensor::matmul(*rawTranspose(a), *b));
}

}  // namespace

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Ptr broadcastTo(const Ptr& gradient, const Shape& target) {
  if (gradient->shape() == target) return gradient;
  auto ones = std::make_shared<Tensor>(Tensor::ones(gradient->dtype(), target));
  return std::make_shared<Tensor>(
      tensor::binaryOp(BinaryOp::Mul, *gradient, *ones));
}

Ptr collapseLastAxis(const Ptr& tensor) {
  const Shape& shape = tensor->shape();
  if (shape.rank() == 0) return tensor;
  std::vector<std::int64_t> dims = shape.dims();
  dims.back() = 1;
  return std::make_shared<Tensor>(tensor->reshape(Shape(dims)));
}

void accumulate(const Ptr& target, const Ptr& gradient, const char* /*label*/) {
  if (!target || !gradient) return;
  if (!target->requiresGrad()) {
    return;
  }
  target->accumulateGrad(gradient);
}

Ptr reduceToShape(const Ptr& gradient, const Shape& target) {
  if (!gradient) return gradient;
  if (gradient->shape() == target) return gradient;

  // A scalar gradient is spread over the whole tensor.
  if (gradient->shape().rank() == 0) {
    return std::make_shared<Tensor>(
        Tensor::full(gradient->dtype(), target, gradient->getFlat(0)));
  }

  // Sum over the axes that were broadcast on the way out.
  const std::size_t targetRank = target.rank();
  const std::size_t gradientRank = gradient->shape().rank();

  std::vector<std::int64_t> axes;
  for (std::size_t i = 0; i < gradientRank - targetRank; ++i) {
    axes.push_back(static_cast<std::int64_t>(i));
  }
  for (std::size_t i = 0; i < targetRank; ++i) {
    const std::size_t gradientAxis = gradientRank - targetRank + i;
    if (target.dim(i) == 1 && gradient->shape().dim(gradientAxis) != 1) {
      axes.push_back(static_cast<std::int64_t>(gradientAxis));
    }
  }

  Ptr result = gradient;
  if (!axes.empty()) {
    result = std::make_shared<Tensor>(tensor::sumOverAxes(*result, axes));
  }
  if (result->shape() != target) {
    result = std::make_shared<Tensor>(result->reshape(target));
  }
  return result;
}

Ptr onesLike(const Ptr& a) {
  return std::make_shared<Tensor>(Tensor::ones(a->dtype(), a->shape()));
}

Ptr zerosLike(const Ptr& a) {
  return std::make_shared<Tensor>(Tensor::zeros(a->dtype(), a->shape()));
}

Ptr trainable(Ptr tensor) {
  if (tensor) tensor->setRequiresGrad(true);
  return tensor;
}

// ---------------------------------------------------------------------------
// Binary operations
// ---------------------------------------------------------------------------

Ptr add(const Ptr& a, const Ptr& b) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Add, *a, *b));
  if (shouldRecord({a, b})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a, b};
    node.outputs = {result};
    node.backward = [a, b, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(gradient, a->shape()), "mean-a");
      accumulate(b, reduceToShape(gradient, b->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr add(const Ptr& a, double b) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Add, *a, b));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(gradient, a->shape()), "mean-a");
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr sub(const Ptr& a, const Ptr& b) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Sub, *a, *b));
  if (shouldRecord({a, b})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a, b};
    node.outputs = {result};
    node.backward = [a, b, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(gradient, a->shape()), "mean-a");
      accumulate(b, reduceToShape(rawMul(gradient, std::make_shared<Tensor>(
                                        Tensor::full(gradient->dtype(), gradient->shape(), -1.0))),
                                    b->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr sub(const Ptr& a, double b) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Sub, *a, b));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(gradient, a->shape()), "mean-a");
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr mul(const Ptr& a, const Ptr& b) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *a, *b));
  if (shouldRecord({a, b})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a, b};
    node.outputs = {result};
    node.backward = [a, b, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(rawMul(gradient, b), a->shape()), "mul-a");
      accumulate(b, reduceToShape(rawMul(gradient, a), b->shape()), "mul-b");
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr mul(const Ptr& a, double b) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *a, b));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, b, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(rawMul(gradient, std::make_shared<Tensor>(
                                              Tensor::full(gradient->dtype(), gradient->shape(), b))),
                                  a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr div(const Ptr& a, const Ptr& b) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Div, *a, *b));
  if (shouldRecord({a, b})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a, b};
    node.outputs = {result};
    node.backward = [a, b, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      // d/da (a / b) = 1 / b
      const Ptr oneOverB = std::make_shared<Tensor>(
          tensor::binaryOp(BinaryOp::Div, *onesLike(b), *b));
      accumulate(a, reduceToShape(rawMul(gradient, oneOverB), a->shape()));
      // d/db (a / b) = -a / b^2
      const Ptr bSquared = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *b, *b));
      const Ptr negative = std::make_shared<Tensor>(
          Tensor::full(gradient->dtype(), a->shape(), -1.0));
      const Ptr term = std::make_shared<Tensor>(
          tensor::binaryOp(BinaryOp::Div, *rawMul(negative, a), *bSquared));
      accumulate(b, reduceToShape(rawMul(gradient, term), b->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr div(const Ptr& a, double b) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Div, *a, b));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, b, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      const Ptr scaled = std::make_shared<Tensor>(
          Tensor::full(gradient->dtype(), gradient->shape(), 1.0 / b));
      accumulate(a, reduceToShape(rawMul(gradient, scaled), a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr matmul(const Ptr& a, const Ptr& b) {
  auto result = std::make_shared<Tensor>(tensor::matmul(*a, *b));
  if (shouldRecord({a, b})) {
    result->setRequiresGrad(true);

    // Work out the two-dimensional shapes, so that one rule covers the
    // vector cases too.
    const bool aIsVector = a->shape().rank() == 1;
    const bool bIsVector = b->shape().rank() == 1;

    Ptr aMatrix = a;
    Ptr bMatrix = b;
    if (aIsVector) {
      aMatrix = std::make_shared<Tensor>(
          a->reshape(Shape{std::vector<std::int64_t>{1, a->shape().dim(0)}}));
    }
    if (bIsVector) {
      bMatrix = std::make_shared<Tensor>(
          b->reshape(Shape{std::vector<std::int64_t>{b->shape().dim(0), 1}}));
    }

    Ptr resultMatrix = result;
    if (aIsVector || bIsVector) {
      std::vector<std::int64_t> dims = result->shape().dims();
      if (aIsVector && bIsVector) {
        dims = {1, 1};
      } else if (aIsVector) {
        dims = {1, result->shape().dim(0)};
      } else {
        dims = {result->shape().dim(0), 1};
      }
      resultMatrix = std::make_shared<Tensor>(result->reshape(Shape(dims)));
    }

    Node node;
    node.inputs = {a, b};
    node.outputs = {result};
    node.backward = [a, b, result, aMatrix, bMatrix, resultMatrix, aIsVector, bIsVector]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;

      Ptr gradientMatrix = gradient;
      if (aIsVector || bIsVector) {
        std::vector<std::int64_t> dims = gradient->shape().dims();
        if (aIsVector && bIsVector) {
          dims = {1, 1};
        } else if (aIsVector) {
          dims = {1, gradient->shape().dim(0)};
        } else {
          dims = {gradient->shape().dim(0), 1};
        }
        gradientMatrix = std::make_shared<Tensor>(gradient->reshape(Shape(dims)));
      }

      // d/da = gradient @ b^T
      Ptr gradA = rawMatmulTimesTranspose(gradientMatrix, bMatrix);
      // d/db = a^T @ gradient
      Ptr gradB = rawTransposeTimesMatmul(aMatrix, gradientMatrix);

      if (aIsVector) {
        gradA = std::make_shared<Tensor>(
            gradA->reshape(Shape{std::vector<std::int64_t>{a->shape().dim(0)}}));
      }
      if (bIsVector) {
        gradB = std::make_shared<Tensor>(
            gradB->reshape(Shape{std::vector<std::int64_t>{b->shape().dim(0)}}));
      }

      accumulate(a, gradA, "matmul-a");
      accumulate(b, gradB, "matmul-b");
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr pow(const Ptr& a, double exponent) {
  auto result = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Pow, *a, exponent));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, exponent, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      // d/da (a^n) = n * a^(n-1)
      Ptr scaled = std::make_shared<Tensor>(
          tensor::binaryOp(BinaryOp::Pow, *a, exponent - 1.0));
      const Ptr factor = std::make_shared<Tensor>(
          Tensor::full(gradient->dtype(), scaled->shape(), exponent));
      accumulate(a, reduceToShape(rawMul(gradient, rawMul(scaled, factor)), a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

namespace {

/// Record a unary operation whose derivative is `derivative(input, output)`.
template <typename Derivative>
Ptr unaryWithDerivative(const Ptr& a, tensor::UnaryOp op, Derivative derivative) {
  auto result = std::make_shared<Tensor>(tensor::unaryOp(op, *a));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result, derivative]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(rawMul(gradient, derivative(a, result)), a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

}  // namespace

Ptr neg(const Ptr& a) {
  return unaryWithDerivative(
      a, UnaryOp::Neg, [](const Ptr& input, const Ptr&) {
        return std::make_shared<Tensor>(
            Tensor::full(input->dtype(), input->shape(), -1.0));
      });
}

Ptr abs(const Ptr& a) {
  return unaryWithDerivative(
      a, UnaryOp::Abs, [](const Ptr& input, const Ptr&) {
        // The derivative of |x| is the sign of x.
        auto sign = std::make_shared<Tensor>(Tensor::zeros(input->dtype(), input->shape()));
        const std::int64_t total = input->numel();
        std::vector<std::int64_t> indices(input->shape().rank(), 0);
        for (std::int64_t i = 0; i < total; ++i) {
          const double value = input->get(indices);
          sign->set(indices, value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0));
          for (std::size_t axis = input->shape().rank(); axis-- > 0;) {
            if (++indices[axis] < input->shape().dim(axis)) break;
            indices[axis] = 0;
          }
        }
        return sign;
      });
}

Ptr relu(const Ptr& a) {
  auto result = std::make_shared<Tensor>(tensor::unaryOp(UnaryOp::Relu, *a));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      // The derivative is one where the input was positive.
      auto mask = std::make_shared<Tensor>(Tensor::zeros(a->dtype(), a->shape()));
      const std::int64_t total = a->numel();
      std::vector<std::int64_t> indices(a->shape().rank(), 0);
      for (std::int64_t i = 0; i < total; ++i) {
        if (a->get(indices) > 0.0) {
          mask->set(indices, 1.0);
        }
        for (std::size_t axis = a->shape().rank(); axis-- > 0;) {
          if (++indices[axis] < a->shape().dim(axis)) break;
          indices[axis] = 0;
        }
      }
      accumulate(a, reduceToShape(rawMul(gradient, mask), a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr gelu(const Ptr& a) {
  auto result = std::make_shared<Tensor>(tensor::unaryOp(UnaryOp::Gelu, *a));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;

      // The tanh approximation of the Gaussian error linear unit.
      constexpr double k = 0.7978845608028654;
      constexpr double c = 0.044715;

      const Ptr xSquared = rawMul(a, a);
      const Ptr xCubed = rawMul(xSquared, a);

      // u = k * (x + c * x^3)
      const Ptr cTerm = std::make_shared<Tensor>(
          tensor::binaryOp(BinaryOp::Mul, *xCubed,
                           *std::make_shared<Tensor>(Tensor::full(a->dtype(), a->shape(), c))));
      const Ptr inner = std::make_shared<Tensor>(
          tensor::binaryOp(BinaryOp::Add, *a, *cTerm));
      const Ptr kTerm = std::make_shared<Tensor>(
          Tensor::full(a->dtype(), a->shape(), k));
      const Ptr u = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *kTerm, *inner));
      const Ptr t = std::make_shared<Tensor>(tensor::unaryOp(UnaryOp::Tanh, *u));

      // 0.5 * (1 + t)
      const Ptr one = std::make_shared<Tensor>(Tensor::ones(a->dtype(), a->shape()));
      const Ptr onePlusT =
          std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Add, *one, *t));
      const Ptr half =
          std::make_shared<Tensor>(Tensor::full(a->dtype(), a->shape(), 0.5));
      const Ptr term1 = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *half, *onePlusT));

      // 0.5 * x * (1 - t^2) * k * (1 + 3 * c * x^2)
      const Ptr oneMinusT2 = std::make_shared<Tensor>(
          tensor::binaryOp(BinaryOp::Sub, *one, *rawMul(t, t)));
      const Ptr threeC = std::make_shared<Tensor>(
          Tensor::full(a->dtype(), a->shape(), 3.0 * c));
      const Ptr duInner = std::make_shared<Tensor>(
          tensor::binaryOp(BinaryOp::Add, *one,
                           *std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *threeC,
                                                                        *xSquared))));
      const Ptr du = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *kTerm, *duInner));

      Ptr term2 = rawMul(a, oneMinusT2);
      term2 = rawMul(term2, du);
      term2 = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *half, *term2));

      const Ptr derivative = rawAdd(term1, term2);
      accumulate(a, reduceToShape(rawMul(gradient, derivative), a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr sigmoid(const Ptr& a) {
  return unaryWithDerivative(
      a, UnaryOp::Sigmoid, [](const Ptr&, const Ptr& output) {
        // y (1 - y)
        const Ptr one = std::make_shared<Tensor>(Tensor::ones(output->dtype(), output->shape()));
        return rawMul(output, std::make_shared<Tensor>(
                                tensor::binaryOp(BinaryOp::Sub, *one, *output)));
      });
}

Ptr tanh(const Ptr& a) {
  return unaryWithDerivative(
      a, UnaryOp::Tanh, [](const Ptr&, const Ptr& output) {
        // 1 - y^2
        const Ptr one = std::make_shared<Tensor>(Tensor::ones(output->dtype(), output->shape()));
        return std::make_shared<Tensor>(
            tensor::binaryOp(BinaryOp::Sub, *one, *rawMul(output, output)));
      });
}

Ptr exp(const Ptr& a) {
  return unaryWithDerivative(
      a, UnaryOp::Exp, [](const Ptr&, const Ptr& output) { return output; });
}

Ptr log(const Ptr& a) {
  return unaryWithDerivative(
      a, UnaryOp::Log, [](const Ptr& input, const Ptr&) {
        // 1 / x
        return std::make_shared<Tensor>(
            tensor::binaryOp(BinaryOp::Div, *onesLike(input), *input));
      });
}

Ptr sqrt(const Ptr& a) {
  return unaryWithDerivative(
      a, UnaryOp::Sqrt, [](const Ptr&, const Ptr& output) {
        // 0.5 / y
        const Ptr half = std::make_shared<Tensor>(
            Tensor::full(output->dtype(), output->shape(), 0.5));
        return std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Div, *half, *output));
      });
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

Ptr sum(const Ptr& a) {
  auto result = std::make_shared<Tensor>(tensor::sum(*a));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(gradient, a->shape()), "mean-a");
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr mean(const Ptr& a) {
  auto result = std::make_shared<Tensor>(tensor::mean(*a));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      const double count = static_cast<double>(a->numel());
      const Ptr scaled = std::make_shared<Tensor>(
          Tensor::full(gradient->dtype(), gradient->shape(), 1.0 / count));
      accumulate(a, reduceToShape(rawMul(gradient, scaled), a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

namespace {

/// The gradient of a maximum or a minimum: it all goes to the winning
/// element.
Ptr extremumGradient(const Ptr& a, const Ptr& result, bool wantMaximum) {
  const Ptr& gradient = result->grad();
  if (!gradient) return nullptr;

  Ptr mask = std::make_shared<Tensor>(Tensor::zeros(a->dtype(), a->shape()));
  const std::int64_t total = a->numel();
  std::vector<std::int64_t> indices(a->shape().rank(), 0);
  bool found = false;
  double best = 0.0;
  for (std::int64_t i = 0; i < total; ++i) {
    const double value = a->get(indices);
    if (!found || (wantMaximum ? value > best : value < best)) {
      best = value;
      found = true;
      mask = std::make_shared<Tensor>(Tensor::zeros(a->dtype(), a->shape()));
      mask->set(indices, 1.0);
    }
    for (std::size_t axis = a->shape().rank(); axis-- > 0;) {
      if (++indices[axis] < a->shape().dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return rawMul(gradient, mask);
}

}  // namespace

Ptr max(const Ptr& a) {
  auto result = std::make_shared<Tensor>(tensor::max(*a));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr gradient = extremumGradient(a, result, true);
      if (!gradient) return;
      accumulate(a, gradient);
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr min(const Ptr& a) {
  auto result = std::make_shared<Tensor>(tensor::min(*a));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr gradient = extremumGradient(a, result, false);
      if (!gradient) return;
      accumulate(a, gradient);
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr argmax(const Ptr& a) {
  // The index of the largest element is not differentiable in a useful
  // way, so this never records.
  return std::make_shared<Tensor>(tensor::argmax(*a));
}

// ---------------------------------------------------------------------------
// Shape operations
// ---------------------------------------------------------------------------

Ptr softmax(const Ptr& a) {
  auto result = std::make_shared<Tensor>(tensor::softmax(*a));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;

      // y = softmax(x), so dy/dx = y * (g - sum(g * y))
      const std::int64_t lastAxis = static_cast<std::int64_t>(a->shape().rank()) - 1;
      const Ptr product = rawMul(gradient, result);
      const Ptr dot = std::make_shared<Tensor>(tensor::sumOverAxis(*product, lastAxis));
      const Ptr dotExpanded = broadcastTo(collapseLastAxis(dot), a->shape());

      const Ptr difference = std::make_shared<Tensor>(
          tensor::binaryOp(BinaryOp::Sub, *gradient, *dotExpanded));
      accumulate(a, reduceToShape(rawMul(result, difference), a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr reshape(const Ptr& a, const Shape& shape) {
  auto result = std::make_shared<Tensor>(a->reshape(shape));
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(
                       std::make_shared<Tensor>(gradient->reshape(a->shape())), a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr transpose(const Ptr& a) {
  auto result = std::make_shared<Tensor>(a->transpose());
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(gradient, a->shape()), "mean-a");
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

Ptr contiguous(const Ptr& a) {
  auto result = std::make_shared<Tensor>(a->contiguous());
  if (shouldRecord({a})) {
    result->setRequiresGrad(true);
    Node node;
    node.inputs = {a};
    node.outputs = {result};
    node.backward = [a, result]() {
      const Ptr& gradient = result->grad();
      if (!gradient) return;
      accumulate(a, reduceToShape(std::make_shared<Tensor>(gradient->reshape(a->shape())),
                                   a->shape()));
    };
    Tape::current().push(std::move(node));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------

Ptr mseLoss(const Ptr& prediction, const Ptr& target) {
  const Ptr difference = sub(prediction, target);
  const Ptr squared = mul(difference, difference);
  return mean(squared);
}

Ptr crossEntropy(const Ptr& logits, const Ptr& target) {
  // The loss is the mean of -sum(target * log(softmax(logits))).
  const Ptr probabilities = softmax(logits);
  const Ptr logProbabilities = log(probabilities);
  const Ptr product = mul(target, logProbabilities);
  const Ptr perRow = std::make_shared<Tensor>(
      tensor::sumOverAxis(*product, static_cast<std::int64_t>(product->shape().rank()) - 1));
  const Ptr negative = std::make_shared<Tensor>(
      Tensor::full(perRow->dtype(), perRow->shape(), -1.0));
  const Ptr negated = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *negative, *perRow));
  return mean(negated);
}

}  // namespace milo::autograd
