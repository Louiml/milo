// The differentiable operations.
//
// Each of these computes the value with the raw tensor engine and, when
// the inputs ask for a gradient, records how to undo it. The raw engine
// stays free of any autograd knowledge: `milo.tensor` ops never record,
// and these are the ones that do.
#ifndef MILO_AUTOGRAD_OPS_HPP
#define MILO_AUTOGRAD_OPS_HPP

#include <memory>

#include "autograd/tape.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

namespace milo::autograd {

using tensor::Tensor;

using Ptr = std::shared_ptr<Tensor>;

/// Add a gradient into a tensor's accumulated gradient.
void accumulate(const Ptr& target, const Ptr& gradient, const char* label = "");

/// Sum a gradient down until it has the shape of `target`.
///
/// A gradient that was broadcast on the way out has to be summed back over
/// the axes that were broadcast.
Ptr reduceToShape(const Ptr& gradient, const tensor::Shape& target);

/// A tensor of ones, the same shape and type as `a`.
Ptr onesLike(const Ptr& a);

/// A tensor of zeros, the same shape and type as `a`.
Ptr zerosLike(const Ptr& a);

/// Spread a gradient over a larger shape, by multiplying it by a tensor of
/// ones. This is how a per-row quantity is spread back over a row.
Ptr broadcastTo(const Ptr& gradient, const tensor::Shape& target);

/// Give a tensor's last dimension a size of one, so that it can be
/// broadcast back over that dimension.
Ptr collapseLastAxis(const Ptr& tensor);

/// Mark a tensor as one that takes part in differentiation.
Ptr trainable(Ptr tensor);

// --- the operations -------------------------------------------------------

Ptr add(const Ptr& a, const Ptr& b);
Ptr add(const Ptr& a, double b);
Ptr sub(const Ptr& a, const Ptr& b);
Ptr sub(const Ptr& a, double b);
Ptr mul(const Ptr& a, const Ptr& b);
Ptr mul(const Ptr& a, double b);
Ptr div(const Ptr& a, const Ptr& b);
Ptr div(const Ptr& a, double b);
Ptr matmul(const Ptr& a, const Ptr& b);
Ptr pow(const Ptr& a, double exponent);

Ptr neg(const Ptr& a);
Ptr abs(const Ptr& a);
Ptr relu(const Ptr& a);
Ptr gelu(const Ptr& a);
Ptr sigmoid(const Ptr& a);
Ptr tanh(const Ptr& a);
Ptr exp(const Ptr& a);
Ptr log(const Ptr& a);
Ptr sqrt(const Ptr& a);

Ptr sum(const Ptr& a);
Ptr mean(const Ptr& a);
Ptr max(const Ptr& a);
Ptr min(const Ptr& a);
Ptr argmax(const Ptr& a);

Ptr softmax(const Ptr& a);
Ptr reshape(const Ptr& a, const tensor::Shape& shape);
Ptr transpose(const Ptr& a);
Ptr contiguous(const Ptr& a);

/// The mean squared error between two tensors.
Ptr mseLoss(const Ptr& prediction, const Ptr& target);

/// The cross entropy between logits and a target distribution.
Ptr crossEntropy(const Ptr& logits, const Ptr& target);

}  // namespace milo::autograd

#endif  // MILO_AUTOGRAD_OPS_HPP
