// The mutable state of an optimiser.
//
// It lives in its own header because two layers need it: the autograd
// module that does the maths, and the runtime value that the program
// holds. The state is shared, so that a copy of the optimiser — which is
// what a method call makes — still sees the same moments and the same
// step count.
#ifndef MILO_AUTOGRAD_STATE_HPP
#define MILO_AUTOGRAD_STATE_HPP

#include <memory>
#include <vector>

#include "tensor/tensor.hpp"

namespace milo::autograd {

using tensor::Tensor;

/// The state an optimiser carries between steps.
struct OptimizerState {
  /// The first moment of the gradient, for Adam and AdamW.
  std::vector<std::shared_ptr<Tensor>> first_moment;
  /// The second moment of the gradient, for Adam and AdamW.
  std::vector<std::shared_ptr<Tensor>> second_moment;
  /// How many times the optimiser has stepped.
  int step = 0;
};

}  // namespace milo::autograd

#endif  // MILO_AUTOGRAD_STATE_HPP
