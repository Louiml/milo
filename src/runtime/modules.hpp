// Layers and optimisers, as the language sees them.
#ifndef MILO_RUNTIME_MODULES_HPP
#define MILO_RUNTIME_MODULES_HPP

#include <memory>
#include <string>
#include <vector>

#include "runtime/value.hpp"
#include "tensor/tensor.hpp"

namespace milo::runtime {

/// Run a layer's forward pass.
Value moduleForward(const ModuleValue& module, const Value& input);

/// Update every parameter that has a gradient.
void optimizerStep(OptimizerValue& optimizer);

/// Set every parameter's gradient to zero.
void optimizerZeroGrad(OptimizerValue& optimizer);

/// The parameters of a layer, as a dictionary of tensors.
Value moduleParameters(const ModuleValue& module);

}  // namespace milo::runtime

#endif  // MILO_RUNTIME_MODULES_HPP
