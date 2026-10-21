#include "runtime/modules.hpp"

#include <map>

#include "autograd/modules.hpp"
#include "autograd/ops.hpp"
#include "runtime/interp.hpp"

namespace milo::runtime {

Value moduleForward(const ModuleValue& module, const Value& input) {
  if (input.kind() != Value::Kind::Tensor) {
    throw milo::RuntimeError{"a layer needs a tensor as its input, not a " +
                           std::string(input.typeName()),
                       SourceSpan::start(), {}};
  }

  autograd::Layer layer;
  layer.kind = module.kind;
  for (const auto& entry : module.parameters) {
    layer.parameters[entry.first] = entry.second;
  }
  for (const auto& entry : module.buffers) {
    layer.buffers[entry.first] = entry.second;
  }
  layer.quantizedWeight = module.quantizedWeight;

  try {
    const autograd::Ptr result = autograd::forward(layer, input.asTensor());
    // The result is shared, not copied: the gradient graph has to stay
    // connected to the tensor the program holds.
    return Value::tensor(result);
  } catch (const std::exception& error) {
    throw milo::RuntimeError{error.what(), SourceSpan::start(), {}};
  }
}

void optimizerStep(OptimizerValue& optimizer) {
  autograd::Optimizer state;
  state.kind = optimizer.kind;
  state.parameters = optimizer.parameters;
  state.learning_rate = optimizer.learning_rate;
  state.beta1 = optimizer.beta1;
  state.beta2 = optimizer.beta2;
  state.epsilon = optimizer.epsilon;
  state.weight_decay = optimizer.weight_decay;
  // The state is shared, so the step count and the moments carry over.
  state.state = optimizer.state;

  autograd::step(state);
}

void optimizerZeroGrad(OptimizerValue& optimizer) {
  for (const auto& parameter : optimizer.parameters) {
    if (parameter->grad()) parameter->zeroGrad();
  }
}

Value moduleParameters(const ModuleValue& module) {
  Value::Dict entries;
  for (const auto& entry : module.parameters) {
    entries.emplace_back(entry.first, Value::tensor(entry.second));
  }
  return Value::dict(std::move(entries));
}

}  // namespace milo::runtime
