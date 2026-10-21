// The training builtins: layers, optimisers, and losses.
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "autograd/modules.hpp"
#include "autograd/ops.hpp"
#include "runtime/builtins.hpp"
#include "runtime/modules.hpp"
#include "tensor/ops.hpp"
#include "tensor/quantization.hpp"
#include "planner/planner.hpp"
#include "tensor/random.hpp"

namespace milo::runtime {
namespace {

using autograd::Ptr;

/// Fetch a tensor argument.
Ptr wantTensor(const Value& value, const char* what) {
  if (value.kind() != Value::Kind::Tensor) {
    throw RuntimeError{std::string(what) + " needs a tensor, not a " + value.typeName(),
                       SourceSpan::start(), {}};
  }
  return value.asTensor();
}

/// The tensors of a dictionary or a list, in a stable order.
std::vector<Ptr> collectTensors(const Value& value, const char* what) {
  std::vector<Ptr> tensors;
  if (value.kind() == Value::Kind::Dict) {
    for (const auto& entry : value.asDict()) {
      if (entry.second.kind() != Value::Kind::Tensor) {
        throw RuntimeError{std::string(what) + " needs a dictionary of tensors",
                           SourceSpan::start(), {}};
      }
      tensors.push_back(entry.second.asTensor());
    }
    return tensors;
  }
  if (value.kind() == Value::Kind::List) {
    for (const Value& item : value.asList()) {
      if (item.kind() != Value::Kind::Tensor) {
        throw RuntimeError{std::string(what) + " needs a list of tensors",
                           SourceSpan::start(), {}};
      }
      tensors.push_back(item.asTensor());
    }
    return tensors;
  }
  throw RuntimeError{std::string(what) + " needs a dictionary or a list of tensors",
                     SourceSpan::start(), {}};
}

double numberArgument(const Value& value, const char* what) {
  if (value.kind() == Value::Kind::Int) return static_cast<double>(value.asInt());
  if (value.kind() == Value::Kind::Float) return value.asFloat();
  throw RuntimeError{std::string(what) + " needs a number", SourceSpan::start(), {}};
}

/// A keyword argument, if it is there.
const Value* keyword(const std::vector<std::pair<std::string, Value>>& kwargs,
                     const std::string& name) {
  for (const auto& entry : kwargs) {
    if (entry.first == name) return &entry.second;
  }
  return nullptr;
}

// --- layers ---------------------------------------------------------------

Value builtinLinear(Interpreter&, const std::vector<Value>& args,
                    const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"linear needs an input size and an output size",
                       SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::Int || args[1].kind() != Value::Kind::Int) {
    throw RuntimeError{"linear needs two whole numbers", SourceSpan::start(), {}};
  }

  const autograd::Layer layer = autograd::linear(args[0].asInt(), args[1].asInt());

  ModuleValue module;
  module.kind = layer.kind;
  for (const auto& entry : layer.parameters) {
    module.parameters[entry.first] = entry.second;
  }
  for (const auto& entry : layer.buffers) {
    module.buffers[entry.first] = entry.second;
  }
  return Value::layer(std::move(module));
}

Value builtinLayerNorm(Interpreter&, const std::vector<Value>& args,
                       const std::vector<std::pair<std::string, Value>>& kwargs) {
  if (args.empty() || args[0].kind() != Value::Kind::Int) {
    throw RuntimeError{"layer_norm needs a size", SourceSpan::start(), {}};
  }
  double epsilon = 1e-5;
  if (const Value* value = keyword(kwargs, "epsilon")) {
    epsilon = numberArgument(*value, "epsilon");
  }

  const autograd::Layer layer = autograd::layerNorm(args[0].asInt(), epsilon);

  ModuleValue module;
  module.kind = layer.kind;
  for (const auto& entry : layer.parameters) {
    module.parameters[entry.first] = entry.second;
  }
  for (const auto& entry : layer.buffers) {
    module.buffers[entry.first] = entry.second;
  }
  return Value::layer(std::move(module));
}

// --- optimisers -----------------------------------------------------------

Value makeOptimizer(const autograd::Optimizer& optimizer) {
  OptimizerValue value;
  value.kind = optimizer.kind;
  value.parameters = optimizer.parameters;
  value.learning_rate = optimizer.learning_rate;
  value.beta1 = optimizer.beta1;
  value.beta2 = optimizer.beta2;
  value.epsilon = optimizer.epsilon;
  value.weight_decay = optimizer.weight_decay;
  value.state = optimizer.state;
  return Value::optimizer(std::move(value));
}

Value builtinSgd(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() < 2) {
    throw RuntimeError{"sgd needs some parameters and a learning rate",
                       SourceSpan::start(), {}};
  }
  const std::vector<Ptr> parameters = collectTensors(args[0], "sgd");
  const double learningRate = numberArgument(args[1], "the learning rate");
  return makeOptimizer(autograd::sgd(parameters, learningRate));
}

Value builtinAdam(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() < 2) {
    throw RuntimeError{"adam needs some parameters and a learning rate",
                       SourceSpan::start(), {}};
  }
  const std::vector<Ptr> parameters = collectTensors(args[0], "adam");
  const double learningRate = numberArgument(args[1], "the learning rate");
  return makeOptimizer(autograd::adam(parameters, learningRate));
}

Value builtinAdamW(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() < 2) {
    throw RuntimeError{"adamw needs some parameters and a learning rate",
                       SourceSpan::start(), {}};
  }
  const std::vector<Ptr> parameters = collectTensors(args[0], "adamw");
  const double learningRate = numberArgument(args[1], "the learning rate");
  double weightDecay = 0.01;
  if (args.size() >= 3) {
    weightDecay = numberArgument(args[2], "the weight decay");
  }
  return makeOptimizer(autograd::adamw(parameters, learningRate, weightDecay));
}

// --- losses and gradients -------------------------------------------------

Value builtinMseLoss(Interpreter&, const std::vector<Value>& args,
                     const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"mse_loss needs a prediction and a target", SourceSpan::start(), {}};
  }
  const Ptr prediction = wantTensor(args[0], "mse_loss");
  const Ptr target = wantTensor(args[1], "mse_loss");
  try {
    const Ptr loss = autograd::mseLoss(prediction, target);
    return Value::tensor(loss);
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
}

Value builtinCrossEntropy(Interpreter&, const std::vector<Value>& args,
                          const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"cross_entropy needs logits and a target", SourceSpan::start(), {}};
  }
  const Ptr logits = wantTensor(args[0], "cross_entropy");
  const Ptr target = wantTensor(args[1], "cross_entropy");
  try {
    const Ptr loss = autograd::crossEntropy(logits, target);
    return Value::tensor(loss);
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
}

Value builtinTrainable(Interpreter&, const std::vector<Value>& args,
                       const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"trainable needs one tensor", SourceSpan::start(), {}};
  }
  const Ptr tensor = wantTensor(args[0], "trainable");
  return Value::tensor(autograd::trainable(tensor));
}

Value builtinBackward(Interpreter&, const std::vector<Value>& args,
                      const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"backward needs one tensor", SourceSpan::start(), {}};
  }
  const Ptr tensor = wantTensor(args[0], "backward");
  try {
    autograd::Tape::current().backward(tensor);
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
  return Value::null();
}

Value builtinParameters(Interpreter&, const std::vector<Value>& args,
                        const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"parameters needs one layer", SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::Module) {
    throw RuntimeError{"parameters needs a layer", SourceSpan::start(), {}};
  }
  return moduleParameters(args[0].asModule());
}

Value builtinQuantize(Interpreter&, const std::vector<Value>& args,
                      const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"quantize needs a tensor and a scheme", SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::Tensor) {
    throw RuntimeError{"quantize needs a tensor", SourceSpan::start(), {}};
  }
  if (args[1].kind() != Value::Kind::String) {
    throw RuntimeError{"quantize needs a scheme: 'int8' or 'nf4'", SourceSpan::start(), {}};
  }
  const std::string scheme = args[1].asString();
  try {
    if (scheme == "int8") {
      return Value::quantized(std::make_shared<tensor::QuantizedTensor>(
          tensor::quantizeInt8(*args[0].asTensor())));
    }
    if (scheme == "nf4") {
      return Value::quantized(std::make_shared<tensor::QuantizedTensor>(
          tensor::quantizeNf4(*args[0].asTensor())));
    }
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
  throw RuntimeError{"'" + scheme + "' is not a quantisation scheme; use 'int8' or 'nf4'",
                     SourceSpan::start(), {}};
}

Value builtinDequantize(Interpreter&, const std::vector<Value>& args,
                         const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"dequantize needs one quantised tensor", SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::Quantized) {
    throw RuntimeError{"dequantize needs a quantised tensor", SourceSpan::start(), {}};
  }
  try {
    return Value::tensor(std::make_shared<tensor::Tensor>(
        tensor::dequantize(*args[0].asQuantized())));
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
}

Value builtinLora(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>& kwargs) {
  if (args.size() < 3) {
    throw RuntimeError{"lora needs an input size, an output size, and a rank",
                       SourceSpan::start(), {}};
  }
  for (std::size_t i = 0; i < 3; ++i) {
    if (args[i].kind() != Value::Kind::Int) {
      throw RuntimeError{"lora needs three whole numbers", SourceSpan::start(), {}};
    }
  }
  double alpha = 16.0;
  if (const Value* value = keyword(kwargs, "alpha")) {
    alpha = numberArgument(*value, "alpha");
  }

  try {
    const autograd::Layer layer =
        autograd::lora(args[0].asInt(), args[1].asInt(), args[2].asInt(), alpha);
    ModuleValue module;
    module.kind = layer.kind;
    for (const auto& entry : layer.parameters) {
      module.parameters[entry.first] = entry.second;
    }
    for (const auto& entry : layer.buffers) {
      module.buffers[entry.first] = entry.second;
    }
    return Value::layer(std::move(module));
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
}

Value builtinQlora(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>& kwargs) {
  if (args.size() < 3) {
    throw RuntimeError{"qlora needs an input size, an output size, and a rank",
                       SourceSpan::start(), {}};
  }
  for (std::size_t i = 0; i < 3; ++i) {
    if (args[i].kind() != Value::Kind::Int) {
      throw RuntimeError{"qlora needs three whole numbers", SourceSpan::start(), {}};
    }
  }
  double alpha = 16.0;
  if (const Value* value = keyword(kwargs, "alpha")) {
    alpha = numberArgument(*value, "alpha");
  }

  try {
    const autograd::Layer layer =
        autograd::qlora(args[0].asInt(), args[1].asInt(), args[2].asInt(), alpha);
    ModuleValue module;
    module.kind = layer.kind;
    for (const auto& entry : layer.parameters) {
      module.parameters[entry.first] = entry.second;
    }
    for (const auto& entry : layer.buffers) {
      module.buffers[entry.first] = entry.second;
    }
    module.quantizedWeight = layer.quantizedWeight;
    return Value::layer(std::move(module));
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
}

Value builtinMerge(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"merge needs one layer", SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::Module) {
    throw RuntimeError{"merge needs a layer", SourceSpan::start(), {}};
  }

  const ModuleValue& module = args[0].asModule();
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
    const autograd::Layer merged = autograd::mergeLora(layer);
    ModuleValue result;
    result.kind = merged.kind;
    for (const auto& entry : merged.parameters) {
      result.parameters[entry.first] = entry.second;
    }
    for (const auto& entry : merged.buffers) {
      result.buffers[entry.first] = entry.second;
    }
    return Value::layer(std::move(result));
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
}

namespace {

/// Read a keyword argument as a number.
bool numberKeyword(const std::vector<std::pair<std::string, Value>>& kwargs,
                   const std::string& name, double& out) {
  for (const auto& entry : kwargs) {
    if (entry.first != name) continue;
    if (entry.second.kind() == Value::Kind::Int) {
      out = static_cast<double>(entry.second.asInt());
      return true;
    }
    if (entry.second.kind() == Value::Kind::Float) {
      out = entry.second.asFloat();
      return true;
    }
    return false;
  }
  return false;
}

/// Read a keyword argument as a string.
bool stringKeyword(const std::vector<std::pair<std::string, Value>>& kwargs,
                   const std::string& name, std::string& out) {
  for (const auto& entry : kwargs) {
    if (entry.first != name) continue;
    if (entry.second.kind() != Value::Kind::String) return false;
    out = entry.second.asString();
    return true;
  }
  return false;
}

/// Read a keyword argument as a flag.
bool flagKeyword(const std::vector<std::pair<std::string, Value>>& kwargs,
                 const std::string& name) {
  for (const auto& entry : kwargs) {
    if (entry.first != name) continue;
    return entry.second.kind() == Value::Kind::Bool ? entry.second.asBool() : true;
  }
  return false;
}

}  // namespace

Value builtinMemoryPlan(Interpreter&, const std::vector<Value>& args,
                         const std::vector<std::pair<std::string, Value>>& kwargs) {
  if (args.size() != 1) {
    throw RuntimeError{"memory_plan needs a model or a parameter count",
                       SourceSpan::start(), {}};
  }

  planner::Config config;

  if (args[0].kind() == Value::Kind::Model) {
    config.parameters = args[0].asModel().parameterCount();
  } else if (args[0].kind() == Value::Kind::Int) {
    config.parameters = args[0].asInt();
  } else {
    throw RuntimeError{"memory_plan needs a model or a parameter count",
                       SourceSpan::start(), {}};
  }

  double number = 0.0;
  if (numberKeyword(kwargs, "hidden", number)) config.hidden = static_cast<std::int64_t>(number);
  if (numberKeyword(kwargs, "layers", number)) config.layers = static_cast<std::int64_t>(number);
  if (numberKeyword(kwargs, "heads", number)) config.heads = static_cast<std::int64_t>(number);
  if (numberKeyword(kwargs, "batch_size", number)) config.batch = static_cast<std::int64_t>(number);
  if (numberKeyword(kwargs, "sequence", number)) {
    config.sequence = static_cast<std::int64_t>(number);
  }
  if (numberKeyword(kwargs, "lora_rank", number)) {
    config.loraRank = static_cast<std::int64_t>(number);
  }
  if (numberKeyword(kwargs, "gradient_accumulation", number)) {
    config.gradientAccumulation = static_cast<std::int64_t>(number);
  }
  if (numberKeyword(kwargs, "device_memory", number)) {
    // The device's memory is given in gibibytes.
    config.deviceMemory = static_cast<std::int64_t>(number * 1024.0 * 1024.0 * 1024.0);
  }

  std::string text;
  if (stringKeyword(kwargs, "quantization", text)) {
    config.quantization = planner::quantizationFromName(text);
  }
  if (stringKeyword(kwargs, "device", text)) config.device = text;

  config.lora = flagKeyword(kwargs, "lora");
  config.gradientCheckpointing = flagKeyword(kwargs, "gradient_checkpointing");
  config.offloadOptimizer = flagKeyword(kwargs, "offload_optimizer");

  const planner::Plan plan = planner::plan(config);

  Value::Dict result;
  result.emplace_back("fits", Value::boolean(plan.fits));
  result.emplace_back("summary", Value::string(plan.summary));
  result.emplace_back("total", Value::integer(plan.estimate.total()));
  result.emplace_back("weights", Value::integer(plan.estimate.weights));
  result.emplace_back("gradients", Value::integer(plan.estimate.gradients));
  result.emplace_back("optimizer", Value::integer(plan.estimate.optimizer));
  result.emplace_back("activations", Value::integer(plan.estimate.activations));
  result.emplace_back("attention", Value::integer(plan.estimate.attention));
  result.emplace_back("scratch", Value::integer(plan.estimate.scratch));
  result.emplace_back("trainable", Value::integer(planner::trainableParameters(config)));

  Value::List suggestions;
  for (const planner::Suggestion& suggestion : plan.suggestions) {
    Value::Dict entry;
    entry.emplace_back("knob", Value::string(suggestion.knob));
    entry.emplace_back("from", Value::string(suggestion.from));
    entry.emplace_back("to", Value::string(suggestion.to));
    entry.emplace_back("reason", Value::string(suggestion.reason));
    suggestions.push_back(Value::dict(std::move(entry)));
  }
  result.emplace_back("suggestions", Value::list(std::move(suggestions)));

  return Value::dict(std::move(result));
}

}  // namespace

void installTrainingBuiltins(Interpreter& interpreter) {
  interpreter.defineBuiltin("linear", &builtinLinear);
  interpreter.defineBuiltin("layer_norm", &builtinLayerNorm);

  interpreter.defineBuiltin("sgd", &builtinSgd);
  interpreter.defineBuiltin("adam", &builtinAdam);
  interpreter.defineBuiltin("adamw", &builtinAdamW);

  interpreter.defineBuiltin("mse_loss", &builtinMseLoss);
  interpreter.defineBuiltin("cross_entropy", &builtinCrossEntropy);
  interpreter.defineBuiltin("trainable", &builtinTrainable);
  interpreter.defineBuiltin("backward", &builtinBackward);
  interpreter.defineBuiltin("parameters", &builtinParameters);
  interpreter.defineBuiltin("quantize", &builtinQuantize);
  interpreter.defineBuiltin("dequantize", &builtinDequantize);
  interpreter.defineBuiltin("lora", &builtinLora);
  interpreter.defineBuiltin("qlora", &builtinQlora);
  interpreter.defineBuiltin("merge", &builtinMerge);
  interpreter.defineBuiltin("memory_plan", &builtinMemoryPlan);
}

}  // namespace milo::runtime
