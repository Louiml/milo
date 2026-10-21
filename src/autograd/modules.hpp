// Layers and optimisers.
#ifndef MILO_AUTOGRAD_MODULES_HPP
#define MILO_AUTOGRAD_MODULES_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "autograd/ops.hpp"
#include "autograd/state.hpp"
#include "tensor/quantization.hpp"

namespace milo::autograd {

/// A layer: a group of named parameters with a forward pass.
struct Layer {
  /// `linear`, `layer_norm`, `lora`, ...
  std::string kind;
  /// The parameters the optimiser should update.
  std::map<std::string, Ptr> parameters;
  /// The parameters that are trained but not updated.
  std::map<std::string, Ptr> buffers;
  /// The base weight, when it is quantised.
  ///
  /// This is what makes QLoRA possible: the base weights stay small, and
  /// only the low-rank adapters are trained.
  std::shared_ptr<tensor::QuantizedTensor> quantizedWeight;
};

/// A linear layer: `y = x @ weight + bias`.
///
/// The weight is `[inputs, outputs]` and the bias is `[outputs]`. Both are
/// trainable.
Layer linear(std::int64_t inputs, std::int64_t outputs);

/// A LoRA layer: a linear layer with a low-rank update on top.
///
/// The base weight and the bias are frozen. Two small matrices are
/// trained: `lora_a` of `[inputs, rank]` and `lora_b` of
/// `[rank, outputs]`. The update is `scaling * (a @ b)`, where the
/// scaling is `alpha / rank`.
///
/// This is the parameter-efficient fine-tuning trick: the number of
/// trained parameters falls by three orders of magnitude, so a big model
/// can be fine-tuned on a small card.
Layer lora(std::int64_t inputs, std::int64_t outputs, std::int64_t rank,
           double alpha = 16.0);

/// A LoRA layer over a quantised base weight: QLoRA.
Layer qlora(std::int64_t inputs, std::int64_t outputs, std::int64_t rank,
            double alpha = 16.0);

/// Merge a LoRA layer's adapters into its base weight.
///
/// The result is a plain linear layer with the same forward pass and no
/// adapters, which is what you want to save.
Layer mergeLora(const Layer& layer);

/// A layer normalisation: `y = (x - mean) / sqrt(var + epsilon) * scale + shift`.
Layer layerNorm(std::int64_t size, double epsilon = 1e-5);

/// The forward pass of a layer.
Ptr forward(const Layer& layer, const Ptr& input);

/// An optimiser: it holds the parameters and knows how to update them.
struct Optimizer {
  /// `sgd`, `adam`, `adamw`.
  std::string kind;
  std::vector<Ptr> parameters;
  double learning_rate = 0.001;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  double weight_decay = 0.0;
  /// The moments and the step count, shared with whoever holds this
  /// optimiser.
  std::shared_ptr<OptimizerState> state = std::make_shared<OptimizerState>();
};

/// A stochastic gradient descent optimiser.
Optimizer sgd(const std::vector<Ptr>& parameters, double learning_rate);

/// The Adam optimiser.
Optimizer adam(const std::vector<Ptr>& parameters, double learning_rate);

/// The AdamW optimiser: Adam with the weight decay taken out of the
/// gradient.
Optimizer adamw(const std::vector<Ptr>& parameters, double learning_rate,
                double weight_decay = 0.01);

/// Set every parameter's gradient to zero.
void zeroGrad(Optimizer& optimizer);

/// Update every parameter that has a gradient.
void step(Optimizer& optimizer);

}  // namespace milo::autograd

#endif  // MILO_AUTOGRAD_MODULES_HPP
