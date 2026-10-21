#include "autograd/modules.hpp"

#include <cmath>
#include <stdexcept>

#include "tensor/random.hpp"

namespace milo::autograd {

using tensor::BinaryOp;
using tensor::DType;
using tensor::Shape;
using tensor::UnaryOp;

Layer linear(std::int64_t inputs, std::int64_t outputs) {
  Layer layer;
  layer.kind = "linear";

  // The weight starts small, so that the layer's output is not enormous.
  const double scale = 1.0 / std::sqrt(static_cast<double>(inputs));
  auto weight = std::make_shared<Tensor>(Tensor::zeros(
      DType::F32, Shape{std::vector<std::int64_t>{inputs, outputs}}));
  for (std::int64_t i = 0; i < weight->numel(); ++i) {
    weight->setFlat(i, tensor::randomNormal() * scale);
  }
  weight->setRequiresGrad(true);

  auto bias = std::make_shared<Tensor>(
      Tensor::zeros(DType::F32, Shape{std::vector<std::int64_t>{outputs}}));
  bias->setRequiresGrad(true);

  layer.parameters["weight"] = weight;
  layer.parameters["bias"] = bias;
  return layer;
}

Layer layerNorm(std::int64_t size, double epsilon) {
  Layer layer;
  layer.kind = "layer_norm";

  auto scale = std::make_shared<Tensor>(Tensor::ones(
      DType::F32, Shape{std::vector<std::int64_t>{size}}));
  scale->setRequiresGrad(true);

  auto shift = std::make_shared<Tensor>(Tensor::zeros(
      DType::F32, Shape{std::vector<std::int64_t>{size}}));
  shift->setRequiresGrad(true);

  layer.parameters["scale"] = scale;
  layer.parameters["shift"] = shift;
  layer.buffers["epsilon"] = std::make_shared<Tensor>(
      Tensor::full(DType::F32, Shape::scalar(), epsilon));
  return layer;
}

Layer lora(std::int64_t inputs, std::int64_t outputs, std::int64_t rank, double alpha) {
  if (rank <= 0) {
    throw std::invalid_argument("the LoRA rank must be positive");
  }

  Layer layer;
  layer.kind = "lora";

  // The base weight and the bias are frozen: they are not trained.
  const double scale = 1.0 / std::sqrt(static_cast<double>(inputs));
  auto weight = std::make_shared<Tensor>(Tensor::zeros(
      DType::F32, Shape{std::vector<std::int64_t>{inputs, outputs}}));
  for (std::int64_t i = 0; i < weight->numel(); ++i) {
    weight->setFlat(i, tensor::randomNormal() * scale);
  }

  auto bias = std::make_shared<Tensor>(
      Tensor::zeros(DType::F32, Shape{std::vector<std::int64_t>{outputs}}));

  // The two adapters are what gets trained. The first is small, so the
  // update starts at nothing.
  auto a = std::make_shared<Tensor>(Tensor::zeros(
      DType::F32, Shape{std::vector<std::int64_t>{inputs, rank}}));
  for (std::int64_t i = 0; i < a->numel(); ++i) {
    a->setFlat(i, tensor::randomNormal() * scale);
  }
  a->setRequiresGrad(true);

  auto b = std::make_shared<Tensor>(Tensor::zeros(
      DType::F32, Shape{std::vector<std::int64_t>{rank, outputs}}));
  b->setRequiresGrad(true);

  layer.parameters["weight"] = weight;
  layer.parameters["bias"] = bias;
  layer.parameters["lora_a"] = a;
  layer.parameters["lora_b"] = b;
  layer.buffers["scaling"] = std::make_shared<Tensor>(
      Tensor::full(DType::F32, Shape::scalar(), alpha / static_cast<double>(rank)));
  return layer;
}

Layer qlora(std::int64_t inputs, std::int64_t outputs, std::int64_t rank, double alpha) {
  Layer layer = lora(inputs, outputs, rank, alpha);
  layer.kind = "qlora";

  // The base weight is stored in NF4: half a byte per weight, plus a
  // scale for every block of 64.
  const Ptr& weight = layer.parameters["weight"];
  layer.quantizedWeight = std::make_shared<tensor::QuantizedTensor>(
      tensor::quantizeNf4(*weight));

  // The frozen base is no longer held in f32.
  layer.parameters["weight"] = std::make_shared<Tensor>(
      Tensor::zeros(DType::F32, Shape{std::vector<std::int64_t>{inputs, outputs}}));

  return layer;
}

Layer mergeLora(const Layer& layer) {
  if (layer.kind != "lora" && layer.kind != "qlora") {
    throw std::invalid_argument("this layer has no adapters to merge");
  }

  const Ptr& a = layer.parameters.at("lora_a");
  const Ptr& b = layer.parameters.at("lora_b");
  const Ptr& scaling = layer.buffers.at("scaling");

  // The merged weight is base + scaling * (a @ b).
  const Ptr update = mul(matmul(a, b), scaling);
  Ptr merged = add(layer.parameters.at("weight"), update);

  Layer result;
  result.kind = "linear";
  result.parameters["weight"] = merged;
  result.parameters["bias"] = layer.parameters.at("bias");
  return result;
}

Ptr forward(const Layer& layer, const Ptr& input) {
  if (layer.kind == "linear") {
    const Ptr& weight = layer.parameters.at("weight");
    const Ptr& bias = layer.parameters.at("bias");
    return add(matmul(input, weight), bias);
  }

  if (layer.kind == "lora" || layer.kind == "qlora") {
    const Ptr& a = layer.parameters.at("lora_a");
    const Ptr& b = layer.parameters.at("lora_b");
    const Ptr& bias = layer.parameters.at("bias");
    const Ptr& scaling = layer.buffers.at("scaling");

    Ptr base;
    if (layer.quantizedWeight) {
      // The base weight stays quantised: it is dequantised a row at a
      // time and never held whole in memory.
      base = std::make_shared<Tensor>(tensor::matmulQuantized(*input, *layer.quantizedWeight));
    } else {
      base = matmul(input, layer.parameters.at("weight"));
    }

    // The low-rank update: scaling * ((x @ a) @ b).
    const Ptr hidden = matmul(input, a);
    const Ptr update = mul(matmul(hidden, b), scaling);

    return add(add(base, update), bias);
  }

  if (layer.kind == "layer_norm") {
    const Ptr& scale = layer.parameters.at("scale");
    const Ptr& shift = layer.parameters.at("shift");
    const double epsilon = layer.buffers.at("epsilon")->getFlat(0);

    // Normalise over the last axis.
    const std::int64_t lastAxis = static_cast<std::int64_t>(input->shape().rank()) - 1;
    const auto shape = input->shape();
    const std::int64_t columns = shape.lastDimSize();

    auto mean = std::make_shared<Tensor>(
        tensor::sumOverAxis(*input, lastAxis));
    mean = std::make_shared<Tensor>(tensor::binaryOp(
        BinaryOp::Div, *mean,
        *std::make_shared<Tensor>(Tensor::full(DType::F32, mean->shape(),
                                               static_cast<double>(columns)))));
    const Ptr meanExpanded = broadcastTo(collapseLastAxis(mean), shape);

    const Ptr centred = sub(input, meanExpanded);
    auto variance = std::make_shared<Tensor>(
        tensor::sumOverAxis(*mul(centred, centred), lastAxis));
    variance = std::make_shared<Tensor>(tensor::binaryOp(
        BinaryOp::Div, *variance,
        *std::make_shared<Tensor>(Tensor::full(DType::F32, variance->shape(),
                                               static_cast<double>(columns)))));
    const Ptr varianceExpanded = broadcastTo(collapseLastAxis(variance), shape);

    auto normaliser = std::make_shared<Tensor>(
        tensor::binaryOp(BinaryOp::Add, *varianceExpanded,
                         *std::make_shared<Tensor>(Tensor::full(DType::F32, shape, epsilon))));
    const Ptr normalised = div(centred, sqrt(normaliser));

    return add(mul(normalised, scale), shift);
  }

  throw std::invalid_argument("I do not know how to run a '" + layer.kind + "' layer");
}

// ---------------------------------------------------------------------------
// Optimisers
// ---------------------------------------------------------------------------

namespace {

std::vector<Ptr> onlyTrainable(const std::vector<Ptr>& parameters) {
  std::vector<Ptr> trainable;
  for (const Ptr& parameter : parameters) {
    if (parameter && parameter->requiresGrad()) trainable.push_back(parameter);
  }
  return trainable;
}

}  // namespace

Optimizer sgd(const std::vector<Ptr>& parameters, double learning_rate) {
  Optimizer optimizer;
  optimizer.kind = "sgd";
  optimizer.parameters = onlyTrainable(parameters);
  optimizer.learning_rate = learning_rate;
  return optimizer;
}

Optimizer adam(const std::vector<Ptr>& parameters, double learning_rate) {
  Optimizer optimizer;
  optimizer.kind = "adam";
  optimizer.parameters = onlyTrainable(parameters);
  optimizer.learning_rate = learning_rate;
  for (const Ptr& parameter : optimizer.parameters) {
    optimizer.state->first_moment.push_back(
        std::make_shared<Tensor>(Tensor::zeros(parameter->dtype(), parameter->shape())));
    optimizer.state->second_moment.push_back(
        std::make_shared<Tensor>(Tensor::zeros(parameter->dtype(), parameter->shape())));
  }
  return optimizer;
}

Optimizer adamw(const std::vector<Ptr>& parameters, double learning_rate, double weight_decay) {
  Optimizer optimizer = adam(parameters, learning_rate);
  optimizer.kind = "adamw";
  optimizer.weight_decay = weight_decay;
  return optimizer;
}

void zeroGrad(Optimizer& optimizer) {
  for (const Ptr& parameter : optimizer.parameters) {
    if (parameter->grad()) parameter->zeroGrad();
  }
}

void step(Optimizer& optimizer) {
  ++optimizer.state->step;

  for (std::size_t i = 0; i < optimizer.parameters.size(); ++i) {
    const Ptr& parameter = optimizer.parameters[i];
    const Ptr& gradient = parameter->grad();
    if (!gradient) continue;

    if (optimizer.kind == "sgd") {
      const Ptr update = std::make_shared<Tensor>(tensor::binaryOp(
          BinaryOp::Mul, *gradient,
          *std::make_shared<Tensor>(
              Tensor::full(gradient->dtype(), gradient->shape(), optimizer.learning_rate))));
      // The parameter is updated in place, so that everyone who holds a
      // reference to it sees the change.
      parameter->assignValues(tensor::binaryOp(BinaryOp::Sub, *parameter, *update));
      continue;
    }

    // Adam and AdamW.
    const double beta1 = optimizer.beta1;
    const double beta2 = optimizer.beta2;
    const double biasCorrection1 = 1.0 - std::pow(beta1, static_cast<double>(optimizer.state->step));
    const double biasCorrection2 = 1.0 - std::pow(beta2, static_cast<double>(optimizer.state->step));

    Ptr& first = optimizer.state->first_moment[i];
    Ptr& second = optimizer.state->second_moment[i];

    // m = beta1 * m + (1 - beta1) * g
    const Ptr beta1Term = std::make_shared<Tensor>(
        Tensor::full(gradient->dtype(), gradient->shape(), beta1));
    const Ptr oneMinusBeta1 = std::make_shared<Tensor>(
        Tensor::full(gradient->dtype(), gradient->shape(), 1.0 - beta1));
    const Ptr firstUpdate = std::make_shared<Tensor>(
        tensor::binaryOp(BinaryOp::Mul, *gradient, *oneMinusBeta1));
    first->assignValues(tensor::binaryOp(
        BinaryOp::Add, tensor::binaryOp(BinaryOp::Mul, *first, *beta1Term), *firstUpdate));

    // v = beta2 * v + (1 - beta2) * g^2
    const Ptr squared = std::make_shared<Tensor>(tensor::binaryOp(BinaryOp::Mul, *gradient, *gradient));
    const Ptr beta2Term = std::make_shared<Tensor>(
        Tensor::full(gradient->dtype(), gradient->shape(), beta2));
    const Ptr oneMinusBeta2 = std::make_shared<Tensor>(
        Tensor::full(gradient->dtype(), gradient->shape(), 1.0 - beta2));
    const Ptr secondUpdate = std::make_shared<Tensor>(
        tensor::binaryOp(BinaryOp::Mul, *squared, *oneMinusBeta2));
    second->assignValues(tensor::binaryOp(
        BinaryOp::Add, tensor::binaryOp(BinaryOp::Mul, *second, *beta2Term), *secondUpdate));

    // The bias-corrected moments.
    const Ptr firstCorrected = std::make_shared<Tensor>(tensor::binaryOp(
        BinaryOp::Div, *first,
        *std::make_shared<Tensor>(Tensor::full(gradient->dtype(), gradient->shape(),
                                               biasCorrection1))));
    const Ptr secondCorrected = std::make_shared<Tensor>(tensor::binaryOp(
        BinaryOp::Div, *second,
        *std::make_shared<Tensor>(Tensor::full(gradient->dtype(), gradient->shape(),
                                               biasCorrection2))));

    // The update: lr * m / (sqrt(v) + epsilon)
    const Ptr denominator = std::make_shared<Tensor>(tensor::binaryOp(
        BinaryOp::Add, *std::make_shared<Tensor>(tensor::unaryOp(UnaryOp::Sqrt, *secondCorrected)),
        *std::make_shared<Tensor>(
            Tensor::full(gradient->dtype(), gradient->shape(), optimizer.epsilon))));
    Ptr update = std::make_shared<Tensor>(tensor::binaryOp(
        BinaryOp::Div, *firstCorrected, *denominator));

    if (optimizer.kind == "adamw") {
      // The weight decay is separate from the gradient.
      const Ptr decay = std::make_shared<Tensor>(tensor::binaryOp(
          BinaryOp::Mul, *parameter,
          *std::make_shared<Tensor>(Tensor::full(gradient->dtype(), gradient->shape(),
                                                 optimizer.learning_rate * optimizer.weight_decay))));
      parameter->assignValues(tensor::binaryOp(BinaryOp::Sub, *parameter, *decay));
    }

    update = std::make_shared<Tensor>(tensor::binaryOp(
        BinaryOp::Mul, *update,
        *std::make_shared<Tensor>(
            Tensor::full(gradient->dtype(), gradient->shape(), optimizer.learning_rate))));

    parameter->assignValues(tensor::binaryOp(BinaryOp::Sub, *parameter, *update));
  }
}

}  // namespace milo::autograd
