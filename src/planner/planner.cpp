#include "planner/planner.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace milo::planner {
namespace {

/// The number of bytes in a f32.
constexpr std::int64_t kF32 = 4;
/// The number of bytes in a bf16.
constexpr std::int64_t kBf16 = 2;
/// The number of bytes in an int8.
constexpr std::int64_t kInt8 = 1;

/// How many elements of activation a transformer layer keeps for one token.
///
/// This is the sum of what a layer has to hold on to for the backward
/// pass: the two layer-normalisation outputs, the three attention inputs
/// (query, key, value), the attention output, the MLP's wide hidden
/// state, and the MLP's output. The wide hidden state is four times the
/// hidden size, which is the usual ratio.
constexpr double kActivationsPerTokenPerLayer = 11.0;

/// How many matrices a transformer layer adapts with LoRA.
///
/// The attention projections (query, key, value, output) and the MLP's two
/// matrices.
constexpr std::int64_t kLoraMatricesPerLayer = 7;

/// The number of values in the NF4 codebook.
constexpr std::int64_t kNf4BlockSize = 64;

std::int64_t safeMultiply(std::int64_t a, std::int64_t b) {
  if (a == 0 || b == 0) return 0;
  const double result = static_cast<double>(a) * static_cast<double>(b);
  if (result > 9.0e18) return 9'000'000'000'000'000'000LL;
  return a * b;
}

}  // namespace

Quantization quantizationFromName(const std::string& name) {
  if (name == "int8" || name == "i8") return Quantization::Int8;
  if (name == "nf4" || name == "4bit") return Quantization::Nf4;
  return Quantization::None;
}

const char* quantizationName(Quantization quantization) {
  switch (quantization) {
    case Quantization::Int8:
      return "int8";
    case Quantization::Nf4:
      return "nf4";
    case Quantization::None:
      break;
  }
  return "none";
}

std::int64_t Estimate::total() const {
  return weights + gradients + optimizer + activations + attention + scratch;
}

std::int64_t trainableParameters(const Config& config) {
  if (!config.lora) return config.parameters;

  // Each adapted matrix of [hidden, hidden] gets a pair of [rank, hidden]
  // matrices.
  const std::int64_t perMatrix = safeMultiply(config.loraRank, 2 * config.hidden);
  return safeMultiply(config.layers, kLoraMatricesPerLayer * perMatrix);
}

Estimate estimate(const Config& config) {
  Estimate result;

  // --- the weights -------------------------------------------------------
  switch (config.quantization) {
    case Quantization::None:
      // The model is kept in bf16.
      result.weights = safeMultiply(config.parameters, kBf16);
      break;
    case Quantization::Int8:
      // One byte per weight, plus a scale for each output channel.
      result.weights = safeMultiply(config.parameters, kInt8);
      result.weights += safeMultiply(config.parameters, 4) / 1024;
      break;
    case Quantization::Nf4:
      // Half a byte per weight, plus a scale and an absolute maximum for
      // each block of 64.
      result.weights = config.parameters / 2;
      result.weights += safeMultiply(config.parameters, 8) / kNf4BlockSize;
      break;
  }

  // --- the gradients and the optimiser ------------------------------------
  const std::int64_t trainable = trainableParameters(config);
  result.gradients = safeMultiply(trainable, kF32);
  // Adam keeps a first and a second moment, both in f32.
  result.optimizer = safeMultiply(trainable, 2 * kF32);
  if (config.offloadOptimizer) {
    // The optimiser's state lives in the host's memory, not the device's.
    result.optimizer = 0;
  }

  // --- the activations ----------------------------------------------------
  const std::int64_t tokens = safeMultiply(config.batch, config.sequence);
  if (config.gradientCheckpointing) {
    // Only the boundaries between the layers are kept, and one layer is
    // recomputed at a time.
    result.activations = safeMultiply(tokens, config.layers * config.hidden);
    result.activations += safeMultiply(tokens, 11 * config.hidden);
  } else {
    result.activations = static_cast<std::int64_t>(
        std::ceil(static_cast<double>(tokens) * config.layers * kActivationsPerTokenPerLayer *
                  static_cast<double>(config.hidden)));
  }
  result.activations *= kBf16;

  // --- the attention scores -----------------------------------------------
  // The scores are a batch by heads by sequence by sequence tensor, for
  // every layer. This is the term that grows quadratically with the
  // sequence length.
  result.attention = safeMultiply(config.batch, config.heads);
  result.attention = safeMultiply(result.attention, config.sequence);
  result.attention = safeMultiply(result.attention, config.sequence);
  result.attention = safeMultiply(result.attention, config.layers);
  result.attention *= kBf16;

  // --- the scratch space --------------------------------------------------
  // Enough for the largest single operation: a couple of activation
  // tensors.
  result.scratch = safeMultiply(tokens, config.hidden * 2);
  result.scratch *= kBf16;

  return result;
}

Plan plan(const Config& config) {
  Plan result;
  result.estimate = estimate(config);

  if (config.deviceMemory <= 0) {
    result.fits = true;
    result.summary = "the device size is not known, so this is an estimate only";
    return result;
  }

  if (result.estimate.total() <= config.deviceMemory) {
    result.fits = true;
    std::ostringstream out;
    out << "it fits: " << result.estimate.total() << " bytes of " << config.deviceMemory;
    result.summary = out.str();
    return result;
  }

  result.fits = false;

  // Work through the changes in the order that costs the least quality.
  // Each one is applied to the candidate, and the search stops as soon as
  // the candidate fits.
  Config candidate = config;
  bool done = false;

  const auto consider = [&](const std::string& knob, const std::string& from,
                            const std::string& to, Config updated) {
    candidate = updated;
    result.suggestions.push_back(Suggestion{knob, from, to, ""});
    if (estimate(candidate).total() <= config.deviceMemory) done = true;
  };

  // 1. The batch size. Halving it is the cheapest change, and the
  //    gradient accumulation makes up for it.
  if (!done && candidate.batch > 1) {
    const std::int64_t halved = std::max<std::int64_t>(candidate.batch / 2, 1);
    Config updated = candidate;
    updated.batch = halved;
    updated.gradientAccumulation =
        candidate.gradientAccumulation * (candidate.batch / halved);
    consider("batch_size", std::to_string(candidate.batch), std::to_string(halved), updated);
  }

  // 2. Gradient checkpointing. It costs about a third of the speed.
  if (!done && !candidate.gradientCheckpointing) {
    Config updated = candidate;
    updated.gradientCheckpointing = true;
    consider("gradient_checkpointing", "off", "on", updated);
  }

  // 3. The sequence length. Truncating loses the tail of long examples.
  if (!done && candidate.sequence > 512) {
    Config updated = candidate;
    updated.sequence = candidate.sequence / 2;
    consider("sequence_length", std::to_string(candidate.sequence),
             std::to_string(updated.sequence), updated);
  }

  // 4. Quantization. NF4 is the smallest; int8 comes first because it
  //    costs less accuracy.
  if (!done && candidate.quantization == Quantization::None) {
    Config updated = candidate;
    updated.quantization = Quantization::Int8;
    consider("quantization", "none", "int8", updated);
  }
  if (!done && candidate.quantization != Quantization::Nf4) {
    Config updated = candidate;
    updated.quantization = Quantization::Nf4;
    consider("quantization", quantizationName(candidate.quantization), "nf4", updated);
  }

  // 5. The optimiser's state on the host. It costs speed, not quality.
  if (!done && !candidate.offloadOptimizer) {
    Config updated = candidate;
    updated.offloadOptimizer = true;
    consider("offload_optimizer", "on device", "on host", updated);
  }

  // Say why each change is worth making.
  for (Suggestion& suggestion : result.suggestions) {
    if (suggestion.knob == "batch_size") {
      suggestion.reason = "smaller batches use less memory; raise the gradient accumulation to keep "
                          "the effective batch the same";
    } else if (suggestion.knob == "gradient_checkpointing") {
      suggestion.reason = "the activations are recomputed instead of kept, which costs about a third "
                          "of the speed";
    } else if (suggestion.knob == "sequence_length") {
      suggestion.reason = "longer examples are truncated, which loses the end of the long ones";
    } else if (suggestion.knob == "quantization") {
      suggestion.reason = "the weights are stored more compactly, which costs a little accuracy";
    } else if (suggestion.knob == "offload_optimizer") {
      suggestion.reason = "the optimiser's state lives in the host's memory, which makes each step "
                          "slower";
    }
  }

  std::ostringstream out;
  out << "it needs " << estimate(candidate).total() << " bytes of " << config.deviceMemory;
  result.summary = out.str();
  return result;
}

Config applySuggestions(const Config& config, const Plan& plan) {
  Config result = config;
  for (const Suggestion& suggestion : plan.suggestions) {
    if (suggestion.knob == "batch_size") {
      const std::int64_t smaller = std::stoll(suggestion.to);
      if (smaller > 0 && result.batch > smaller) {
        // Keep the effective batch the same by accumulating more.
        result.gradientAccumulation *= result.batch / smaller;
        result.batch = smaller;
      }
    } else if (suggestion.knob == "gradient_checkpointing") {
      result.gradientCheckpointing = suggestion.to == "on";
    } else if (suggestion.knob == "sequence_length") {
      result.sequence = std::stoll(suggestion.to);
    } else if (suggestion.knob == "quantization") {
      result.quantization = quantizationFromName(suggestion.to);
    } else if (suggestion.knob == "offload_optimizer") {
      result.offloadOptimizer = suggestion.to == "on host";
    }
  }
  return result;
}

}  // namespace milo::planner
