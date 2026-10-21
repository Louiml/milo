// The memory planner.
//
// Before a training run starts, the planner answers two questions:
//
//   1. Will this fit?
//   2. If not, what should I change?
//
// It answers them by *reporting*, never by acting. The planner returns a
// plan — an estimate and an ordered list of suggestions — and the user
// decides what to do with it. A tool that silently changes your training
// run is a tool you cannot trust.
#ifndef MILO_PLANNER_PLANNER_HPP
#define MILO_PLANNER_PLANNER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace milo::planner {

/// How the weights are stored.
enum class Quantization {
  None,
  Int8,
  Nf4,
};

Quantization quantizationFromName(const std::string& name);
const char* quantizationName(Quantization quantization);

/// What the training run looks like.
struct Config {
  /// How many parameters the model has.
  std::int64_t parameters = 0;
  /// The size of the model's hidden state.
  std::int64_t hidden = 0;
  /// How many layers the model has.
  std::int64_t layers = 0;
  /// How many attention heads the model has.
  std::int64_t heads = 0;
  /// How many examples are trained at once.
  std::int64_t batch = 1;
  /// How long each example is.
  std::int64_t sequence = 2048;
  /// How the weights are stored.
  Quantization quantization = Quantization::None;
  /// Is LoRA being used?
  bool lora = false;
  /// The rank of the LoRA adapters.
  std::int64_t loraRank = 16;
  /// Are the activations recomputed during the backward pass?
  bool gradientCheckpointing = false;
  /// How many batches are accumulated before a step.
  std::int64_t gradientAccumulation = 1;
  /// Is the optimiser's state kept on the CPU?
  bool offloadOptimizer = false;
  /// How much memory the device has, in bytes. Zero means "unknown".
  std::int64_t deviceMemory = 0;
  /// The name of the device, for the report.
  std::string device = "cpu";
};

/// How much memory each part of the run needs.
struct Estimate {
  std::int64_t weights = 0;
  std::int64_t gradients = 0;
  std::int64_t optimizer = 0;
  std::int64_t activations = 0;
  std::int64_t attention = 0;
  std::int64_t scratch = 0;

  /// The total.
  std::int64_t total() const;
};

/// One thing the user could change.
struct Suggestion {
  /// The setting: `batch_size`, `gradient_checkpointing`, ...
  std::string knob;
  /// What it is now.
  std::string from;
  /// What it would become.
  std::string to;
  /// Why it is worth doing.
  std::string reason;
};

/// The answer.
struct Plan {
  /// What the run needs, as it stands.
  Estimate estimate;
  /// Does it fit? Always true when the device size is unknown.
  bool fits = true;
  /// The changes that would make it fit, in the order they should be tried.
  std::vector<Suggestion> suggestions;
  /// A sentence saying what would happen.
  std::string summary;
};

/// Work out what a run needs.
Estimate estimate(const Config& config);

/// How many parameters would actually be trained.
std::int64_t trainableParameters(const Config& config);

/// Work out what a run needs, and what could be changed.
Plan plan(const Config& config);

/// The configuration that the plan's suggestions describe.
///
/// This does *not* change the user's configuration: it returns a new one,
/// so the user can look at it before using it.
Config applySuggestions(const Config& config, const Plan& plan);

}  // namespace milo::planner

#endif  // MILO_PLANNER_PLANNER_HPP
