#include "cli/plan.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#include "planner/planner.hpp"
#include "safetensors/reader.hpp"

namespace milo::cli {
namespace {

/// Format a byte count the way a person would say it.
std::string humanBytes(std::int64_t bytes) {
  static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < std::size(kUnits)) {
    value /= 1024.0;
    ++unit;
  }
  std::ostringstream out;
  if (unit == 0) {
    out << bytes << ' ' << kUnits[unit];
  } else {
    out << std::fixed << std::setprecision(value < 10.0 ? 2 : 1) << value << ' ' << kUnits[unit];
  }
  return out.str();
}

/// What the model file tells us about the model's shape.
struct ModelShape {
  std::int64_t parameters = 0;
  std::int64_t hidden = 0;
  std::int64_t layers = 0;
  std::int64_t heads = 0;
};

/// Guess the model's shape from its tensor names and shapes.
///
/// This is a heuristic, and the report says so: the hidden size is the
/// most common second dimension of the two-dimensional weights, and the
/// number of layers is the number of distinct layer numbers in the names.
ModelShape guessShape(const safetensors::File& file) {
  ModelShape shape;
  std::map<std::int64_t, int> dimensions;
  std::vector<std::string> layerNumbers;

  for (const safetensors::TensorEntry& entry : file.header().tensors) {
    shape.parameters += static_cast<std::int64_t>(entry.numel());
    if (entry.shape.size() == 2) {
      dimensions[entry.shape[1]] += 1;
    }

    // Look for "layers.12" in the name.
    const std::size_t at = entry.name.find("layers.");
    if (at != std::string::npos) {
      std::size_t start = at + 7;
      std::size_t end = start;
      while (end < entry.name.size() && std::isdigit(static_cast<unsigned char>(entry.name[end]))) {
        ++end;
      }
      if (end > start) {
        layerNumbers.push_back(entry.name.substr(start, end - start));
      }
    }
  }

  int bestCount = 0;
  for (const auto& entry : dimensions) {
    if (entry.second > bestCount) {
      bestCount = entry.second;
      shape.hidden = entry.first;
    }
  }

  std::sort(layerNumbers.begin(), layerNumbers.end());
  layerNumbers.erase(std::unique(layerNumbers.begin(), layerNumbers.end()), layerNumbers.end());
  shape.layers = static_cast<std::int64_t>(layerNumbers.size());

  // A common head size is 64.
  if (shape.hidden > 0 && shape.hidden % 64 == 0) {
    shape.heads = shape.hidden / 64;
  }
  return shape;
}

/// Find an option's value.
std::string option(const std::vector<std::string>& options, const std::string& name,
                   const std::string& fallback) {
  for (std::size_t i = 0; i + 1 < options.size(); ++i) {
    if (options[i] == name) return options[i + 1];
  }
  return fallback;
}

bool hasOption(const std::vector<std::string>& options, const std::string& name) {
  return std::find(options.begin(), options.end(), name) != options.end();
}

}  // namespace

int planModel(const std::string& path, const std::vector<std::string>& options, std::ostream& out,
              std::ostream& err) {
  std::shared_ptr<safetensors::File> file;
  try {
    file = std::make_shared<safetensors::File>(safetensors::File::open(path));
  } catch (const safetensors::Error& error) {
    err << "error: " << error.what() << '\n';
    return 1;
  }

  const ModelShape shape = guessShape(*file);

  planner::Config config;
  config.parameters = shape.parameters;
  config.hidden = shape.hidden;
  config.layers = shape.layers;
  config.heads = shape.heads;

  // The guesses can be overridden, because the planner's answer depends on
  // them and a file does not always say what the model looks like.
  if (hasOption(options, "--hidden")) {
    config.hidden = std::stoll(option(options, "--hidden", "0"));
  }
  if (hasOption(options, "--layers")) {
    config.layers = std::stoll(option(options, "--layers", "0"));
  }
  if (hasOption(options, "--heads")) {
    config.heads = std::stoll(option(options, "--heads", "0"));
  }

  if (hasOption(options, "--parameters")) {
    config.parameters = std::stoll(option(options, "--parameters", "0"));
  }
  config.batch = std::stoll(option(options, "--batch-size", "1"));
  config.sequence = std::stoll(option(options, "--sequence", "2048"));
  config.quantization =
      planner::quantizationFromName(option(options, "--quantization", "none"));
  config.lora = hasOption(options, "--lora");
  config.loraRank = std::stoll(option(options, "--lora-rank", "16"));
  config.gradientCheckpointing = hasOption(options, "--gradient-checkpointing");
  config.gradientAccumulation = std::stoll(option(options, "--gradient-accumulation", "1"));
  config.offloadOptimizer = hasOption(options, "--offload-optimizer");
  config.device = option(options, "--device", "cpu");
  config.deviceMemory = std::stoll(option(options, "--memory", "0")) * 1024 * 1024 * 1024;

  const planner::Plan plan = planner::plan(config);
  const planner::Estimate& estimate = plan.estimate;

  out << path << '\n';
  out << "  parameters        " << config.parameters;
  if (config.parameters != shape.parameters) {
    out << " (overridden)";
  }
  out << '\n';
  if (config.hidden > 0) {
    out << "  hidden size       " << config.hidden
        << (config.hidden == shape.hidden ? " (guessed from the tensor shapes)" : " (overridden)")
        << '\n';
    out << "  layers            " << config.layers
        << (config.layers == shape.layers ? " (guessed from the tensor names)" : " (overridden)")
        << '\n';
    out << "  attention heads   " << config.heads
        << (config.heads == shape.heads ? " (guessed)" : " (overridden)") << '\n';
  }
  out << "  batch size        " << config.batch << '\n';
  out << "  sequence length   " << config.sequence << '\n';
  out << "  quantization      " << planner::quantizationName(config.quantization) << '\n';
  if (config.lora) {
    out << "  lora              rank " << config.loraRank << '\n';
  }
  out << "  trainable         " << planner::trainableParameters(config) << " of " << config.parameters
      << " parameters\n";
  out << "  device            " << config.device;
  if (config.deviceMemory > 0) {
    out << " (" << humanBytes(config.deviceMemory) << ")";
  }
  out << '\n';
  out << '\n';

  out << "  weights           " << std::setw(10) << humanBytes(estimate.weights) << '\n';
  out << "  gradients         " << std::setw(10) << humanBytes(estimate.gradients) << '\n';
  out << "  optimizer state   " << std::setw(10) << humanBytes(estimate.optimizer) << '\n';
  out << "  activations       " << std::setw(10) << humanBytes(estimate.activations) << '\n';
  out << "  attention scores  " << std::setw(10) << humanBytes(estimate.attention) << '\n';
  out << "  scratch           " << std::setw(10) << humanBytes(estimate.scratch) << '\n';
  out << "  --------------------------------------------\n";
  out << "  total             " << std::setw(10) << humanBytes(estimate.total()) << '\n';
  out << '\n';

  if (plan.fits) {
    out << "  it fits\n";
    if (config.deviceMemory <= 0) {
      out << "  (the device's memory is not known, so this is an estimate)\n";
    }
    out << '\n';
    return 0;
  }

  out << "  it does not fit: " << plan.summary << '\n';
  out << '\n';
  if (plan.suggestions.empty()) {
    out << "  Nothing in the plan's reach will make it fit. The model needs\n";
    out << "  more memory than the device has; a smaller model, or more memory,\n";
    out << "  is the only way.\n";
    out << '\n';
    return 0;
  }

  out << "  What would make it fit, in the order to try it:\n";
  for (std::size_t i = 0; i < plan.suggestions.size(); ++i) {
    const planner::Suggestion& suggestion = plan.suggestions[i];
    out << "    " << (i + 1) << ". " << suggestion.knob << ": " << suggestion.from << " -> "
        << suggestion.to << '\n';
    out << "         " << suggestion.reason << '\n';
  }
  out << '\n';
  out << "  Nothing has been changed. Apply these yourself if you want them.\n";
  out << '\n';
  return 0;
}

}  // namespace milo::cli
