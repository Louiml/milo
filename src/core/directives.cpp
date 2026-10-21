#include "core/directives.hpp"

#include <algorithm>

namespace milo::core {
namespace {

TemplatePart literal(std::string text) {
  TemplatePart part;
  part.literal = true;
  part.literal_text = std::move(text);
  return part;
}

TemplatePart slot(SlotKind kind, std::string name,
                  std::vector<std::string> allowed = {}) {
  TemplatePart part;
  part.literal = false;
  part.slot = kind;
  part.slot_name = std::move(name);
  part.allowed = std::move(allowed);
  return part;
}

/// `MODEL on DATASET` — the shape shared by fine-tune, train, evaluate
/// and predict.
std::vector<TemplatePart> onDatasetTemplate() {
  return {
      slot(SlotKind::Identifier, "model"),
      literal("on"),
      slot(SlotKind::Identifier, "dataset"),
  };
}

std::vector<VerbSpec> buildVerbSpecs() {
  std::vector<VerbSpec> specs;

  {
    VerbSpec spec;
    spec.verb = "load";
    spec.alternatives.push_back({
        slot(SlotKind::Kind, "kind",
             {"model", "data", "dataset", "weights", "tokenizer", "adapter"}),
        slot(SlotKind::Path, "path"),
        literal("as"),
        slot(SlotKind::Identifier, "name"),
    });
    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "split";
    std::vector<TemplatePart> base = {
        slot(SlotKind::Identifier, "source"),
        literal("into"),
        slot(SlotKind::Identifier, "left"),
        literal("and"),
        slot(SlotKind::Identifier, "right"),
    };
    spec.alternatives.push_back(base);

    std::vector<TemplatePart> withRatio = base;
    withRatio.push_back(literal("by"));
    withRatio.push_back(slot(SlotKind::Number, "ratio"));
    withRatio.push_back(literal("percent"));
    spec.alternatives.push_back(std::move(withRatio));

    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "fine-tune";
    spec.alternatives.push_back(onDatasetTemplate());
    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "train";
    spec.alternatives.push_back(onDatasetTemplate());
    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "evaluate";
    spec.alternatives.push_back(onDatasetTemplate());
    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "predict";
    spec.alternatives.push_back(onDatasetTemplate());
    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "save";
    spec.alternatives.push_back({
        slot(SlotKind::Identifier, "model"),
        literal("to"),
        slot(SlotKind::Path, "path"),
    });
    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "show";
    spec.alternatives.push_back({slot(SlotKind::Identifier, "metric")});
    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "use";
    spec.alternatives.push_back({
        literal("python"),
        slot(SlotKind::Value, "module"),
        literal("as"),
        slot(SlotKind::Identifier, "name"),
    });
    specs.push_back(std::move(spec));
  }

  {
    VerbSpec spec;
    spec.verb = "quantize";
    spec.alternatives.push_back({
        slot(SlotKind::Identifier, "model"),
        literal("to"),
        slot(SlotKind::Identifier, "scheme"),
    });
    specs.push_back(std::move(spec));
  }

  return specs;
}

std::vector<OptionKey> buildOptionKeys() {
  std::vector<OptionKey> keys = {
      {"method", {"using"}},
      {"method", {"method"}},
      {"rank", {"rank"}},
      {"alpha", {"alpha"}},
      {"dropout", {"dropout"}},
      {"batch_size", {"batch", "size"}},
      {"epochs", {"epochs"}},
      {"learning_rate", {"learning", "rate"}},
      {"optimizer", {"optimizer"}},
      {"precision", {"precision"}},
      {"quantization", {"quantization"}},
      {"gradient_accumulation", {"gradient", "accumulation"}},
      {"max_length", {"max", "length"}},
      {"warmup_steps", {"warmup", "steps"}},
      {"seed", {"seed"}},
      {"device", {"device"}},
      {"shuffle", {"shuffle"}},
      {"save", {"save", "to"}},
      {"save", {"save"}},
  };
  // Longest key first so that `batch size` wins over any shorter key.
  std::stable_sort(keys.begin(), keys.end(),
                   [](const OptionKey& a, const OptionKey& b) {
                     return a.words.size() > b.words.size();
                   });
  return keys;
}

}  // namespace

const std::vector<VerbSpec>& verbSpecs() {
  static const std::vector<VerbSpec> specs = buildVerbSpecs();
  return specs;
}

const VerbSpec* findVerb(const std::string& word) {
  std::string lower = word;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  for (const VerbSpec& spec : verbSpecs()) {
    if (spec.verb == lower) return &spec;
  }
  return nullptr;
}

const std::vector<OptionKey>& optionKeys() {
  static const std::vector<OptionKey> keys = buildOptionKeys();
  return keys;
}

const std::vector<std::string>& optionSeparators() {
  static const std::vector<std::string> separators = {"with", "via", "at"};
  return separators;
}

}  // namespace milo::core
