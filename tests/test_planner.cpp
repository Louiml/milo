#include "planner/planner.hpp"

#include <string>

#include "test_framework.hpp"

using namespace milo;
using namespace milo::planner;

namespace {

/// A 7B model, as it would be described.
Config sevenBillion() {
  Config config;
  config.parameters = 7'000'000'000;
  config.hidden = 4096;
  config.layers = 32;
  config.heads = 32;
  config.batch = 4;
  config.sequence = 2048;
  config.device = "cuda";
  config.deviceMemory = 12LL * 1024 * 1024 * 1024;
  return config;
}

}  // namespace

TEST(plannerCountsTheWeights) {
  Config config = sevenBillion();
  config.layers = 0;
  config.heads = 0;
  config.sequence = 0;
  config.batch = 0;
  const Estimate estimate = planner::estimate(config);
  // Seven billion weights in bf16.
  CHECK_EQ(estimate.weights, static_cast<std::int64_t>(14'000'000'000));
}

TEST(plannerCountsTheOptimizerState) {
  Config config = sevenBillion();
  config.layers = 0;
  config.heads = 0;
  config.sequence = 0;
  config.batch = 0;
  const Estimate estimate = planner::estimate(config);
  // Adam keeps two f32 moments per parameter.
  CHECK_EQ(estimate.optimizer, static_cast<std::int64_t>(56'000'000'000));
}

TEST(plannerKnowsThatLoraTrainsFarFewerParameters) {
  Config config = sevenBillion();
  const std::int64_t full = trainableParameters(config);
  config.lora = true;
  config.loraRank = 16;
  const std::int64_t withLora = trainableParameters(config);

  CHECK(withLora < full / 100);
  // Seven matrices per layer, each rank by two times hidden.
  CHECK_EQ(withLora, static_cast<std::int64_t>(32 * 7 * 16 * 2 * 4096));
}

TEST(plannerKnowsThatQuantizationShrinksTheWeights) {
  Config config = sevenBillion();
  config.layers = 0;
  config.heads = 0;
  config.sequence = 0;
  config.batch = 0;

  const std::int64_t plain = planner::estimate(config).weights;
  config.quantization = Quantization::Int8;
  const std::int64_t int8 = planner::estimate(config).weights;
  config.quantization = Quantization::Nf4;
  const std::int64_t nf4 = planner::estimate(config).weights;

  CHECK(int8 < plain);
  CHECK(nf4 < int8);
  CHECK(nf4 < plain / 3);
}

TEST(plannerSaysWhenItFits) {
  Config config = sevenBillion();
  config.parameters = 1'000'000;  // a tiny model
  config.layers = 0;
  config.heads = 0;
  config.sequence = 0;
  config.batch = 0;
  const Plan plan = planner::plan(config);
  CHECK(plan.fits);
  CHECK(plan.suggestions.empty());
}

TEST(plannerSaysWhenItDoesNotFit) {
  const Plan plan = planner::plan(sevenBillion());
  CHECK(!plan.fits);
  CHECK(!plan.suggestions.empty());
}

TEST(plannerSuggestsTheCheapestChangesFirst) {
  const Plan plan = planner::plan(sevenBillion());
  CHECK(!plan.suggestions.empty());
  // The batch size comes first: it costs the least quality.
  CHECK_EQ(plan.suggestions.front().knob, std::string("batch_size"));
  // Every suggestion says why it is worth doing.
  for (const Suggestion& suggestion : plan.suggestions) {
    CHECK(!suggestion.reason.empty());
  }
}

TEST(plannerReachesTheQuantizationSuggestion) {
  // A model so big that only quantization can save it.
  Config config = sevenBillion();
  config.batch = 1;
  config.sequence = 512;
  config.layers = 0;
  config.heads = 0;
  const Plan plan = planner::plan(config);

  bool suggestedQuantization = false;
  for (const Suggestion& suggestion : plan.suggestions) {
    if (suggestion.knob == "quantization") suggestedQuantization = true;
  }
  CHECK(suggestedQuantization);
}

TEST(plannerChangesNothingOnItsOwn) {
  const Config config = sevenBillion();
  const Plan plan = planner::plan(config);
  // The planner reports; it does not act. The caller asks for the changes
  // it wants.
  const Config updated = applySuggestions(config, plan);
  CHECK(updated.batch <= config.batch);
  CHECK(updated.deviceMemory == config.deviceMemory);
  CHECK(updated.parameters == config.parameters);
}

TEST(plannerKeepsTheEffectiveBatchWhenItShrinksIt) {
  const Config config = sevenBillion();
  const Plan plan = planner::plan(config);
  const Config updated = applySuggestions(config, plan);
  if (updated.batch < config.batch) {
    CHECK(updated.gradientAccumulation > config.gradientAccumulation);
    // The effective batch is the same.
    CHECK_EQ(updated.batch * updated.gradientAccumulation,
             config.batch * config.gradientAccumulation);
  }
}

TEST(plannerKnowsTheQuantizationNames) {
  CHECK(quantizationFromName("nf4") == Quantization::Nf4);
  CHECK(quantizationFromName("int8") == Quantization::Int8);
  CHECK(quantizationFromName("none") == Quantization::None);
  CHECK(quantizationFromName("nonsense") == Quantization::None);
  CHECK_EQ(std::string(quantizationName(Quantization::Nf4)), std::string("nf4"));
}
