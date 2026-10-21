#include "core/directives.hpp"

#include <string>

#include "test_framework.hpp"

using namespace milo;
using namespace milo::core;

TEST(directivesKnowTheirVerbs) {
  CHECK(findVerb("load") != nullptr);
  CHECK(findVerb("fine-tune") != nullptr);
  CHECK(findVerb("split") != nullptr);
  CHECK(findVerb("evaluate") != nullptr);
  CHECK(findVerb("predict") != nullptr);
  CHECK(findVerb("save") != nullptr);
  CHECK(findVerb("show") != nullptr);
  CHECK(findVerb("use") != nullptr);
  CHECK(findVerb("quantize") != nullptr);
  CHECK(findVerb("train") != nullptr);
  CHECK(findVerb("nonsense") == nullptr);
}

TEST(directivesMatchVerbsCaseInsensitively) {
  CHECK(findVerb("LOAD") != nullptr);
  CHECK(findVerb("Fine-Tune") != nullptr);
}

TEST(directivesHaveTheExpectedSlots) {
  const VerbSpec* load = findVerb("load");
  CHECK(load != nullptr);
  CHECK_EQ(load->alternatives.size(), static_cast<std::size_t>(1));
  const std::vector<TemplatePart>& parts = load->alternatives.front();
  CHECK_EQ(parts.size(), static_cast<std::size_t>(4));
  CHECK(parts[0].slot == SlotKind::Kind);
  CHECK_EQ(parts[0].slot_name, std::string("kind"));
  CHECK(parts[1].slot == SlotKind::Path);
  CHECK_EQ(parts[2].literal_text, std::string("as"));
  CHECK(parts[3].slot == SlotKind::Identifier);
  CHECK_EQ(parts[3].slot_name, std::string("name"));
}

TEST(directivesOfferTheLongestOptionKeysFirst) {
  // `batch size` must win over any shorter key, and `save to` over `save`.
  const std::vector<OptionKey>& keys = optionKeys();
  bool sawBatchSize = false;
  for (const OptionKey& key : keys) {
    if (key.canonical == "batch_size") {
      CHECK_EQ(key.words.size(), static_cast<std::size_t>(2));
      sawBatchSize = true;
    }
  }
  CHECK(sawBatchSize);

  for (std::size_t i = 1; i < keys.size(); ++i) {
    CHECK(keys[i - 1].words.size() >= keys[i].words.size());
  }
}
