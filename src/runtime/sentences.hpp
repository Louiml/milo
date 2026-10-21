// Carrying out a sentence.
//
// A sentence is the same thing whichever engine runs it, so the work lives
// here rather than in the evaluator.
#ifndef MILO_RUNTIME_SENTENCES_HPP
#define MILO_RUNTIME_SENTENCES_HPP

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ast.hpp"
#include "runtime/environment.hpp"
#include "runtime/interp.hpp"
#include "runtime/value.hpp"

namespace milo::runtime {

/// The values a sentence needs: its slots and its options.
struct SentenceValues {
  /// The slots, by name: `kind`, `path`, `model`, `dataset`, ...
  std::vector<std::pair<std::string, Value>> slots;
  /// The options, by key: `method`, `rank`, `precision`, ...
  std::vector<std::pair<std::string, Value>> options;

  /// The value of a slot, or nullptr when the sentence has no such slot.
  const Value* slot(const std::string& name) const;

  /// The value of an option, or nullptr.
  const Value* option(const std::string& key) const;
};

/// Carry out one clause of a sentence.
///
/// Throws a `RuntimeError` when the sentence cannot be carried out.
void executeSentence(Interpreter& interpreter, const std::string& verb, SourceSpan span,
                     const SentenceValues& values, const std::shared_ptr<Environment>& env);

}  // namespace milo::runtime

#endif  // MILO_RUNTIME_SENTENCES_HPP
