// The sentence vocabulary.
//
// Every plain-English sentence in Milo is a *template match* against this
// table. Templates are data, not code, so the set of sentences is finite,
// closed, and statically known — which is what makes English-like syntax
// analysable rather than ambiguous.
#ifndef MILO_CORE_DIRECTIVES_HPP
#define MILO_CORE_DIRECTIVES_HPP

#include <string>
#include <vector>

namespace milo::core {

/// What a template slot accepts.
enum class SlotKind {
  /// A plain name.
  Identifier,
  /// A quoted path.
  Path,
  /// A whole number.
  Number,
  /// A name, a number, or a quoted string.
  Value,
  /// A name restricted to a fixed set.
  Kind,
};

/// One piece of a template.
struct TemplatePart {
  /// True when this part is a literal word that must appear verbatim.
  bool literal = false;
  /// The literal word (lower case).
  std::string literal_text;
  /// What the slot accepts.
  SlotKind slot = SlotKind::Identifier;
  /// The name the filled slot is stored under.
  std::string slot_name;
  /// For `Kind` slots: the accepted words.
  std::vector<std::string> allowed;
};

/// A sentence verb and the template(s) it may take.
struct VerbSpec {
  /// The verb, lower case. `fine-tune` is hyphenated.
  std::string verb;
  /// Alternative templates, tried in order.
  std::vector<std::vector<TemplatePart>> alternatives;
};

/// One option key.
struct OptionKey {
  /// The canonical name, used by the runtime.
  std::string canonical;
  /// The words the user writes, e.g. `["batch", "size"]`.
  std::vector<std::string> words;
};

/// All sentence verbs.
const std::vector<VerbSpec>& verbSpecs();

/// Look up a verb by its (case-insensitive) text. Returns nullptr when the
/// word does not start a sentence.
const VerbSpec* findVerb(const std::string& word);

/// All option keys, longest first so that matching is greedy.
const std::vector<OptionKey>& optionKeys();

/// Words that may precede an option key and are ignored, e.g. `with rank 16`.
const std::vector<std::string>& optionSeparators();

}  // namespace milo::core

#endif  // MILO_CORE_DIRECTIVES_HPP
