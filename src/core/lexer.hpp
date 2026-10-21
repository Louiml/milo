// The lexer: bytes -> tokens, with indentation handling.
#ifndef MILO_CORE_LEXER_HPP
#define MILO_CORE_LEXER_HPP

#include <string>
#include <vector>

#include "core/diagnostic.hpp"
#include "core/token.hpp"

namespace milo {

/// Turns Milo source text into a token stream.
///
/// The lexer is responsible for three things the parser should not have to
/// think about:
///
///   * **Indentation.** Blocks are converted into explicit `Indent` and
///     `Dedent` tokens, so the parser never inspects whitespace.
///   * **Line structure.** A `Newline` token ends a logical line. Newlines
///     inside brackets are suppressed, and blank or comment-only lines do
///     not produce a `Newline` at all.
///   * **Contextual words.** Hard keywords are recognised here; contextual
///     words stay as identifiers and are resolved by the parser.
class Lexer {
 public:
  Lexer(std::string source, std::string filename);

  /// Tokenize the whole input.
  ///
  /// Always returns a token vector ending in `EndOfFile`. Errors are
  /// appended to `diagnostics()` rather than thrown.
  std::vector<Token> tokenize();

  const DiagnosticBag& diagnostics() const { return diagnostics_; }
  DiagnosticBag& diagnostics() { return diagnostics_; }

  const std::string& source() const { return source_; }
  const std::string& filename() const { return filename_; }

  /// Is `word` a hard keyword? Keywords are matched case-insensitively.
  static bool isKeyword(const std::string& word);

  /// The canonical (lower-case) spelling of a hard keyword, or `word`
  /// itself if it is not a keyword.
  static std::string canonicalKeyword(const std::string& word);

  /// All hard keywords, sorted, for "did you mean" suggestions.
  static const std::vector<std::string>& keywords();

  /// All contextual keywords, sorted. They are ordinary identifiers
  /// everywhere except where the grammar expects them.
  static const std::vector<std::string>& contextualKeywords();

 private:
  char peek(std::size_t offset = 0) const;
  bool atEnd() const { return position_ >= source_.size(); }
  char advance();
  bool match(char expected);
  SourceSpan makeSpan(std::size_t start_offset, std::size_t start_line, int start_column) const;
  void pushToken(Token token);

  /// Lex a single logical line's worth of tokens.
  void lexNumber();
  void lexString(char quote);
  void lexIdentifier();
  void lexOperator();

  std::string source_;
  std::string filename_;
  std::size_t position_ = 0;
  int line_ = 1;
  int column_ = 1;

  /// Stack of indentation widths, in columns. The bottom entry is 0.
  std::vector<int> indents_{0};
  /// Bracket nesting depth. Newlines are suppressed while > 0.
  int bracket_depth_ = 0;

  std::vector<Token> tokens_;
  DiagnosticBag diagnostics_;
};

}  // namespace milo

#endif  // MILO_CORE_LEXER_HPP
