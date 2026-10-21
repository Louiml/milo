// Tokens produced by the lexer.
#ifndef MILO_CORE_TOKEN_HPP
#define MILO_CORE_TOKEN_HPP

#include <cstdint>
#include <string>

#include "core/span.hpp"

namespace milo {

enum class TokenKind {
  EndOfFile,
  Newline,
  Indent,
  Dedent,

  Identifier,
  Integer,
  Float,
  String,

  // Punctuation and operators.
  Plus,
  Minus,
  Star,
  Slash,
  SlashSlash,
  Percent,
  StarStar,
  Equal,
  EqualEqual,
  BangEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  Bang,
  Arrow,
  LeftParen,
  RightParen,
  LeftBracket,
  RightBracket,
  LeftBrace,
  RightBrace,
  Comma,
  Colon,
  Dot,
  Semicolon,
  Question,

  Unknown,
};

const char* to_string(TokenKind kind);

/// A single lexical token.
struct Token {
  TokenKind kind = TokenKind::Unknown;
  SourceSpan span;

  /// The raw text of the token. For identifiers and keywords this is the
  /// identifier itself; for strings it is the *decoded* value; for numbers
  /// it is the literal text.
  std::string text;

  /// Numeric payloads.
  std::int64_t int_value = 0;
  double float_value = 0.0;

  Token() = default;
  Token(TokenKind k, SourceSpan s, std::string t = {})
      : kind(k), span(s), text(std::move(t)) {}

  /// A short human description, used in error messages.
  std::string describe() const;
};

}  // namespace milo

#endif  // MILO_CORE_TOKEN_HPP
