#include "core/lexer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <unordered_set>

namespace milo {
namespace {

const std::vector<std::string>& hardKeywords() {
  static const std::vector<std::string> keywords = {
      "and",   "as",    "be",     "break", "by",      "continue", "elif",
      "else",  "false", "fn",     "for",   "from",    "function", "if",
      "import", "in",   "is",     "let",   "not",     "null",     "of",
      "on",    "or",    "return", "set",   "then",    "to",       "true",
      "use",   "while", "with",
  };
  return keywords;
}

/// Words that are only meaningful in a particular grammatical position.
/// They stay identifiers so that programs may still use them as names.
const std::unordered_set<std::string>& contextualWords() {
  static const std::unordered_set<std::string> words = {
      // Sentence verbs and their slot words.
      "load", "data", "dataset", "model", "weights", "tokenizer", "adapter",
      "split", "into", "train", "evaluate", "predict", "save", "show",
      "python", "quantize", "percent", "via",
      // Option keys.
      "using", "method", "rank", "alpha", "dropout", "batch", "size",
      "epochs", "learning", "rate", "optimizer", "precision", "quantization",
      "gradient", "accumulation", "max", "length", "warmup", "steps",
      "seed", "device", "shuffle",
      // Type names.
      "tensor", "int", "float", "bool", "str", "bytes", "any", "list", "dict",
      "f32", "f16", "bf16", "f64", "i8", "i32", "i64", "u8",
  };
  return words;
}

/// Hyphenated words that the lexer folds into a single token.
const std::unordered_set<std::string>& hyphenatedWords() {
  static const std::unordered_set<std::string> words = {"fine-tune"};
  return words;
}

std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool isIdentStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool isIdentContinue(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

}  // namespace

bool Lexer::isKeyword(const std::string& word) {
  const std::string lower = toLower(word);
  return std::binary_search(hardKeywords().begin(), hardKeywords().end(), lower);
}

std::string Lexer::canonicalKeyword(const std::string& word) {
  const std::string lower = toLower(word);
  if (isKeyword(lower)) return lower;
  return word;
}

const std::vector<std::string>& Lexer::keywords() { return hardKeywords(); }

const std::vector<std::string>& Lexer::contextualKeywords() {
  static const std::vector<std::string> words = [] {
    std::vector<std::string> v(contextualWords().begin(), contextualWords().end());
    std::sort(v.begin(), v.end());
    return v;
  }();
  return words;
}

Lexer::Lexer(std::string source, std::string filename)
    : source_(std::move(source)), filename_(std::move(filename)) {
  // A UTF-8 byte-order mark is permitted and ignored.
  if (source_.size() >= 3 && static_cast<unsigned char>(source_[0]) == 0xEF &&
      static_cast<unsigned char>(source_[1]) == 0xBB &&
      static_cast<unsigned char>(source_[2]) == 0xBF) {
    source_.erase(0, 3);
  }
}

char Lexer::peek(std::size_t offset) const {
  const std::size_t index = position_ + offset;
  if (index >= source_.size()) return '\0';
  return source_[index];
}

char Lexer::advance() {
  const char c = source_[position_++];
  if (c == '\n') {
    ++line_;
    column_ = 1;
  } else {
    ++column_;
  }
  return c;
}

bool Lexer::match(char expected) {
  if (atEnd() || peek() != expected) return false;
  advance();
  return true;
}

SourceSpan Lexer::makeSpan(std::size_t start_offset, std::size_t start_line, int start_column) const {
  return SourceSpan{start_offset, position_ - start_offset, static_cast<int>(start_line),
                     start_column};
}

void Lexer::pushToken(Token token) { tokens_.push_back(std::move(token)); }

std::vector<Token> Lexer::tokenize() {
  while (true) {
    if (bracket_depth_ == 0 && column_ == 1) {
      // Measure the indentation of this line. Blank lines and comment-only
      // lines are skipped entirely: they must not open or close a block.
      int indent = 0;
      bool used_tab = false;
      bool used_space = false;
      bool has_content = false;

      while (!atEnd()) {
        const char c = peek();
        if (c == ' ') {
          ++indent;
          used_space = true;
          advance();
        } else if (c == '\t') {
          indent = (indent / 8 + 1) * 8;
          used_tab = true;
          advance();
        } else if (c == '\n' || c == '\r') {
          break;  // blank line
        } else if (c == '#') {
          while (!atEnd() && peek() != '\n') advance();
          break;  // comment-only line
        } else {
          has_content = true;
          break;
        }
      }

      if (used_tab && used_space) {
        diagnostics_.error(
            "this line mixes tabs and spaces for indentation; use one or the other",
            SourceSpan{position_, 0, line_, 1});
      }

      if (!has_content) {
        if (atEnd()) break;
        // Consume the line ending of the blank / comment line.
        if (peek() == '\r') advance();
        if (peek() == '\n') advance();
        continue;
      }

      // Apply the indentation change.
      if (indent > indents_.back()) {
        indents_.push_back(indent);
        pushToken(Token(TokenKind::Indent, SourceSpan{position_, 0, line_, 1}));
      } else if (indent < indents_.back()) {
        while (indents_.size() > 1 && indents_.back() > indent) {
          indents_.pop_back();
          pushToken(Token(TokenKind::Dedent, SourceSpan{position_, 0, line_, 1}));
        }
        if (indents_.back() != indent) {
          diagnostics_.error(
              "this line is indented oddly compared with the lines around it",
              SourceSpan{position_, 0, line_, 1});
          indents_.back() = indent;
        }
      }
    }

    if (atEnd()) break;

    // Skip intra-line whitespace.
    while (!atEnd() && (peek() == ' ' || peek() == '\t')) advance();
    if (atEnd()) break;

    const char c = peek();

    if (c == '\r' || c == '\n') {
      if (c == '\r') advance();
      if (peek() == '\n') advance();
      if (bracket_depth_ == 0) {
        pushToken(Token(TokenKind::Newline, SourceSpan{position_, 0, line_, 1}));
      }
      continue;
    }

    if (c == '#') {
      while (!atEnd() && peek() != '\n') advance();
      continue;
    }

    if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
      lexNumber();
      continue;
    }

    if (isIdentStart(c)) {
      lexIdentifier();
      continue;
    }

    if (c == '"' || c == '\'') {
      lexString(c);
      continue;
    }

    lexOperator();
  }

  // Close every open block.
  while (indents_.size() > 1) {
    indents_.pop_back();
    pushToken(Token(TokenKind::Dedent, SourceSpan{position_, 0, line_, column_}));
  }

  pushToken(Token(TokenKind::EndOfFile, SourceSpan{position_, 0, line_, column_}));
  return std::move(tokens_);
}

void Lexer::lexNumber() {
  const std::size_t start = position_;
  const int startLine = line_;
  const int startColumn = column_;

  if (peek() == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
    advance();
    advance();
    const std::size_t digitsStart = position_;
    while (!atEnd() && (std::isxdigit(static_cast<unsigned char>(peek())) != 0 || peek() == '_')) {
      advance();
    }
    if (position_ == digitsStart) {
      diagnostics_.error("this number is missing hexadecimal digits",
                         SourceSpan{start, position_ - start, startLine, startColumn});
    }
    std::string text = source_.substr(start, position_ - start);
    std::string cleaned;
    for (const char ch : text) {
      if (ch != '_') cleaned.push_back(ch);
    }
    const unsigned long long value = std::strtoull(cleaned.c_str() + 2, nullptr, 16);
    Token token(TokenKind::Integer, makeSpan(start, startLine, startColumn), text);
    token.int_value = static_cast<std::int64_t>(value);
    pushToken(std::move(token));
    return;
  }

  while (!atEnd() && (std::isdigit(static_cast<unsigned char>(peek())) != 0 || peek() == '_')) {
    advance();
  }

  bool isFloat = false;

  // A '.' only starts a fraction when a digit follows it. This is what lets
  // `1.` be the integer 1 followed by a dot, and keeps the optional
  // sentence terminator unambiguous.
  if (peek() == '.' && std::isdigit(static_cast<unsigned char>(peek(1))) != 0) {
    isFloat = true;
    advance();
    while (!atEnd() && (std::isdigit(static_cast<unsigned char>(peek())) != 0 || peek() == '_')) {
      advance();
    }
  }

  if (peek() == 'e' || peek() == 'E') {
    const std::size_t save = position_;
    const int saveLine = line_;
    const int saveColumn = column_;
    advance();
    if (peek() == '+' || peek() == '-') advance();
    if (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
      isFloat = true;
      while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())) != 0) advance();
    } else {
      // Not an exponent after all; put the 'e' back.
      position_ = save;
      line_ = saveLine;
      column_ = saveColumn;
    }
  }

  const std::string text = source_.substr(start, position_ - start);
  std::string cleaned;
  for (const char ch : text) {
    if (ch != '_') cleaned.push_back(ch);
  }

  if (isFloat) {
    Token token(TokenKind::Float, makeSpan(start, startLine, startColumn), text);
    token.float_value = std::strtod(cleaned.c_str(), nullptr);
    pushToken(std::move(token));
  } else {
    Token token(TokenKind::Integer, makeSpan(start, startLine, startColumn), text);
    token.int_value = static_cast<std::int64_t>(std::strtoll(cleaned.c_str(), nullptr, 10));
    pushToken(std::move(token));
  }
}

void Lexer::lexString(char quote) {
  const std::size_t start = position_;
  const int startLine = line_;
  const int startColumn = column_;

  advance();  // opening quote

  std::string value;
  bool terminated = false;

  while (!atEnd()) {
    const char c = advance();
    if (c == quote) {
      terminated = true;
      break;
    }
    if (c == '\n') {
      break;  // strings do not span lines
    }
    if (c != '\\') {
      value.push_back(c);
      continue;
    }
    if (atEnd()) break;
    const char escaped = advance();
    switch (escaped) {
      case 'n':
        value.push_back('\n');
        break;
      case 't':
        value.push_back('\t');
        break;
      case 'r':
        value.push_back('\r');
        break;
      case '0':
        value.push_back('\0');
        break;
      case '\\':
        value.push_back('\\');
        break;
      case '"':
        value.push_back('"');
        break;
      case '\'':
        value.push_back('\'');
        break;
      case 'x': {
        if (atEnd() || std::isxdigit(static_cast<unsigned char>(peek())) == 0) {
          diagnostics_.error("`\\x` needs two hexadecimal digits",
                             SourceSpan{start, position_ - start, startLine, startColumn});
          break;
        }
        const std::string hex = std::string(1, advance()) + std::string(1, advance());
        value.push_back(static_cast<char>(std::strtol(hex.c_str(), nullptr, 16)));
        break;
      }
      case 'u': {
        if (peek() != '{') {
          diagnostics_.error("`\\u` needs braces, like `\\u{1F600}`",
                             SourceSpan{start, position_ - start, startLine, startColumn});
          break;
        }
        advance();  // '{'
        std::string hex;
        while (!atEnd() && peek() != '}') hex.push_back(advance());
        if (peek() == '}') advance();
        const unsigned long code = std::strtoul(hex.c_str(), nullptr, 16);
        // Encode as UTF-8.
        if (code < 0x80) {
          value.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
          value.push_back(static_cast<char>(0xC0 | (code >> 6)));
          value.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
          value.push_back(static_cast<char>(0xE0 | (code >> 12)));
          value.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
          value.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
        break;
      }
      default:
        diagnostics_.warning(std::string("unknown escape sequence `\\") + escaped + "`",
                             SourceSpan{start, position_ - start, startLine, startColumn});
        value.push_back(escaped);
        break;
    }
  }

  if (!terminated) {
    diagnostics_.error("this string is missing its closing quote",
                       SourceSpan{start, position_ - start, startLine, startColumn});
  }

  pushToken(Token(TokenKind::String, makeSpan(start, startLine, startColumn), value));
}

void Lexer::lexIdentifier() {
  const std::size_t start = position_;
  const int startLine = line_;
  const int startColumn = column_;

  while (!atEnd() && isIdentContinue(peek())) advance();

  // Look ahead for a hyphenated word such as `fine-tune`.
  if (peek() == '-' && isIdentStart(peek(1))) {
    const std::size_t save = position_;
    const int saveLine = line_;
    const int saveColumn = column_;

    std::string word = source_.substr(start, position_ - start);
    while (!atEnd() && (isIdentContinue(peek()) || peek() == '-')) {
      word.push_back(advance());
    }
    if (hyphenatedWords().count(word) != 0) {
      pushToken(Token(TokenKind::Identifier, makeSpan(start, startLine, startColumn), word));
      return;
    }
    // Not a word we know: rewind and let '-' be an operator.
    position_ = save;
    line_ = saveLine;
    column_ = saveColumn;
  }

  const std::string text = source_.substr(start, position_ - start);
  pushToken(Token(TokenKind::Identifier, makeSpan(start, startLine, startColumn), text));
}

void Lexer::lexOperator() {
  const std::size_t start = position_;
  const int startLine = line_;
  const int startColumn = column_;
  const char c = advance();

  auto emit = [&](TokenKind kind, std::size_t length) {
    pushToken(Token(kind, makeSpan(start, startLine, startColumn),
                    source_.substr(start, length)));
  };

  switch (c) {
    case '+':
      emit(TokenKind::Plus, 1);
      return;
    case '-':
      if (peek() == '>') {
        advance();
        emit(TokenKind::Arrow, 2);
      } else {
        emit(TokenKind::Minus, 1);
      }
      return;
    case '*':
      if (peek() == '*') {
        advance();
        emit(TokenKind::StarStar, 2);
      } else {
        emit(TokenKind::Star, 1);
      }
      return;
    case '/':
      if (peek() == '/') {
        advance();
        emit(TokenKind::SlashSlash, 2);
      } else {
        emit(TokenKind::Slash, 1);
      }
      return;
    case '%':
      emit(TokenKind::Percent, 1);
      return;
    case '=':
      if (peek() == '=') {
        advance();
        emit(TokenKind::EqualEqual, 2);
      } else {
        emit(TokenKind::Equal, 1);
      }
      return;
    case '!':
      if (peek() == '=') {
        advance();
        emit(TokenKind::BangEqual, 2);
      } else {
        emit(TokenKind::Bang, 1);
      }
      return;
    case '<':
      if (peek() == '=') {
        advance();
        emit(TokenKind::LessEqual, 2);
      } else {
        emit(TokenKind::Less, 1);
      }
      return;
    case '>':
      if (peek() == '=') {
        advance();
        emit(TokenKind::GreaterEqual, 2);
      } else {
        emit(TokenKind::Greater, 1);
      }
      return;
    case '(':
      ++bracket_depth_;
      emit(TokenKind::LeftParen, 1);
      return;
    case ')':
      if (bracket_depth_ > 0) --bracket_depth_;
      emit(TokenKind::RightParen, 1);
      return;
    case '[':
      ++bracket_depth_;
      emit(TokenKind::LeftBracket, 1);
      return;
    case ']':
      if (bracket_depth_ > 0) --bracket_depth_;
      emit(TokenKind::RightBracket, 1);
      return;
    case '{':
      ++bracket_depth_;
      emit(TokenKind::LeftBrace, 1);
      return;
    case '}':
      if (bracket_depth_ > 0) --bracket_depth_;
      emit(TokenKind::RightBrace, 1);
      return;
    case ',':
      emit(TokenKind::Comma, 1);
      return;
    case ':':
      emit(TokenKind::Colon, 1);
      return;
    case '.':
      emit(TokenKind::Dot, 1);
      return;
    case ';':
      emit(TokenKind::Semicolon, 1);
      return;
    case '?':
      emit(TokenKind::Question, 1);
      return;
    default:
      break;
  }

  diagnostics_.error(std::string("I do not understand the character `") + c + "`",
                     SourceSpan{start, 1, startLine, startColumn});
  pushToken(Token(TokenKind::Unknown, makeSpan(start, startLine, startColumn),
                  std::string(1, c)));
}

}  // namespace milo
