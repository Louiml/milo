#include "core/json.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace milo::json {
namespace {

/// A recursive-descent JSON parser.
class Parser {
 public:
  Parser(const std::string& text) : text_(text) {}

  bool parseDocument(Value& out, ParseError& error) {
    skipWhitespace();
    if (!parseValue(out)) {
      error = error_;
      return false;
    }
    skipWhitespace();
    if (position_ != text_.size()) {
      fail("there is more text after the JSON value");
      error = error_;
      return false;
    }
    return true;
  }

 private:
  bool atEnd() const { return position_ >= text_.size(); }
  char peek() const { return atEnd() ? '\0' : text_[position_]; }
  char advance() {
    const char c = text_[position_++];
    if (c == '\n') {
      ++line_;
      column_ = 1;
    } else {
      ++column_;
    }
    return c;
  }

  void skipWhitespace() {
    while (!atEnd()) {
      const char c = peek();
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        advance();
        continue;
      }
      break;
    }
  }

  bool fail(const std::string& message) {
    if (error_.message.empty()) {
      error_.line = line_;
      error_.column = column_;
      error_.message = message;
    }
    return false;
  }

  bool parseValue(Value& out) {
    if (atEnd()) return fail("the JSON value is missing");
    switch (peek()) {
      case '{':
        return parseObject(out);
      case '[':
        return parseArray(out);
      case '"': {
        std::string text;
        if (!parseString(text)) return false;
        out = Value::string(std::move(text));
        return true;
      }
      case 't':
        if (!expectLiteral("true")) return false;
        out = Value::boolean(true);
        return true;
      case 'f':
        if (!expectLiteral("false")) return false;
        out = Value::boolean(false);
        return true;
      case 'n':
        if (!expectLiteral("null")) return false;
        out = Value::null();
        return true;
      default:
        return parseNumber(out);
    }
  }

  bool expectLiteral(const char* literal) {
    for (const char* p = literal; *p != '\0'; ++p) {
      if (atEnd() || peek() != *p) {
        return fail(std::string("expected `") + literal + "`");
      }
      advance();
    }
    return true;
  }

  bool parseObject(Value& out) {
    advance();  // '{'
    std::map<std::string, Value> members;

    skipWhitespace();
    if (peek() == '}') {
      advance();
      out = Value::object(std::move(members));
      return true;
    }

    while (true) {
      skipWhitespace();
      if (peek() != '"') {
        return fail("a key in a JSON object must be a string");
      }
      std::string key;
      if (!parseString(key)) return false;

      skipWhitespace();
      if (peek() != ':') return fail("expected ':' after a key");
      advance();

      skipWhitespace();
      Value value;
      if (!parseValue(value)) return false;
      members[key] = std::move(value);

      skipWhitespace();
      if (peek() == ',') {
        advance();
        skipWhitespace();
        if (peek() == '}') return fail("a JSON object may not have a trailing comma");
        continue;
      }
      if (peek() == '}') {
        advance();
        break;
      }
      return fail("expected ',' or '}' in a JSON object");
    }

    out = Value::object(std::move(members));
    return true;
  }

  bool parseArray(Value& out) {
    advance();  // '['
    std::vector<Value> items;

    skipWhitespace();
    if (peek() == ']') {
      advance();
      out = Value::array(std::move(items));
      return true;
    }

    while (true) {
      skipWhitespace();
      Value value;
      if (!parseValue(value)) return false;
      items.push_back(std::move(value));

      skipWhitespace();
      if (peek() == ',') {
        advance();
        skipWhitespace();
        if (peek() == ']') return fail("a JSON array may not have a trailing comma");
        continue;
      }
      if (peek() == ']') {
        advance();
        break;
      }
      return fail("expected ',' or ']' in a JSON array");
    }

    out = Value::array(std::move(items));
    return true;
  }

  bool parseString(std::string& out) {
    advance();  // opening quote
    std::string result;

    while (true) {
      if (atEnd()) return fail("a JSON string is not closed");
      const char c = advance();
      if (c == '"') break;
      if (c == '\\') {
        if (atEnd()) return fail("a JSON string is not closed");
        const char escaped = advance();
        switch (escaped) {
          case '"':
            result.push_back('"');
            break;
          case '\\':
            result.push_back('\\');
            break;
          case '/':
            result.push_back('/');
            break;
          case 'b':
            result.push_back('\b');
            break;
          case 'f':
            result.push_back('\f');
            break;
          case 'n':
            result.push_back('\n');
            break;
          case 'r':
            result.push_back('\r');
            break;
          case 't':
            result.push_back('\t');
            break;
          case 'u': {
            std::uint32_t code = 0;
            for (int i = 0; i < 4; ++i) {
              if (atEnd()) return fail("a JSON escape is incomplete");
              const char digit = advance();
              code <<= 4;
              if (digit >= '0' && digit <= '9') {
                code |= static_cast<std::uint32_t>(digit - '0');
              } else if (digit >= 'a' && digit <= 'f') {
                code |= static_cast<std::uint32_t>(digit - 'a' + 10);
              } else if (digit >= 'A' && digit <= 'F') {
                code |= static_cast<std::uint32_t>(digit - 'A' + 10);
              } else {
                return fail("a JSON escape has a bad hexadecimal digit");
              }
            }
            // Encode as UTF-8.
            if (code < 0x80) {
              result.push_back(static_cast<char>(code));
            } else if (code < 0x800) {
              result.push_back(static_cast<char>(0xC0 | (code >> 6)));
              result.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
              result.push_back(static_cast<char>(0xE0 | (code >> 12)));
              result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
              result.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            break;
          }
          default:
            return fail("a JSON string has an unknown escape");
        }
        continue;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        return fail("a JSON string may not contain a control character");
      }
      result.push_back(c);
    }

    out = std::move(result);
    return true;
  }

  bool parseNumber(Value& out) {
    const std::size_t start = position_;
    if (peek() == '-') advance();
    if (atEnd() || std::isdigit(static_cast<unsigned char>(peek())) == 0) {
      return fail("a JSON number is malformed");
    }
    if (peek() == '0') {
      advance();
    } else {
      while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())) != 0) advance();
    }
    if (peek() == '.') {
      advance();
      if (atEnd() || std::isdigit(static_cast<unsigned char>(peek())) == 0) {
        return fail("a JSON number is malformed");
      }
      while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())) != 0) advance();
    }
    if (peek() == 'e' || peek() == 'E') {
      advance();
      if (peek() == '+' || peek() == '-') advance();
      if (atEnd() || std::isdigit(static_cast<unsigned char>(peek())) == 0) {
        return fail("a JSON number is malformed");
      }
      while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())) != 0) advance();
    }

    const std::string text = text_.substr(start, position_ - start);
    out = Value::number(std::strtod(text.c_str(), nullptr));
    return true;
  }

  const std::string& text_;
  std::size_t position_ = 0;
  std::size_t line_ = 1;
  std::size_t column_ = 1;
  ParseError error_;
};

}  // namespace

Value Value::null() { return Value(); }

Value Value::boolean(bool value) {
  Value v;
  v.kind_ = Kind::Bool;
  v.bool_ = value;
  return v;
}

Value Value::number(double value) {
  Value v;
  v.kind_ = Kind::Number;
  v.number_ = value;
  return v;
}

Value Value::string(std::string value) {
  Value v;
  v.kind_ = Kind::String;
  v.string_ = std::move(value);
  return v;
}

Value Value::array(std::vector<Value> value) {
  Value v;
  v.kind_ = Kind::Array;
  v.array_ = std::move(value);
  return v;
}

Value Value::object(std::map<std::string, Value> value) {
  Value v;
  v.kind_ = Kind::Object;
  v.object_ = std::move(value);
  return v;
}

const Value* Value::find(const std::string& key) const {
  if (kind_ != Kind::Object) return nullptr;
  const auto it = object_.find(key);
  if (it == object_.end()) return nullptr;
  return &it->second;
}

std::string ParseError::toString() const {
  std::ostringstream out;
  out << "line " << line << ", column " << column << ": " << message;
  return out.str();
}

bool parse(const std::string& text, Value& out, ParseError& error) {
  Parser parser(text);
  return parser.parseDocument(out, error);
}

std::string escape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 2);
  out.push_back('"');
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          static const char* kHex = "0123456789abcdef";
          out += "\\u00";
          out.push_back(kHex[(static_cast<unsigned char>(c) >> 4) & 0xF]);
          out.push_back(kHex[static_cast<unsigned char>(c) & 0xF]);
        } else {
          out.push_back(c);
        }
        break;
    }
  }
  out.push_back('"');
  return out;
}

}  // namespace milo::json
