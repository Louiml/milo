// A small, strict JSON parser.
//
// The project has no third-party dependencies, so this is what reads the
// safetensors header. It is deliberately strict: trailing commas,
// unquoted keys, comments, and NaN are all errors. A file that is not
// valid JSON is a file we do not trust.
#ifndef MILO_CORE_JSON_HPP
#define MILO_CORE_JSON_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace milo::json {

/// A JSON value.
class Value {
 public:
  enum class Kind {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
  };

  Value() = default;
  static Value null();
  static Value boolean(bool value);
  static Value number(double value);
  static Value string(std::string value);
  static Value array(std::vector<Value> value);
  static Value object(std::map<std::string, Value> value);

  Kind kind() const { return kind_; }
  bool isNull() const { return kind_ == Kind::Null; }
  bool isBool() const { return kind_ == Kind::Bool; }
  bool isNumber() const { return kind_ == Kind::Number; }
  bool isString() const { return kind_ == Kind::String; }
  bool isArray() const { return kind_ == Kind::Array; }
  bool isObject() const { return kind_ == Kind::Object; }

  bool asBool() const { return bool_; }
  double asNumber() const { return number_; }
  const std::string& asString() const { return string_; }
  const std::vector<Value>& asArray() const { return array_; }
  const std::map<std::string, Value>& asObject() const { return object_; }

  /// Look up a key in an object. Returns nullptr when it is not there.
  const Value* find(const std::string& key) const;

 private:
  Kind kind_ = Kind::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Value> array_;
  std::map<std::string, Value> object_;
};

/// A parse failure, with the 1-based line and column.
struct ParseError {
  std::size_t line = 1;
  std::size_t column = 1;
  std::string message;

  std::string toString() const;
};

/// Parse a complete JSON document. Trailing content is an error.
bool parse(const std::string& text, Value& out, ParseError& error);

/// Escape a string as a JSON string, including the surrounding quotes.
std::string escape(const std::string& text);

}  // namespace milo::json

#endif  // MILO_CORE_JSON_HPP
