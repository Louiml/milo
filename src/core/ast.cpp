#include "core/ast.hpp"

#include <sstream>

namespace milo::ast {
namespace {

std::string joinShape(const std::vector<std::int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ", ";
    out << shape[i];
  }
  out << ']';
  return out.str();
}

}  // namespace

std::string TypeRef::toString() const {
  std::ostringstream out;
  out << name;
  if (has_shape) out << joinShape(shape);
  if (!dtype.empty()) out << " of " << dtype;
  if (element) out << " of " << element->toString();
  if (value_type) out << "[" << (element ? element->toString() : std::string("any")) << ", "
                      << value_type->toString() << "]";
  return out.str();
}

std::string IntLiteral::describe() const { return "the number " + std::to_string(value); }
std::string FloatLiteral::describe() const { return "the number " + std::to_string(value); }
std::string StringLiteral::describe() const { return "the string \"" + value + "\""; }
std::string BoolLiteral::describe() const { return value ? "`true`" : "`false`"; }
std::string NullLiteral::describe() const { return "`null`"; }
std::string NameExpr::describe() const { return "the name `" + name + "`"; }
std::string ListExpr::describe() const { return "a list"; }
std::string DictExpr::describe() const { return "a dictionary"; }
std::string UnaryExpr::describe() const { return "the expression `" + op + " ...`"; }
std::string BinaryExpr::describe() const { return "the expression `" + op + "`"; }
std::string CallExpr::describe() const {
  return callee ? callee->describe() : std::string("a function call");
}
std::string IndexExpr::describe() const {
  return target ? target->describe() : std::string("an index expression");
}
std::string MemberExpr::describe() const {
  return target ? target->describe() : std::string("a member access");
}
std::string TypeTestExpr::describe() const {
  return operand ? operand->describe() : std::string("a type test");
}

}  // namespace milo::ast
