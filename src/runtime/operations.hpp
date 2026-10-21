// The operations on values.
//
// Both execution engines use these, so that the tree-walking evaluator
// and the bytecode machine cannot drift apart.
#ifndef MILO_RUNTIME_OPERATIONS_HPP
#define MILO_RUNTIME_OPERATIONS_HPP

#include <string>
#include <utility>
#include <vector>

#include "core/span.hpp"
#include "runtime/environment.hpp"
#include "runtime/value.hpp"

namespace milo::runtime {

/// Are two values equal?
bool valuesEqual(const Value& lhs, const Value& rhs);

/// Compare two values. Returns false when they cannot be compared.
bool valuesCompare(const Value& lhs, const Value& rhs, int& result);

/// A binary operation on two values.
Value binaryOperation(const std::string& op, const Value& lhs, const Value& rhs, SourceSpan span);

/// A unary operation on one value.
Value unaryOperation(const std::string& op, const Value& value, SourceSpan span);

/// Index into a value.
Value indexOperation(const Value& target, const std::vector<Value>& indices, SourceSpan span);

/// Read one of a value's fields.
Value memberOperation(const Value& target, const std::string& name, SourceSpan span);

/// A type test: `value is name`.
Value typeTestOperation(const Value& value, const std::string& type);

/// Call a value that is not a user function: a builtin, a layer, or an
/// optimiser.
///
/// Returns false when the value is a user function, which the caller then
/// has to deal with itself.
bool callSpecialValue(const Value& callee, const std::vector<Value>& args,
                      const std::vector<std::pair<std::string, Value>>& kwargs,
                      const std::shared_ptr<Environment>& env, Value& out, SourceSpan span);

}  // namespace milo::runtime

#endif  // MILO_RUNTIME_OPERATIONS_HPP
