// Lexical scopes.
#ifndef MILO_RUNTIME_ENVIRONMENT_HPP
#define MILO_RUNTIME_ENVIRONMENT_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/value.hpp"

namespace milo::runtime {

/// A lexical scope.
///
/// Scopes are chained: a name is looked up in the current scope, then in
/// its parent, and so on to the global scope. Assignment walks the same
/// chain; declaring always happens in the current scope.
class Environment {
 public:
  explicit Environment(std::shared_ptr<Environment> parent = nullptr)
      : parent_(std::move(parent)) {}

  const std::shared_ptr<Environment>& parent() const { return parent_; }

  /// Find a name, or nullptr when it is not bound anywhere.
  const Value* find(const std::string& name) const;

  /// Find a name in this scope only.
  const Value* findLocal(const std::string& name) const;

  /// Declare a name in this scope, replacing any existing binding.
  void define(const std::string& name, Value value);

  /// Assign to an existing name, walking outwards. Returns false when the
  /// name is not bound anywhere.
  bool assign(const std::string& name, Value value);

  /// Every name visible from this scope, nearest first. Used for "did you
  /// mean" suggestions.
  std::vector<std::string> visibleNames() const;

  /// The number of names bound directly in this scope.
  std::size_t localCount() const { return values_.size(); }

  /// May a name be declared again in this scope?
  ///
  /// Inside a loop body, a name may be declared on every iteration. This
  /// is the flag that allows it.
  void setAllowRedeclaration(bool value) { allow_redeclaration_ = value; }
  bool allowRedeclaration() const { return allow_redeclaration_; }

  /// Is this the very same scope as `other`?
  bool is(const Environment* other) const { return this == other; }

  /// Keep a scope alive on behalf of a function that closes over it.
  ///
  /// A function closes over the scope it was written in, and that scope
  /// holds the function. The two would keep each other alive for ever, so
  /// the function's own reference is weak and the scope keeps it alive
  /// here instead.
  void holdClosure(std::shared_ptr<Environment> scope) {
    if (scope) closures_.push_back(std::move(scope));
  }

 private:
  std::shared_ptr<Environment> parent_;
  std::unordered_map<std::string, Value> values_;
  /// The scopes that the functions in this scope close over.
  ///
  /// A function closes over the scope it was written in, and that scope
  /// holds the function: the two would keep each other alive for ever.
  /// The scope holds on to the closed-over scope here instead, so that the
  /// function's own reference can be weak.
  std::vector<std::shared_ptr<Environment>> closures_;
  bool allow_redeclaration_ = false;
};

}  // namespace runtime

#endif  // MILO_RUNTIME_ENVIRONMENT_HPP
