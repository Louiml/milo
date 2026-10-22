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

  /// The scope this one sits in.
  ///
  /// This is a weak reference, and it has to be. A function closes over
  /// the scope it was written in, and a scope holds the functions written
  /// in it. With strong references on both sides, a scope that holds a
  /// function that was returned out of it would keep that function's scope
  /// alive, and that scope keeps the global scope alive, and the global
  /// scope keeps the function: the three would hold each other for ever.
  ///
  /// A scope that is in use keeps its own parent alive: the call that
  /// made it holds a reference for as long as the call lasts.
  std::shared_ptr<Environment> parent() const { return parent_.lock(); }

  /// Find a name, or nullptr when it is not bound anywhere.
  const Value* find(const std::string& name) const;

  /// Find a name in this scope only.
  const Value* findLocal(const std::string& name) const;

  /// Declare a name in this scope, replacing any existing binding.
  void define(const std::string& name, Value value);

  /// Declare a name in this scope, for a value that is a function.
  ///
  /// A function that is stored here but was written somewhere else takes
  /// that scope with it, and the scope lets go of the function, so that
  /// the two do not hold each other up for ever.
  void defineFunction(const std::string& name, Value value);

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

  /// Let go of everything in this scope.
  ///
  /// A function closes over the scope it was written in, and that scope
  /// holds the function: the two keep each other alive, and the scope is
  /// never given back. Clearing the scope breaks that, and is safe once
  /// the scope can no longer be reached by name.
  void clear() {
    values_.clear();
  }

  /// Let go of one name.
  ///
  /// A function that leaves the scope it was written in takes the scope
  /// with it, and the scope lets go of the function, so that the two do
  /// not hold each other up.
  void release(const std::string& name) { values_.erase(name); }

 private:
  /// The scope this one sits in.
  std::weak_ptr<Environment> parent_;
  std::unordered_map<std::string, Value> values_;
  bool allow_redeclaration_ = false;
};

}  // namespace runtime

#endif  // MILO_RUNTIME_ENVIRONMENT_HPP
