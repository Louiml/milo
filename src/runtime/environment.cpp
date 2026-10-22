#include "runtime/environment.hpp"

namespace milo::runtime {

const Value* Environment::find(const std::string& name) const {
  const Environment* scope = this;
  // The parent is held weakly, so the walk keeps it alive as it goes.
  std::shared_ptr<Environment> held;
  while (scope != nullptr) {
    const auto it = scope->values_.find(name);
    if (it != scope->values_.end()) return &it->second;
    held = scope->parent();
    scope = held.get();
  }
  return nullptr;
}

const Value* Environment::findLocal(const std::string& name) const {
  const auto it = values_.find(name);
  if (it != values_.end()) return &it->second;
  return nullptr;
}

void Environment::define(const std::string& name, Value value) {
  values_[name] = std::move(value);
}

void Environment::defineFunction(const std::string& name, Value value) {
  // A function that is stored here but was written somewhere else takes
  // that scope with it, and the scope lets go of the function, so that
  // the two do not hold each other up for ever.
  if (value.kind() == Value::Kind::Function) {
    const auto& function = value.asFunction();
    if (auto closed = function->closure.lock()) {
      if (closed.get() != this) {
        function->keepAlive = closed;
        closed->release(function->name);
      }
    }
  }
  values_[name] = std::move(value);
}

bool Environment::assign(const std::string& name, Value value) {
  Environment* scope = this;
  std::shared_ptr<Environment> held;
  while (scope != nullptr) {
    const auto it = scope->values_.find(name);
    if (it != scope->values_.end()) {
      // A function that is stored here but was written somewhere else
      // takes that scope with it, and the scope lets go of the function,
      // so that the two do not hold each other up for ever.
      if (value.kind() == Value::Kind::Function) {
        const auto& function = value.asFunction();
        if (auto closed = function->closure.lock()) {
          if (closed.get() != scope) {
            function->keepAlive = closed;
            closed->release(function->name);
          }
        }
      }
      it->second = std::move(value);
      return true;
    }
    held = scope->parent();
    scope = held.get();
  }
  return false;
}

std::vector<std::string> Environment::visibleNames() const {
  std::vector<std::string> names;
  const Environment* scope = this;
  std::shared_ptr<Environment> held;
  while (scope != nullptr) {
    for (const auto& entry : scope->values_) {
      names.push_back(entry.first);
    }
    held = scope->parent();
    scope = held.get();
  }
  return names;
}

}  // namespace milo::runtime
