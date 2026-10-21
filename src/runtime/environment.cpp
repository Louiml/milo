#include "runtime/environment.hpp"

namespace milo::runtime {

const Value* Environment::find(const std::string& name) const {
  const Environment* scope = this;
  while (scope != nullptr) {
    const auto it = scope->values_.find(name);
    if (it != scope->values_.end()) return &it->second;
    scope = scope->parent_.get();
  }
  return nullptr;
}

const Value* Environment::findLocal(const std::string& name) const {
  const auto it = values_.find(name);
  if (it != values_.end()) return &it->second;
  return nullptr;
}

void Environment::define(const std::string& name, Value value) {
  // A function closes over a scope. This scope is the one that holds the
  // function, so it takes over the job of keeping that scope alive.
  if (value.kind() == Value::Kind::Function) {
    const auto& function = value.asFunction();
    if (auto scope = function->closure.lock()) {
      closures_.push_back(std::move(scope));
    }
  }
  values_[name] = std::move(value);
}

bool Environment::assign(const std::string& name, Value value) {
  Environment* scope = this;
  while (scope != nullptr) {
    const auto it = scope->values_.find(name);
    if (it != scope->values_.end()) {
      it->second = std::move(value);
      return true;
    }
    scope = scope->parent_.get();
  }
  return false;
}

std::vector<std::string> Environment::visibleNames() const {
  std::vector<std::string> names;
  const Environment* scope = this;
  while (scope != nullptr) {
    for (const auto& entry : scope->values_) {
      names.push_back(entry.first);
    }
    scope = scope->parent_.get();
  }
  return names;
}

}  // namespace runtime
