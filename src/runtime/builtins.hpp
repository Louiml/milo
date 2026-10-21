// The builtin functions.
#ifndef MILO_RUNTIME_BUILTINS_HPP
#define MILO_RUNTIME_BUILTINS_HPP

#include <vector>

#include "runtime/interp.hpp"
#include "runtime/value.hpp"

namespace milo::runtime {

/// Register every builtin in the interpreter's global scope.
void installBuiltins(Interpreter& interpreter);

/// Register the tensor builtins. Called by `installBuiltins`.
void installTensorBuiltins(Interpreter& interpreter);

/// Register the training builtins: layers, optimisers, and losses.
void installTrainingBuiltins(Interpreter& interpreter);

}  // namespace milo::runtime

#endif  // MILO_RUNTIME_BUILTINS_HPP
