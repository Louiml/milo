// The bytecode machine.
#ifndef MILO_CORE_VM_MACHINE_HPP
#define MILO_CORE_VM_MACHINE_HPP

#include <memory>
#include <vector>

#include "core/vm/bytecode.hpp"
#include "runtime/interp.hpp"

namespace milo::vm {

/// The bytecode machine.
///
/// It is a stack machine: values are pushed and popped, and names are
/// looked up in a chain of scopes. It uses the runtime's operations, so
/// it and the tree-walking evaluator cannot disagree about what `+` means.
class Machine {
 public:
  explicit Machine(Interpreter& interpreter) : interpreter_(interpreter) {}

  /// Run a compiled program. Returns the value of the last statement.
  runtime::Value run(const Program& program);

 private:
  struct Frame {
    /// The scope the code in this frame runs in.
    std::shared_ptr<runtime::Environment> env;
    /// Where the operand stack starts.
    std::size_t stackBase = 0;
    /// The instruction to carry on with when the frame returns.
    std::size_t returnAddress = 0;
    /// The code the caller was in.
    const Chunk* chunk = nullptr;
  };

  /// Call a compiled function.
  runtime::Value callFunction(const std::shared_ptr<Function>& function,
                              const std::vector<runtime::Value>& args);

  /// Start running a function's code.
  void enterFunction(const std::shared_ptr<Function>& function,
                     const std::vector<runtime::Value>& args);

  void push(runtime::Value value);
  runtime::Value pop();
  runtime::Value& peek();

  Interpreter& interpreter_;
  const Program* program_ = nullptr;
  std::vector<runtime::Value> stack_;
  std::vector<Frame> frames_;
  /// The code being run.
  const Chunk* currentChunk_ = nullptr;
  /// Where we are in it.
  std::size_t currentIp_ = 0;
  int depth_ = 0;
};

}  // namespace milo::vm

#endif  // MILO_CORE_VM_MACHINE_HPP
