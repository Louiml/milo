// The bytecode.
//
// Milo has two execution engines. The tree-walking evaluator is the
// reference: it is simple, and it is what the tests are written against.
// The bytecode machine is the fast one.
//
// The machine is a stack machine. Values are pushed and popped; names are
// looked up in a chain of scopes. The instruction set is deliberately
// small, because every opcode is a place a future compiler has to
// understand.
#ifndef MILO_CORE_VM_BYTECODE_HPP
#define MILO_CORE_VM_BYTECODE_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/ast.hpp"
#include "runtime/value.hpp"

namespace milo::vm {

/// The instructions.
enum class Op {
  Nop,
  /// Push a constant.
  Constant,
  Null,
  True,
  False,
  /// Throw the top of the stack away.
  Pop,
  /// Copy the top of the stack.
  Dup,
  /// Push the value of a name.
  LoadName,
  /// Pop a value and assign it to a name.
  StoreName,
  /// Pop a value and declare it in the current scope.
  DefineName,
  /// Pop a value and bind it to a name, replacing any binding there is.
  ///
  /// This is what a loop's variable does.
  DefineVar,
  /// Let the current scope be declared in again.
  ///
  /// A loop's body may declare the same name on every turn.
  AllowRedeclaration,
  /// Stop allowing the current scope to be declared in again.
  DisallowRedeclaration,
  /// Replace the value on top of the stack with one of its fields.
  GetAttr,
  /// Call the value below the arguments.
  ///
  /// The first operand is the number of positional arguments. The second
  /// is the group of keyword-argument names, plus one; zero means there
  /// are none.
  Call,
  /// Return from a function.
  Return,
  /// Jump.
  Jump,
  /// Pop a value and jump if it is false.
  JumpIfFalse,
  /// Jump if the value on top of the stack is false, leaving it there.
  JumpIfFalseKeep,
  /// Pop a value and jump if it is true.
  JumpIfTrue,
  /// Jump if the value on top of the stack is true, leaving it there.
  JumpIfTrueKeep,
  /// Replace an iterable with an iterator.
  GetIter,
  /// Advance an iterator; jump if it is finished.
  ForIter,
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  Pow,
  Neg,
  Not,
  Eq,
  Ne,
  Lt,
  Le,
  Gt,
  Ge,
  /// Index into the value below the indices.
  Index,
  /// Assign through an index.
  SetIndex,
  /// Join two strings.
  Concat,
  /// Build a list from the values on the stack.
  MakeList,
  /// Build a dictionary from the key/value pairs on the stack.
  MakeDict,
  /// A type test: `value is name`.
  TypeTest,
  /// Make a function.
  Closure,
  /// Import a Python module and push it.
  Import,
  /// Carry out a sentence. The slots and the options are on the stack.
  Sentence,
};

/// One instruction.
struct Instruction {
  Op op = Op::Nop;
  /// The first operand: a constant, a name, or a count.
  std::uint32_t a = 0;
  /// The second operand: a count.
  std::uint32_t b = 0;
};

/// A block of instructions, with its constants and its names.
struct Chunk {
  std::vector<Instruction> code;
  std::vector<runtime::Value> constants;
  std::vector<std::string> names;
  /// The names of the keyword arguments of each call, one group per call
  /// that has any.
  std::vector<std::vector<std::string>> keywordGroups;

  /// The index of a constant, adding it if it is not there yet.
  std::uint32_t addConstant(runtime::Value value);
  /// The index of a name, adding it if it is not there yet.
  std::uint32_t addName(const std::string& name);
  /// The index of a group of keyword-argument names.
  std::uint32_t addKeywordGroup(std::vector<std::string> names);
};

/// A compiled function.
struct Function {
  std::string name;
  /// The names of the parameters, in order.
  std::vector<std::string> params;
  /// How many parameters it takes.
  std::uint32_t arity = 0;
  /// The instructions.
  Chunk chunk;
  /// The scope the function was defined in.
  ///
  /// A weak reference, for the same reason as in the runtime: a function
  /// is usually defined in the scope it closes over, and a strong
  /// reference would make the two hold each other up for ever.
  std::weak_ptr<runtime::Environment> closure;
};

/// A sentence the machine can carry out.
///
/// The sentences are flattened at compile time: the slots and the options
/// become constants, so only the names are left to remember.
struct CompiledSentence {
  /// The verb: `load`, `split`, `fine-tune`, ...
  std::string verb;
  /// The names of the slots, in the order their values are pushed.
  std::vector<std::string> slotNames;
  /// The keys of the options, in the order their values are pushed.
  std::vector<std::string> optionKeys;
};

/// A whole program: the top-level code, and every function in it.
struct Program {
  Chunk main;
  std::vector<std::shared_ptr<Function>> functions;
  /// The sentences the program uses, in the order they were compiled.
  std::vector<CompiledSentence> sentences;
};

}  // namespace milo::vm

#endif  // MILO_CORE_VM_BYTECODE_HPP
