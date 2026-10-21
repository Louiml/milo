// The tree-walking evaluator.
//
// Phase 1 ships an AST interpreter rather than a bytecode VM. The reason
// is deliberate: the grammar is the thing most likely to change, and
// putting a code generator in front of an unstable grammar slows down
// every grammar experiment. The evaluator is small enough to keep as a
// reference implementation, and it becomes the differential-testing
// oracle for the VM that arrives in Phase 2.
#ifndef MILO_RUNTIME_INTERP_HPP
#define MILO_RUNTIME_INTERP_HPP

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "core/ast.hpp"
#include "runtime/environment.hpp"
#include "runtime/value.hpp"

namespace milo {

namespace python {
class Context;
}

/// A runtime error.
///
/// The interpreter throws this; the CLI and the C ABI catch it and turn
/// it into a diagnostic.
struct RuntimeError {
  std::string message;
  SourceSpan span;
  /// An optional "did you mean ...?" hint.
  std::string suggestion;
};

/// Evaluates a Milo program.
class Interpreter {
 public:
  Interpreter();
  ~Interpreter();

  /// Run a whole program in the global scope.
  runtime::Value runProgram(const std::vector<ast::StmtPtr>& program);

  /// Run a sequence of statements, returning the value of the last one.
  runtime::Value execute(const std::vector<ast::StmtPtr>& body,
                          const std::shared_ptr<runtime::Environment>& env);

  /// Call a function value.
  runtime::Value call(runtime::Value callee, const std::vector<runtime::Value>& args,
                      const std::vector<std::pair<std::string, runtime::Value>>& kwargs);

  /// The Python interpreter, started when it is first needed.
  python::Context& pythonContext();

  // --- the operations on values -------------------------------------------
  //
  // These are public because the bytecode machine uses them too: both
  // engines must agree on what `+` means.

  /// A binary operation on two values.
  runtime::Value binaryOperation(const std::string& op, const runtime::Value& lhs,
                                 const runtime::Value& rhs, SourceSpan span);

  /// A unary operation on one value: `-x`, `not x`, `+x`.
  runtime::Value unaryOperation(const std::string& op, const runtime::Value& value, SourceSpan span);

  /// Index into a value.
  runtime::Value indexOperation(const runtime::Value& target,
                                const std::vector<runtime::Value>& indices, SourceSpan span);

  /// Read one of a value's fields.
  runtime::Value memberOperation(const runtime::Value& target, const std::string& name,
                                 SourceSpan span);

  /// A type test: `value is name`.
  runtime::Value typeTestOperation(const runtime::Value& value, const std::string& type);

  /// Call a value that is not a user function.
  ///
  /// Returns false when the value is a user function, which the caller
  /// then has to deal with itself.
  bool callSpecialValue(const runtime::Value& callee, const std::vector<runtime::Value>& args,
                        const std::vector<std::pair<std::string, runtime::Value>>& kwargs,
                        runtime::Value& out);

  /// Report a runtime error.
  [[noreturn]] void raise(const std::string& message, SourceSpan span);
  [[noreturn]] void raiseUnknownName(const std::string& name, SourceSpan span);
  [[noreturn]] void raiseWrongType(const std::string& what, const runtime::Value& value,
                                   SourceSpan span);

  const std::shared_ptr<runtime::Environment>& globals() const { return globals_; }

  /// Where `print` and the informational notes go.
  void setOutput(std::ostream* out) { output_ = out; }
  std::ostream& out() const;

  /// Where informational notes go (kept apart from program output).
  void setNotes(std::ostream* out) { notes_ = out; }
  std::ostream& notes() const;

  /// Register a builtin function.
  void defineBuiltin(const std::string& name,
                     runtime::Value (*fn)(Interpreter&, const std::vector<runtime::Value>&,
                                          const std::vector<std::pair<std::string,
                                                                      runtime::Value>>&));

 private:
  runtime::Value eval(const ast::Expr& expr, const std::shared_ptr<runtime::Environment>& env);
  runtime::Value evalBinary(const ast::BinaryExpr& node,
                            const std::shared_ptr<runtime::Environment>& env);
  runtime::Value tensorBinary(const std::string& op, const runtime::Value& lhs,
                              const runtime::Value& rhs, SourceSpan span);
  runtime::Value evalCall(const ast::CallExpr& node,
                          const std::shared_ptr<runtime::Environment>& env);
  runtime::Value evalMember(const ast::MemberExpr& node,
                             const std::shared_ptr<runtime::Environment>& env);
  runtime::Value evalIndex(const ast::IndexExpr& node,
                           const std::shared_ptr<runtime::Environment>& env);
  runtime::Value indexTensor(const std::shared_ptr<tensor::Tensor>& tensor,
                             const std::vector<runtime::Value>& indices, SourceSpan span);

  runtime::Value exec(ast::Stmt& stmt, const std::shared_ptr<runtime::Environment>& env);
  void execLet(const ast::LetStmt& node, const std::shared_ptr<runtime::Environment>& env);
  void execAssign(const ast::AssignStmt& node, const std::shared_ptr<runtime::Environment>& env);
  void execSentence(ast::SentenceStmt& node, const std::shared_ptr<runtime::Environment>& env);
  void execSentenceClause(const ast::SentenceClause& clause,
                          const std::shared_ptr<runtime::Environment>& env,
                          const std::vector<ast::SentenceOption>& extra_options);

  runtime::Value evalOptions(const std::vector<ast::SentenceOption>& clause_options,
                             const std::vector<ast::SentenceOption>& sentence_options,
                             const std::shared_ptr<runtime::Environment>& env);

  std::shared_ptr<runtime::Environment> globals_;
  std::ostream* output_ = nullptr;
  std::ostream* notes_ = nullptr;
  int depth_ = 0;
  /// The Python interpreter, if the program has used one.
  std::unique_ptr<python::Context> python_;
};

}  // namespace milo

#endif  // MILO_RUNTIME_INTERP_HPP
