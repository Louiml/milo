// The compiler: an AST becomes bytecode.
#ifndef MILO_CORE_VM_COMPILER_HPP
#define MILO_CORE_VM_COMPILER_HPP

#include <vector>

#include "core/ast.hpp"
#include "core/diagnostic.hpp"
#include "core/vm/bytecode.hpp"

namespace milo::vm {

/// Compile a program.
///
/// The compiler never throws: a problem becomes a diagnostic. When the
/// diagnostics contain an error, the program is not runnable.
class Compiler {
 public:
  /// Compile a list of statements.
  static bool compile(const std::vector<ast::StmtPtr>& statements, Program& out,
                      DiagnosticBag& diagnostics);

 private:
  struct Loop {
    /// Where `continue` jumps to.
    std::size_t continueTarget = 0;
    /// The jumps that have to be patched to the end of the loop.
    std::vector<std::size_t> breakPatches;
    /// The jumps that have to be patched back to the start of the loop.
    std::vector<std::size_t> continuePatches;
  };

  void compileStatement(const ast::Stmt& statement);
  void compileBlock(const std::vector<ast::StmtPtr>& body);
  void compileExpression(const ast::Expr& expression);

  void compileIf(const ast::IfStmt& statement);
  void compileWhile(const ast::WhileStmt& statement);
  void compileFor(const ast::ForStmt& statement);
  void compileFunction(const ast::FnStmt& statement);
  void compileSentence(const ast::SentenceStmt& statement);

  /// Emit an instruction.
  void emit(Op op, std::uint32_t a = 0, std::uint32_t b = 0);
  /// Emit a jump whose target is not known yet, and return its place.
  std::size_t emitJump(Op op);
  /// Point a jump at the instruction that comes next.
  void patchJump(std::size_t place);

  /// The chunk being filled in.
  Chunk* chunk_ = nullptr;
  /// The program being built.
  Program* program_ = nullptr;
  /// The diagnostics.
  DiagnosticBag* diagnostics_ = nullptr;
  /// The loops we are inside.
  std::vector<Loop> loops_;
  /// How many values the current expression has left on the stack.
  int depth_ = 0;
};

}  // namespace milo::vm

#endif  // MILO_CORE_VM_COMPILER_HPP
