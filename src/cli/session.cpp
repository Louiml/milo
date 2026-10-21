#include "cli/session.hpp"

#include <utility>

#include "core/lexer.hpp"
#include "core/parser.hpp"
#include "core/vm/compiler.hpp"
#include "core/vm/machine.hpp"

namespace milo::cli {

Session::Session(std::string filename)
    : filename_(std::move(filename)), interpreter_(std::make_unique<Interpreter>()) {}

RunOutcome Session::check(const std::string& source) {
  diagnostics_.clear();

  Lexer lexer(source, filename_);
  std::vector<Token> tokens = lexer.tokenize();
  if (lexer.diagnostics().hasErrors()) {
    for (Diagnostic& diagnostic : lexer.diagnostics().diagnostics()) {
      diagnostics_.add(std::move(diagnostic));
    }
    return RunOutcome::ParseError;
  }

  Parser parser(std::move(tokens), filename_);
  parser.parseProgram();
  if (parser.diagnostics().hasErrors()) {
    for (Diagnostic& diagnostic : parser.diagnostics().diagnostics()) {
      diagnostics_.add(std::move(diagnostic));
    }
    return RunOutcome::ParseError;
  }

  return RunOutcome::Ok;
}

RunOutcome Session::run(const std::string& source) {
  diagnostics_.clear();

  Lexer lexer(source, filename_);
  std::vector<Token> tokens = lexer.tokenize();
  if (lexer.diagnostics().hasErrors()) {
    for (Diagnostic& diagnostic : lexer.diagnostics().diagnostics()) {
      diagnostics_.add(std::move(diagnostic));
    }
    return RunOutcome::ParseError;
  }

  Parser parser(std::move(tokens), filename_);
  std::vector<ast::StmtPtr> program = parser.parseProgram();
  if (parser.diagnostics().hasErrors()) {
    for (Diagnostic& diagnostic : parser.diagnostics().diagnostics()) {
      diagnostics_.add(std::move(diagnostic));
    }
    return RunOutcome::ParseError;
  }

  try {
    if (engine_ == Engine::Machine) {
      // The bytecode machine. It uses the interpreter's operations, so the
      // two engines cannot disagree.
      vm::Program compiled;
      vm::Compiler compiler;
      if (!vm::Compiler::compile(program, compiled, diagnostics_)) {
        return RunOutcome::ParseError;
      }
      vm::Machine machine(*interpreter_);
      last_value_ = machine.run(compiled);
    } else {
      last_value_ = interpreter_->runProgram(program);
    }
  } catch (const RuntimeError& error) {
    Diagnostic diagnostic(Severity::Error, error.message, error.span);
    if (!error.suggestion.empty()) diagnostic.label = error.suggestion;
    diagnostics_.add(std::move(diagnostic));
    return RunOutcome::RuntimeError;
  }

  return RunOutcome::Ok;
}

}  // namespace milo::cli
