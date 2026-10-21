// A single Milo session: source in, diagnostics out.
#ifndef MILO_CLI_SESSION_HPP
#define MILO_CLI_SESSION_HPP

#include <memory>
#include <string>
#include <vector>

#include "core/diagnostic.hpp"
#include "core/vm/bytecode.hpp"
#include "runtime/interp.hpp"

namespace milo::cli {

/// What happened when a chunk of source was run.
enum class RunOutcome {
  Ok,
  ParseError,
  RuntimeError,
};

/// Which engine runs a program.
enum class Engine {
  /// The tree-walking evaluator: simple, and the reference.
  Evaluator,
  /// The bytecode machine: faster, and checked against the evaluator.
  Machine,
};

/// A session owns one interpreter, so that definitions made in one chunk
/// are still there in the next. This is what makes the REPL useful.
class Session {
 public:
  explicit Session(std::string filename);

  /// Choose the engine. The default is the evaluator.
  void setEngine(Engine engine) { engine_ = engine; }
  Engine engine() const { return engine_; }

  /// Parse and run a chunk of source.
  RunOutcome run(const std::string& source);

  /// Parse a chunk of source without running it.
  RunOutcome check(const std::string& source);

  const DiagnosticBag& diagnostics() const { return diagnostics_; }
  DiagnosticBag& diagnostics() { return diagnostics_; }

  /// The value of the last statement that was run, if it produced one.
  const runtime::Value& lastValue() const { return last_value_; }

  void clearDiagnostics() { diagnostics_.clear(); }

  Interpreter& interpreter() { return *interpreter_; }

  const std::string& filename() const { return filename_; }
  void setFilename(std::string filename) { filename_ = std::move(filename); }

 private:
  std::string filename_;
  std::unique_ptr<Interpreter> interpreter_;
  DiagnosticBag diagnostics_;
  runtime::Value last_value_;
  Engine engine_ = Engine::Evaluator;
};

}  // namespace milo::cli

#endif  // MILO_CLI_SESSION_HPP
