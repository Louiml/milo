// The `milo` command-line tool.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cli/inspect.hpp"
#include "cli/plan.hpp"
#include "cli/repl.hpp"
#include "cli/session.hpp"
#include "milo/milo.h"

namespace {

const char* kUsage = R"(milo — a programming language for AI and data work

Usage:
  milo run <file.milo>     run a program
  milo check <file.milo>   parse a program and report problems, without running it
  milo inspect <file>      report on a safetensors file
  milo plan <model>        work out what a training run would need
  milo repl                start the interactive prompt
  milo version             print version information
  milo help                show this message

Options:
  -e, --eval <source>      run a snippet given on the command line
  --vm                     run with the bytecode machine instead of the evaluator
)";

/// Read a whole file into a string.
bool readFile(const std::string& path, std::string& out, std::string& error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    error = "I could not open '" + path + "'";
    return false;
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  out = buffer.str();
  return true;
}

int runFile(const std::string& path, bool checkOnly, bool useMachine) {
  std::string source;
  std::string error;
  if (!readFile(path, source, error)) {
    std::cerr << "error: " << error << '\n';
    return 1;
  }

  milo::cli::Session session(path);
  if (useMachine) {
    session.setEngine(milo::cli::Engine::Machine);
  }
  const milo::cli::RunOutcome outcome = checkOnly ? session.check(source) : session.run(source);
  if (outcome != milo::cli::RunOutcome::Ok) {
    std::cerr << session.diagnostics().render(source, path);
    return 1;
  }
  return 0;
}

int runSnippet(const std::string& source) {
  milo::cli::Session session("<eval>");
  const milo::cli::RunOutcome outcome = session.run(source);
  if (outcome != milo::cli::RunOutcome::Ok) {
    std::cerr << session.diagnostics().render(source, session.filename());
    return 1;
  }
  if (!session.lastValue().isNull()) {
    std::cout << session.lastValue().toReprString() << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

  if (args.empty()) {
    std::cout << kUsage;
    return 0;
  }

  const std::string command = args.front();
  const std::vector<std::string> rest(args.begin() + 1, args.end());

  // Look for -e/--eval anywhere in the arguments, including first.
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "-e" || args[i] == "--eval") {
      if (i + 1 >= args.size()) {
        std::cerr << "error: " << args[i] << " needs some source to run\n";
        return 2;
      }
      return runSnippet(args[i + 1]);
    }
  }

  if (command == "help" || command == "--help" || command == "-h") {
    std::cout << kUsage;
    return 0;
  }

  if (command == "version" || command == "--version" || command == "-V") {
    std::cout << "milo " << MILO_RUNTIME_VERSION << '\n';
    std::cout << "language " << MILO_LANGUAGE_VERSION << '\n';
    std::cout << "abi " << MILO_ABI_VERSION << '\n';
    return 0;
  }

  if (command == "repl") {
    return milo::cli::runRepl(std::cin, std::cout, std::cerr);
  }

  if (command == "inspect") {
    if (rest.empty()) {
      std::cerr << "error: inspect needs a file name\n";
      return 2;
    }
    return milo::cli::inspectFile(rest.front(), std::cout, std::cerr);
  }

  if (command == "plan") {
    if (rest.empty()) {
      std::cerr << "error: plan needs a model file\n";
      return 2;
    }
    // The first argument that is not an option is the model.
    std::string model;
    std::vector<std::string> options;
    for (std::size_t i = 0; i < rest.size(); ++i) {
      if (!rest[i].empty() && rest[i][0] == '-') {
        options.push_back(rest[i]);
        if (i + 1 < rest.size() && (rest[i + 1].empty() || rest[i + 1][0] != '-')) {
          options.push_back(rest[++i]);
        }
      } else if (model.empty()) {
        model = rest[i];
      }
    }
    if (model.empty()) {
      std::cerr << "error: plan needs a model file\n";
      return 2;
    }
    return milo::cli::planModel(model, options, std::cout, std::cerr);
  }

  if (command == "run" || command == "check") {
    // The engine can be chosen: the bytecode machine, or the tree-walking
    // evaluator that checks it.
    bool useMachine = false;
    std::vector<std::string> files;
    for (const std::string& argument : rest) {
      if (argument == "--vm") {
        useMachine = true;
      } else {
        files.push_back(argument);
      }
    }
    if (files.empty()) {
      std::cerr << "error: " << command << " needs a file name\n";
      return 2;
    }
    return runFile(files.front(), command == "check", useMachine);
  }

  std::cerr << "error: I do not know the command '" << command << "'\n\n" << kUsage;
  return 2;
}
