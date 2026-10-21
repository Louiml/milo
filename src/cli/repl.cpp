#include "cli/repl.hpp"

#include <iostream>
#include <sstream>
#include <string>

#include "cli/session.hpp"
#include "core/lexer.hpp"
#include "core/parser.hpp"
#include "milo/milo.h"

namespace milo::cli {
namespace {

/// The last line of `source` that has something on it.
std::string lastMeaningfulLine(const std::string& source) {
  std::istringstream stream(source);
  std::string line;
  std::string last;
  while (std::getline(stream, line)) {
    bool blank = true;
    for (const char c : line) {
      if (c != ' ' && c != '\t' && c != '\r' && c != '#') {
        blank = false;
        break;
      }
    }
    if (!blank) last = line;
  }
  return last;
}

}  // namespace

bool sourceIsIncomplete(const std::string& source) {
  // An open bracket or an unterminated string means more input is needed.
  Lexer lexer(source, "<repl>");
  lexer.tokenize();
  if (lexer.diagnostics().hasErrors()) {
    // A lexer error at the very end usually means the chunk stops in the
    // middle of something. Treat it as incomplete so the user can carry on.
    return true;
  }

  const std::string last = lastMeaningfulLine(source);
  if (last.empty()) return false;

  // A trailing colon means an indented block is coming.
  std::size_t end = last.size();
  while (end > 0 && (last[end - 1] == ' ' || last[end - 1] == '\t' || last[end - 1] == '\r')) {
    --end;
  }
  if (end > 0 && last[end - 1] == ':') return true;

  return false;
}

int runRepl(std::istream& in, std::ostream& out, std::ostream& notes) {
  Session session("<repl>");
  session.interpreter().setOutput(&out);
  session.interpreter().setNotes(&notes);

  out << "Milo " << MILO_RUNTIME_VERSION << " (language " << MILO_LANGUAGE_VERSION << ")\n";
  out << "Type an expression to see its value, or `help` for what you can do.\n";
  out << "Press Ctrl-D (or Ctrl-Z on Windows) to leave.\n\n";

  std::string buffer;
  bool running = true;

  while (running) {
    out << (buffer.empty() ? "milo> " : "...  ");
    out.flush();

    std::string line;
    if (!std::getline(in, line)) {
      out << '\n';
      break;
    }

    const std::string trimmed = [&] {
      std::size_t begin = line.find_first_not_of(" \t");
      if (begin == std::string::npos) return std::string();
      std::size_t end = line.find_last_not_of(" \t\r");
      return line.substr(begin, end - begin + 1);
    }();

    if (buffer.empty()) {
      if (trimmed.empty()) continue;
      if (trimmed == "exit" || trimmed == "quit") break;
      if (trimmed == "help") {
        out << "Commands:\n"
            << "  help          show this message\n"
            << "  exit          leave the REPL\n"
            << "\n"
            << "Everything else is Milo source. Definitions persist between lines.\n";
        continue;
      }
      if (trimmed == "clear") {
        out << "The REPL keeps its definitions; use `exit` and start again to reset.\n";
        continue;
      }
    }

    buffer += line;
    buffer += '\n';

    if (sourceIsIncomplete(buffer)) continue;

    const RunOutcome outcome = session.run(buffer);
    if (outcome != RunOutcome::Ok) {
      notes << session.diagnostics().render(buffer, session.filename());
      notes.flush();
    } else if (!session.lastValue().isNull()) {
      out << session.lastValue().toReprString() << '\n';
      out.flush();
    }
    buffer.clear();
  }

  return 0;
}

}  // namespace milo::cli
