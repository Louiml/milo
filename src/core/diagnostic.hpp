// Diagnostics: severity, message, and how to render them.
#ifndef MILO_CORE_DIAGNOSTIC_HPP
#define MILO_CORE_DIAGNOSTIC_HPP

#include <string>
#include <vector>

#include "core/span.hpp"

namespace milo {

enum class Severity {
  Error,
  Warning,
  Note,
  Help,
};

const char* to_string(Severity severity);

/// One diagnostic message.
///
/// A diagnostic carries a primary span (where the problem is) and an
/// optional label (a short hint such as "did you mean `chats`?"). Later
/// phases add secondary spans for "this was defined here" notes.
struct Diagnostic {
  Severity severity = Severity::Error;
  std::string message;
  SourceSpan span;
  /// Optional inline hint rendered under the squiggle.
  std::string label;
  /// Optional "help:" line rendered below the snippet.
  std::string help;

  Diagnostic() = default;
  Diagnostic(Severity sev, std::string msg, SourceSpan sp)
      : severity(sev), message(std::move(msg)), span(sp) {}
};

/// A collection of diagnostics produced by one compilation.
class DiagnosticBag {
 public:
  void add(Diagnostic diagnostic) { diagnostics_.push_back(std::move(diagnostic)); }

  void error(std::string message, SourceSpan span) {
    add(Diagnostic(Severity::Error, std::move(message), span));
  }

  void warning(std::string message, SourceSpan span) {
    add(Diagnostic(Severity::Warning, std::move(message), span));
  }

  void note(std::string message, SourceSpan span) {
    add(Diagnostic(Severity::Note, std::move(message), span));
  }

  bool hasErrors() const;
  std::size_t errorCount() const;

  const std::vector<Diagnostic>& diagnostics() const { return diagnostics_; }
  std::vector<Diagnostic>& diagnostics() { return diagnostics_; }

  void clear() { diagnostics_.clear(); }

  /// Render every diagnostic in the bag.
  ///
  /// `source` is the full text of the file, `filename` its name. The
  /// renderer draws the offending line, a caret run under the primary span,
  /// and the label. It is deliberately plain: no box drawing, no colour
  /// codes, so the output is readable in a terminal, a log file, and a CI
  /// transcript alike.
  std::string render(const std::string& source, const std::string& filename) const;

  /// Render a single diagnostic.
  static std::string renderOne(const Diagnostic& diagnostic, const std::string& source,
                               const std::string& filename);

 private:
  std::vector<Diagnostic> diagnostics_;
};

/// The candidate closest to `word`, or an empty string when nothing is
/// close enough. Used for "did you mean ...?" hints.
std::string closestMatch(const std::string& word, const std::vector<std::string>& candidates);

}  // namespace milo

#endif  // MILO_CORE_DIAGNOSTIC_HPP
