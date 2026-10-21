#include "core/diagnostic.hpp"

#include <algorithm>
#include <sstream>

namespace milo {
namespace {

/// Extract the 1-based line `line_number` from `source`.
std::string lineAt(const std::string& source, int line_number) {
  if (line_number <= 0) return {};
  int current = 1;
  std::size_t start = 0;
  while (current < line_number) {
    const std::size_t nl = source.find('\n', start);
    if (nl == std::string::npos) return {};
    start = nl + 1;
    ++current;
  }
  const std::size_t end = source.find('\n', start);
  if (end == std::string::npos) return source.substr(start);
  return source.substr(start, end - start);
}

/// The width of the line-number gutter, e.g. 3 for line 100.
int gutterWidth(int line_number) {
  int digits = 1;
  int n = line_number;
  while (n >= 10) {
    n /= 10;
    ++digits;
  }
  return digits;
}

/// Expand tabs so that caret alignment is correct in a terminal.
std::string expandTabs(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  int column = 0;
  for (const char c : text) {
    if (c == '\t') {
      const int width = 8 - (column % 8);
      out.append(static_cast<std::size_t>(width), ' ');
      column += width;
    } else {
      out.push_back(c);
      ++column;
    }
  }
  return out;
}

}  // namespace

const char* to_string(Severity severity) {
  switch (severity) {
    case Severity::Error:
      return "error";
    case Severity::Warning:
      return "warning";
    case Severity::Note:
      return "note";
    case Severity::Help:
      return "help";
  }
  return "error";
}

bool DiagnosticBag::hasErrors() const { return errorCount() > 0; }

std::size_t DiagnosticBag::errorCount() const {
  return static_cast<std::size_t>(
      std::count_if(diagnostics_.begin(), diagnostics_.end(),
                    [](const Diagnostic& d) { return d.severity == Severity::Error; }));
}

std::string DiagnosticBag::render(const std::string& source, const std::string& filename) const {
  std::ostringstream out;
  for (const Diagnostic& diagnostic : diagnostics_) {
    out << renderOne(diagnostic, source, filename);
    out << '\n';
  }
  return out.str();
}

std::string DiagnosticBag::renderOne(const Diagnostic& diagnostic, const std::string& source,
                                     const std::string& filename) {
  std::ostringstream out;

  out << to_string(diagnostic.severity) << ": " << diagnostic.message << '\n';

  if (diagnostic.span.line <= 0) {
    // No useful location; the message alone is the whole story.
    if (!diagnostic.help.empty()) {
      out << "  help: " << diagnostic.help << '\n';
    }
    return out.str();
  }

  const int gutter = gutterWidth(diagnostic.span.line);
  const std::string pad(static_cast<std::size_t>(gutter), ' ');

  out << pad << "--> " << filename << ':' << diagnostic.span.line << ':' << diagnostic.span.column
      << '\n';

  const std::string rawLine = lineAt(source, diagnostic.span.line);
  const std::string shown = expandTabs(rawLine);

  out << pad << " | " << shown << '\n';
  out << pad << " | ";

  // Column is 1-based and counts bytes; the displayed line may have had
  // tabs expanded, so recompute the visual column.
  int visualColumn = 1;
  for (int i = 1; i < diagnostic.span.column && i <= static_cast<int>(rawLine.size()); ++i) {
    if (rawLine[static_cast<std::size_t>(i - 1)] == '\t') {
      visualColumn += 8 - ((visualColumn - 1) % 8);
    } else {
      ++visualColumn;
    }
  }
  out << std::string(static_cast<std::size_t>(visualColumn - 1), ' ');

  const std::size_t length = std::max<std::size_t>(1, diagnostic.span.length);
  out << std::string(length, '^');

  if (!diagnostic.label.empty()) {
    out << ' ' << diagnostic.label;
  }
  out << '\n';

  if (!diagnostic.help.empty()) {
    out << pad << " = help: " << diagnostic.help << '\n';
  }

  return out.str();
}

namespace {

/// Levenshtein edit distance, with an early exit for long inputs.
std::size_t editDistance(const std::string& a, const std::string& b) {
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  if (n == 0) return m;
  if (m == 0) return n;

  std::vector<std::size_t> previous(m + 1);
  std::vector<std::size_t> current(m + 1);
  for (std::size_t j = 0; j <= m; ++j) previous[j] = j;

  for (std::size_t i = 1; i <= n; ++i) {
    current[0] = i;
    for (std::size_t j = 1; j <= m; ++j) {
      const std::size_t substitution = previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
      const std::size_t insertion = current[j - 1] + 1;
      const std::size_t deletion = previous[j] + 1;
      current[j] = std::min({substitution, insertion, deletion});
    }
    previous.swap(current);
  }
  return previous[m];
}

}  // namespace

std::string closestMatch(const std::string& word, const std::vector<std::string>& candidates) {
  const std::size_t limit = word.size() <= 4 ? 1 : 2;
  std::string best;
  std::size_t bestDistance = limit + 1;

  for (const std::string& candidate : candidates) {
    if (candidate == word) return candidate;
    const std::size_t distance = editDistance(word, candidate);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  }

  if (bestDistance > limit) return {};
  return best;
}

}  // namespace milo