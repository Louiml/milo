// Source locations and spans.
#ifndef MILO_CORE_SPAN_HPP
#define MILO_CORE_SPAN_HPP

#include <cstddef>
#include <string>

namespace milo {

/// A half-open range of source text, with a human-readable position.
///
/// `offset`/`length` are byte offsets into the source buffer. `line` and
/// `column` are 1-based and are what the diagnostic renderer shows.
struct SourceSpan {
  std::size_t offset = 0;
  std::size_t length = 0;
  int line = 0;
  int column = 0;

  SourceSpan() = default;
  SourceSpan(std::size_t off, std::size_t len, int ln, int col)
      : offset(off), length(len), line(ln), column(col) {}

  /// A zero-length span at the start of the file, used when a diagnostic
  /// has no better location.
  static SourceSpan start() { return SourceSpan{0, 0, 1, 1}; }

  /// The span covering `other` and this one.
  SourceSpan join(const SourceSpan& other) const {
    if (line == 0) return other;
    if (other.line == 0) return *this;
    const std::size_t end = offset + length;
    const std::size_t otherEnd = other.offset + other.length;
    const std::size_t newEnd = end > otherEnd ? end : otherEnd;
    // The line/column of the *start* is kept; the renderer re-derives the
    // end position from the source text.
    return SourceSpan{offset, newEnd - offset, line, column};
  }
};

}  // namespace milo

#endif  // MILO_CORE_SPAN_HPP
