// The interactive read-eval-print loop.
#ifndef MILO_CLI_REPL_HPP
#define MILO_CLI_REPL_HPP

#include <iosfwd>
#include <string>

namespace milo::cli {

/// Run the REPL until the user leaves.
int runRepl(std::istream& in, std::ostream& out, std::ostream& notes);

/// Is this source obviously unfinished?
///
/// The REPL uses this to decide whether to keep reading lines. A chunk is
/// unfinished when a bracket is still open or when the last line ends in
/// a colon (which means a block is about to start).
bool sourceIsIncomplete(const std::string& source);

}  // namespace milo::cli

#endif  // MILO_CLI_REPL_HPP
