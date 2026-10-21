// The `milo inspect` command: a report on a safetensors file.
#ifndef MILO_CLI_INSPECT_HPP
#define MILO_CLI_INSPECT_HPP

#include <iosfwd>
#include <string>

namespace milo::cli {

/// Print a report on a safetensors file.
///
/// Returns 0 when the file was read, and 1 when it could not be.
int inspectFile(const std::string& path, std::ostream& out, std::ostream& err);

}  // namespace milo::cli

#endif  // MILO_CLI_INSPECT_HPP
