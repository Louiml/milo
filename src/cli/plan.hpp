// The `milo plan` command: what a training run would need.
#ifndef MILO_CLI_PLAN_HPP
#define MILO_CLI_PLAN_HPP

#include <iosfwd>
#include <string>
#include <vector>

namespace milo::cli {

/// Print a memory plan for a model.
///
/// Returns 0 when the plan was produced, and 1 when the model could not
/// be read.
int planModel(const std::string& path, const std::vector<std::string>& options,
              std::ostream& out, std::ostream& err);

}  // namespace milo::cli

#endif  // MILO_CLI_PLAN_HPP
