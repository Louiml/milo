#include "cli/inspect.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#include "safetensors/reader.hpp"

namespace milo::cli {
namespace {

/// Format a byte count the way a person would say it.
std::string humanBytes(std::uint64_t bytes) {
  static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < std::size(kUnits)) {
    value /= 1024.0;
    ++unit;
  }
  std::ostringstream out;
  if (unit == 0) {
    out << bytes << ' ' << kUnits[unit];
  } else {
    out << std::fixed << std::setprecision(value < 10.0 ? 2 : 1) << value << ' ' << kUnits[unit];
  }
  return out.str();
}

/// Format a number with thousands separators.
std::string withSeparators(std::uint64_t value) {
  std::string digits = std::to_string(value);
  std::string out;
  int count = 0;
  for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
    if (count != 0 && count % 3 == 0) out.push_back(',');
    out.push_back(*it);
    ++count;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

/// The shape of a tensor, the way a person would write it.
std::string shapeText(const safetensors::TensorEntry& entry) {
  if (entry.shape.empty()) return "scalar";
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < entry.shape.size(); ++i) {
    if (i != 0) out << ", ";
    out << entry.shape[i];
  }
  out << ']';
  return out.str();
}

}  // namespace

int inspectFile(const std::string& path, std::ostream& out, std::ostream& err) {
  std::shared_ptr<safetensors::File> file;
  try {
    file = std::make_shared<safetensors::File>(safetensors::File::open(path));
  } catch (const safetensors::Error& error) {
    err << "error: " << error.what() << '\n';
    return 1;
  }

  const safetensors::Header& header = file->header();

  std::uint64_t parameterCount = 0;
  std::uint64_t tensorBytes = 0;
  std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> byDtype;  // count, bytes
  std::map<std::string, std::uint64_t> byShape;

  for (const safetensors::TensorEntry& entry : header.tensors) {
    parameterCount += entry.numel();
    tensorBytes += entry.byteSize();
    auto& counts = byDtype[entry.dtype_name];
    counts.first += 1;
    counts.second += entry.byteSize();
    byShape[shapeText(entry)] += 1;
  }

  out << path << '\n';
  out << "  format       safetensors\n";
  out << "  file size    " << humanBytes(file->fileSize()) << " (" << withSeparators(file->fileSize())
      << " bytes)\n";
  out << "  tensor data  " << humanBytes(tensorBytes) << " (" << withSeparators(tensorBytes)
      << " bytes)\n";
  out << "  tensors      " << withSeparators(header.tensors.size()) << '\n';
  out << "  parameters   " << withSeparators(parameterCount) << '\n';
  if (!header.metadata.empty()) {
    out << "  metadata     " << withSeparators(header.metadata.size()) << " entries\n";
  }
  out << '\n';

  // The table of tensors.
  std::size_t nameWidth = 4;
  for (const safetensors::TensorEntry& entry : header.tensors) {
    nameWidth = std::max<std::size_t>(nameWidth, entry.name.size());
  }
  nameWidth = std::min<std::size_t>(nameWidth, 60);

  out << "  " << std::left << std::setw(static_cast<int>(nameWidth) + 2) << "name" << std::right
      << std::setw(7) << "dtype" << "  " << std::setw(18) << std::left << "shape" << std::right
      << std::setw(12) << "size" << '\n';

  for (const safetensors::TensorEntry& entry : header.tensors) {
    std::string name = entry.name;
    if (name.size() > nameWidth) {
      name = name.substr(0, nameWidth - 3) + "...";
    }
    out << "  " << std::left << std::setw(static_cast<int>(nameWidth) + 2) << name << std::right
        << std::setw(7) << entry.dtype_name << "  " << std::setw(18) << std::left
        << shapeText(entry) << std::right << std::setw(12) << humanBytes(entry.byteSize()) << '\n';
  }
  out << '\n';

  // The breakdown by dtype.
  if (!byDtype.empty()) {
    out << "  dtypes\n";
    for (const auto& entry : byDtype) {
      out << "    " << std::left << std::setw(6) << entry.first << std::right << std::setw(6)
          << entry.second.first << " tensors" << std::setw(6)
          << humanBytes(entry.second.second) << '\n';
    }
    out << '\n';
  }

  // The breakdown by shape.
  if (byShape.size() > 1) {
    out << "  shapes\n";
    for (const auto& entry : byShape) {
      out << "    " << std::left << std::setw(20) << entry.first << std::right << std::setw(6)
          << entry.second << " tensors\n";
    }
    out << '\n';
  }

  // The metadata.
  if (!header.metadata.empty()) {
    out << "  metadata\n";
    for (const auto& entry : header.metadata) {
      out << "    " << entry.first << ": " << entry.second << '\n';
    }
    out << '\n';
  }

  return 0;
}

}  // namespace milo::cli
