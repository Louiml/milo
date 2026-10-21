#include "safetensors/format.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "core/json.hpp"

namespace milo::safetensors {
namespace {

/// Every element type the format defines.
constexpr DTypeInfo kDTypes[] = {
    {"BOOL", 1, true, tensor::DType::Bool},  {"U8", 1, true, tensor::DType::U8},
    {"I8", 1, true, tensor::DType::I8},      {"I16", 2, false, tensor::DType::I32},
    {"U16", 2, false, tensor::DType::I32},   {"F16", 2, true, tensor::DType::F16},
    {"BF16", 2, true, tensor::DType::BF16},  {"I32", 4, true, tensor::DType::I32},
    {"U32", 4, false, tensor::DType::I64},   {"F32", 4, true, tensor::DType::F32},
    {"F64", 8, true, tensor::DType::F64},    {"I64", 8, true, tensor::DType::I64},
    {"U64", 8, false, tensor::DType::I64},
};

}  // namespace

const DTypeInfo* findDType(const std::string& name) {
  for (const DTypeInfo& info : kDTypes) {
    if (name == info.name) return &info;
  }
  return nullptr;
}

const char* dtypeFileName(tensor::DType dtype) {
  for (const DTypeInfo& info : kDTypes) {
    if (info.supported && info.type == dtype) return info.name;
  }
  return "F32";
}

std::uint64_t TensorEntry::numel() const {
  std::uint64_t total = 1;
  for (const std::int64_t dim : shape) {
    total *= static_cast<std::uint64_t>(dim);
  }
  return total;
}

tensor::Shape TensorEntry::toShape() const { return tensor::Shape(shape); }

const TensorEntry* Header::find(const std::string& name) const {
  for (const TensorEntry& entry : tensors) {
    if (entry.name == name) return &entry;
  }
  return nullptr;
}

Header parseHeader(const std::string& headerJson, std::uint64_t dataBufferSize) {
  json::Value document;
  json::ParseError error;
  if (!json::parse(headerJson, document, error)) {
    throw Error("the header is not valid JSON (" + error.toString() + ")");
  }
  if (!document.isObject()) {
    throw Error("the header is not a JSON object");
  }

  Header header;

  for (const auto& member : document.asObject()) {
    if (member.first == "__metadata__") {
      if (!member.second.isObject()) {
        throw Error("`__metadata__` is not a JSON object");
      }
      for (const auto& pair : member.second.asObject()) {
        if (!pair.second.isString()) {
          throw Error("the metadata entry '" + pair.first + "' is not a string");
        }
        header.metadata.emplace_back(pair.first, pair.second.asString());
      }
      continue;
    }

    const json::Value& value = member.second;
    if (!value.isObject()) {
      throw Error("the entry for '" + member.first + "' is not a JSON object");
    }

    TensorEntry entry;
    entry.name = member.first;

    const json::Value* dtype = value.find("dtype");
    if (dtype == nullptr || !dtype->isString()) {
      throw Error("the entry for '" + entry.name + "' has no dtype");
    }
    entry.dtype_name = dtype->asString();
    const DTypeInfo* info = findDType(entry.dtype_name);
    if (info == nullptr) {
      throw Error("the entry for '" + entry.name + "' has an unknown dtype '" +
                  entry.dtype_name + "'");
    }
    entry.supported = info->supported;
    entry.dtype = info->type;

    const json::Value* shape = value.find("shape");
    if (shape == nullptr || !shape->isArray()) {
      throw Error("the entry for '" + entry.name + "' has no shape");
    }
    for (const json::Value& dimension : shape->asArray()) {
      if (!dimension.isNumber()) {
        throw Error("the shape of '" + entry.name + "' is not made of numbers");
      }
      const double raw = dimension.asNumber();
      if (!std::isfinite(raw) || raw < 0.0 || raw != std::floor(raw)) {
        throw Error("the shape of '" + entry.name + "' has a size that is not a whole number");
      }
      if (raw > 9.2e18) {
        throw Error("the shape of '" + entry.name + "' is too large");
      }
      entry.shape.push_back(static_cast<std::int64_t>(raw));
    }

    const json::Value* offsets = value.find("data_offsets");
    if (offsets == nullptr || !offsets->isArray() || offsets->asArray().size() != 2) {
      throw Error("the entry for '" + entry.name + "' has no data_offsets");
    }
    std::uint64_t bounds[2] = {0, 0};
    std::size_t offsetIndex = 0;
    for (const json::Value& offset : offsets->asArray()) {
      if (!offset.isNumber()) {
        throw Error("the data_offsets of '" + entry.name + "' are not numbers");
      }
      const double raw = offset.asNumber();
      if (!std::isfinite(raw) || raw < 0.0 || raw != std::floor(raw)) {
        throw Error("the data_offsets of '" + entry.name + "' are not whole numbers");
      }
      if (raw > 1.8e19) {
        throw Error("the data_offsets of '" + entry.name + "' are too large");
      }
      bounds[offsetIndex++] = static_cast<std::uint64_t>(raw);
    }
    entry.start = bounds[0];
    entry.end = bounds[1];

    if (entry.start > entry.end) {
      throw Error("the data_offsets of '" + entry.name + "' run backwards");
    }
    if (entry.end > dataBufferSize) {
      throw Error("the entry for '" + entry.name + "' points past the end of the file");
    }

    // The number of bytes must match the shape exactly.
    std::uint64_t elements = 1;
    for (const std::int64_t dim : entry.shape) {
      if (dim != 0 && elements > (~0ULL) / static_cast<std::uint64_t>(dim)) {
        throw Error("the shape of '" + entry.name + "' is too large");
      }
      elements *= static_cast<std::uint64_t>(dim);
    }
    if (elements > (~0ULL) / info->size) {
      throw Error("the shape of '" + entry.name + "' is too large");
    }
    const std::uint64_t expected = elements * info->size;
    if (expected != entry.byteSize()) {
      throw Error("the entry for '" + entry.name + "' says it holds " +
                  std::to_string(expected) + " bytes but its data range holds " +
                  std::to_string(entry.byteSize()));
    }

    header.tensors.push_back(std::move(entry));
  }

  // The data ranges must not overlap, and they must be laid out in order.
  std::vector<const TensorEntry*> ordered;
  ordered.reserve(header.tensors.size());
  for (const TensorEntry& entry : header.tensors) {
    ordered.push_back(&entry);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const TensorEntry* a, const TensorEntry* b) { return a->start < b->start; });

  std::uint64_t previousEnd = 0;
  for (const TensorEntry* entry : ordered) {
    if (entry->start < previousEnd) {
      throw Error("the data of '" + entry->name + "' overlaps another tensor's data");
    }
    previousEnd = entry->end;
  }

  return header;
}

std::string buildHeader(const std::vector<TensorEntry>& tensors,
                        const std::vector<std::pair<std::string, std::string>>& metadata) {
  std::ostringstream out;
  out << '{';

  bool first = true;
  if (!metadata.empty()) {
    out << "\"__metadata__\":{";
    for (std::size_t i = 0; i < metadata.size(); ++i) {
      if (i != 0) out << ',';
      out << json::escape(metadata[i].first) << ':' << json::escape(metadata[i].second);
    }
    out << '}';
    first = false;
  }

  for (const TensorEntry& entry : tensors) {
    if (!first) out << ',';
    first = false;
    out << json::escape(entry.name) << ":{";
    out << "\"dtype\":\"" << entry.dtype_name << "\",";
    out << "\"shape\":[";
    for (std::size_t i = 0; i < entry.shape.size(); ++i) {
      if (i != 0) out << ',';
      out << entry.shape[i];
    }
    out << "],";
    out << "\"data_offsets\":[" << entry.start << ',' << entry.end << ']';
    out << '}';
  }

  out << '}';
  return out.str();
}

}  // namespace milo::safetensors
