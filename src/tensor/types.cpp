#include "tensor/types.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace milo::tensor {
namespace {

struct DTypeInfo {
  DType dtype;
  std::size_t size;
  const char* name;
  bool is_float;
  bool is_integer;
  bool is_signed;
};

constexpr DTypeInfo kDTypes[] = {
    {DType::F32, 4, "f32", true, false, true},
    {DType::F16, 2, "f16", true, false, true},
    {DType::BF16, 2, "bf16", true, false, true},
    {DType::F64, 8, "f64", true, false, true},
    {DType::I8, 1, "i8", false, true, true},
    {DType::I32, 4, "i32", false, true, true},
    {DType::I64, 8, "i64", false, true, true},
    {DType::U8, 1, "u8", false, true, false},
    {DType::Bool, 1, "bool", false, true, false},
};

const DTypeInfo& info(DType dtype) {
  for (const DTypeInfo& entry : kDTypes) {
    if (entry.dtype == dtype) return entry;
  }
  return kDTypes[0];
}

}  // namespace

std::size_t dtypeSize(DType dtype) { return info(dtype).size; }

const char* dtypeName(DType dtype) { return info(dtype).name; }

bool dtypeIsFloat(DType dtype) { return info(dtype).is_float; }
bool dtypeIsInteger(DType dtype) { return info(dtype).is_integer; }
bool dtypeIsSigned(DType dtype) { return info(dtype).is_signed; }

bool dtypeFromName(std::string_view name, DType& out) {
  for (const DTypeInfo& entry : kDTypes) {
    if (name == entry.name) {
      out = entry.dtype;
      return true;
    }
  }
  return false;
}

const char* deviceName(Device device) {
  switch (device) {
    case Device::Cpu:
      return "cpu";
    case Device::Cuda:
      return "cuda";
    case Device::Metal:
      return "metal";
    case Device::Vulkan:
      return "vulkan";
  }
  return "cpu";
}

std::int64_t Shape::numel() const {
  std::int64_t total = 1;
  for (const std::int64_t dim : dims_) {
    total *= dim;
  }
  return total;
}

std::string Shape::toString() const {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < dims_.size(); ++i) {
    if (i != 0) out << ", ";
    out << dims_[i];
  }
  out << ']';
  return out.str();
}

std::string Shape::toDisplayString() const {
  if (dims_.empty()) return "scalar";
  std::ostringstream out;
  for (std::size_t i = 0; i < dims_.size(); ++i) {
    if (i != 0) out << 'x';
    out << dims_[i];
  }
  return out.str();
}

Shape Shape::permuted(const std::vector<std::size_t>& axes) const {
  if (axes.size() != dims_.size()) {
    throw std::invalid_argument("a permutation must name every axis");
  }
  Shape result;
  std::vector<std::int64_t> permuted;
  permuted.reserve(axes.size());
  for (const std::size_t axis : axes) {
    if (axis >= dims_.size()) {
      throw std::invalid_argument("a permutation names an axis that does not exist");
    }
    permuted.push_back(dims_[axis]);
  }
  return Shape(std::move(permuted));
}

Strides Strides::contiguous(const Shape& shape) {
  const std::size_t rank = shape.rank();
  std::vector<std::int64_t> strides(rank, 0);
  std::int64_t running = 1;
  for (std::size_t axis = rank; axis-- > 0;) {
    strides[axis] = running;
    running *= shape.dim(axis);
  }
  return Strides(std::move(strides));
}

bool canBroadcast(const Shape& a, const Shape& b) {
  const std::size_t rank = std::max(a.rank(), b.rank());
  for (std::size_t i = 0; i < rank; ++i) {
    const std::int64_t da = i < a.rank() ? a.dim(a.rank() - 1 - i) : 1;
    const std::int64_t db = i < b.rank() ? b.dim(b.rank() - 1 - i) : 1;
    if (da != db && da != 1 && db != 1) return false;
  }
  return true;
}

Shape broadcastShapes(const Shape& a, const Shape& b) {
  if (!canBroadcast(a, b)) {
    throw std::invalid_argument("these shapes cannot be broadcast together");
  }
  const std::size_t rank = std::max(a.rank(), b.rank());
  std::vector<std::int64_t> result(rank, 1);
  for (std::size_t i = 0; i < rank; ++i) {
    const std::int64_t da = i < a.rank() ? a.dim(a.rank() - 1 - i) : 1;
    const std::int64_t db = i < b.rank() ? b.dim(b.rank() - 1 - i) : 1;
    result[rank - 1 - i] = std::max(da, db);
  }
  return Shape(std::move(result));
}

}  // namespace milo::tensor
