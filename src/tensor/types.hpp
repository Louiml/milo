// Core tensor vocabulary: element types, shapes, strides, devices.
#ifndef MILO_TENSOR_TYPES_HPP
#define MILO_TENSOR_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace milo::tensor {

/// The element type of a tensor.
enum class DType {
  F32,
  F16,
  BF16,
  F64,
  I8,
  I32,
  I64,
  U8,
  Bool,
};

/// The size of one element, in bytes.
std::size_t dtypeSize(DType dtype);

/// The canonical name, as it appears in source code.
const char* dtypeName(DType dtype);

/// Parse a dtype name. Returns false when the name is not a dtype.
bool dtypeFromName(std::string_view name, DType& out);

/// Is this a floating-point type?
bool dtypeIsFloat(DType dtype);

/// Is this an integer type?
bool dtypeIsInteger(DType dtype);

/// Is this a signed integer type?
bool dtypeIsSigned(DType dtype);

/// Where a tensor's memory lives.
enum class Device {
  Cpu,
  Cuda,
  Metal,
  Vulkan,
};

const char* deviceName(Device device);

/// A tensor's shape: a list of dimension sizes.
///
/// A dimension of -1 means "unknown" and is only legal in a type
/// annotation, never in a live tensor.
class Shape {
 public:
  Shape() = default;
  explicit Shape(std::vector<std::int64_t> dims) : dims_(std::move(dims)) {}

  /// The shape of a scalar: no dimensions, one element.
  static Shape scalar() { return Shape{}; }

  std::size_t rank() const { return dims_.size(); }
  bool isScalar() const { return dims_.empty(); }

  std::int64_t dim(std::size_t axis) const { return dims_[axis]; }
  const std::vector<std::int64_t>& dims() const { return dims_; }

  /// The total number of elements.
  std::int64_t numel() const;

  /// The number of bytes one element row of the last axis occupies.
  std::int64_t lastDimSize() const { return dims_.empty() ? 1 : dims_.back(); }

  bool operator==(const Shape& other) const { return dims_ == other.dims_; }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  /// "[2, 3]" — the way a shape is written in source code.
  std::string toString() const;

  /// "2x3" — the way a shape is written in a diagnostic.
  std::string toDisplayString() const;

  /// The shape with the dimensions in a different order.
  Shape permuted(const std::vector<std::size_t>& axes) const;

 private:
  std::vector<std::int64_t> dims_;
};

/// A tensor's strides, in elements rather than bytes.
///
/// A stride of zero means "this dimension is broadcast".
class Strides {
 public:
  Strides() = default;
  explicit Strides(std::vector<std::int64_t> values) : values_(std::move(values)) {}

  std::int64_t operator[](std::size_t axis) const { return values_[axis]; }
  std::size_t size() const { return values_.size(); }
  const std::vector<std::int64_t>& values() const { return values_; }

  bool operator==(const Strides& other) const { return values_ == other.values_; }

  /// Row-major strides for `shape`.
  static Strides contiguous(const Shape& shape);

 private:
  std::vector<std::int64_t> values_;
};

/// Multiply two shapes by broadcasting them against each other.
///
/// Throws `std::invalid_argument` when the shapes cannot be broadcast.
Shape broadcastShapes(const Shape& a, const Shape& b);

/// Check that `a` and `b` can be broadcast together.
bool canBroadcast(const Shape& a, const Shape& b);

}  // namespace milo::tensor

#endif  // MILO_TENSOR_TYPES_HPP
