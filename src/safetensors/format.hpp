// The safetensors format.
//
// A safetensors file is:
//
//   * 8 bytes: the length of the header, as a little-endian u64
//   * that many bytes: a JSON header
//   * the rest: the tensor data
//
// The header maps each tensor's name to its dtype, shape, and the byte
// range of its data, relative to the start of the data buffer.
#ifndef MILO_SAFETENSORS_FORMAT_HPP
#define MILO_SAFETENSORS_FORMAT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "tensor/tensor.hpp"
#include "tensor/types.hpp"

namespace milo::safetensors {

/// The largest header the format allows, in bytes.
constexpr std::uint64_t kMaxHeaderSize = 100'000'000;

/// Something went wrong reading or writing a safetensors file.
class Error : public std::exception {
 public:
  explicit Error(std::string message) : message_(std::move(message)) {}
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

/// What is known about one element type.
struct DTypeInfo {
  /// The name as it appears in the file, e.g. "BF16".
  const char* name;
  /// The size of one element, in bytes.
  std::size_t size;
  /// Whether Milo's tensor engine can hold this type.
  bool supported;
  /// The tensor type, when `supported` is true.
  tensor::DType type;
};

/// Look up an element type by its name in the file.
const DTypeInfo* findDType(const std::string& name);

/// The name of a tensor element type, as it appears in a file.
const char* dtypeFileName(tensor::DType dtype);

/// One tensor in the header.
struct TensorEntry {
  std::string name;
  /// The name as it appears in the file, e.g. "F32".
  std::string dtype_name;
  /// The element type, when the file's type is one Milo supports.
  tensor::DType dtype = tensor::DType::F32;
  /// Whether Milo's tensor engine can hold this type.
  bool supported = true;
  /// The shape, as written in the file.
  std::vector<std::int64_t> shape;
  /// The byte range of the data, relative to the start of the data buffer.
  std::uint64_t start = 0;
  std::uint64_t end = 0;

  /// The number of elements.
  std::uint64_t numel() const;
  /// The number of bytes the data occupies.
  std::uint64_t byteSize() const { return end - start; }
  /// The shape, as a Milo shape.
  tensor::Shape toShape() const;
};

/// The parsed header of a file.
struct Header {
  /// The tensors, in the order they appear in the file.
  std::vector<TensorEntry> tensors;
  /// The `__metadata__` entries, in the order they appear.
  std::vector<std::pair<std::string, std::string>> metadata;

  /// Find a tensor by name.
  const TensorEntry* find(const std::string& name) const;
};

/// Parse a header from its JSON text.
///
/// The text is checked thoroughly: every shape must be non-negative, every
/// byte range must be inside the buffer and must match the shape, and the
/// ranges must not overlap. A header that fails any of these is rejected.
Header parseHeader(const std::string& headerJson, std::uint64_t dataBufferSize);

/// Build the JSON header for a set of tensors.
///
/// The tensors are written in the order given, and their data ranges are
/// laid out one after another.
std::string buildHeader(const std::vector<TensorEntry>& tensors,
                        const std::vector<std::pair<std::string, std::string>>& metadata);

}  // namespace milo::safetensors

#endif  // MILO_SAFETENSORS_FORMAT_HPP
