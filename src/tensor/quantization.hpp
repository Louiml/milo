// Quantisation.
//
// Two schemes are supported:
//
//   * **int8**, symmetric, one scale per output channel. The weights are
//     kept in f32, the scales in f32, and the values in one byte each.
//   * **NF4**, one scale and one absolute maximum per block of 64 values.
//     The values are stored as four-bit codes from a fixed codebook, so
//     they take half a byte each.
//
// NF4 is the scheme QLoRA uses. It is small enough that a large model's
// weights fit in a modest amount of memory, and the loss of accuracy is
// small enough that the model still learns.
#ifndef MILO_TENSOR_QUANTIZATION_HPP
#define MILO_TENSOR_QUANTIZATION_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "tensor/tensor.hpp"
#include "tensor/types.hpp"

namespace milo::tensor {

/// The number of values in an NF4 block.
constexpr std::int64_t kNf4BlockSize = 64;

/// The NF4 codebook: sixteen values, from -1 to 1.
const float* nf4Codebook();

/// A quantised tensor.
struct QuantizedTensor {
  /// The element type the tensor had before it was quantised.
  DType original = DType::F32;
  /// The shape of the original tensor.
  Shape shape;
  /// `int8` or `nf4`.
  std::string scheme;
  /// The quantised values. For NF4 they are packed two to a byte.
  std::vector<std::byte> codes;
  /// One scale per channel (int8) or per block (NF4).
  std::vector<float> scales;
  /// The absolute maximum of each block (NF4 only).
  std::vector<float> absmax;
  /// The number of values in a block (NF4 only).
  std::int64_t blockSize = 0;

  /// How many bytes the quantised form takes.
  std::size_t byteSize() const { return codes.size() + scales.size() * 4 + absmax.size() * 4; }

  /// The shape the original tensor had.
  const Shape& originalShape() const { return shape; }
};

/// Quantise a tensor with int8, one scale per row of the last axis.
QuantizedTensor quantizeInt8(const Tensor& tensor);

/// Quantise a tensor with NF4, one scale per block of 64 values.
QuantizedTensor quantizeNf4(const Tensor& tensor);

/// Turn a quantised tensor back into a normal one.
Tensor dequantize(const QuantizedTensor& quantized);

/// Multiply a matrix by a quantised matrix without ever holding the
/// whole dequantised matrix in memory.
///
/// This is the operation that makes QLoRA possible: the weight is read
/// one row at a time, dequantised into a small buffer, multiplied, and
/// thrown away.
Tensor matmulQuantized(const Tensor& x, const QuantizedTensor& weight);

/// How much smaller the quantised form is than the original, as a ratio.
double compressionRatio(const QuantizedTensor& quantized, const Tensor& original);

}  // namespace milo::tensor

#endif  // MILO_TENSOR_QUANTIZATION_HPP
