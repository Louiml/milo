// Element conversion: reading and writing tensor data of any dtype.
#ifndef MILO_TENSOR_CONVERT_HPP
#define MILO_TENSOR_CONVERT_HPP

#include <cstddef>
#include <cstdint>

#include "tensor/types.hpp"

namespace milo::tensor {

/// IEEE-754 binary16 to binary32.
float halfToFloat(std::uint16_t half);

/// binary32 to IEEE-754 binary16, rounded to nearest with ties to even.
std::uint16_t floatToHalf(float value);

/// bfloat16 to binary32. bfloat16 is the top half of a binary32.
float bfloat16ToFloat(std::uint16_t bits);

/// binary32 to bfloat16, rounded to nearest with ties to even.
std::uint16_t floatToBfloat16(float value);

/// Read one element as a double. `index` counts elements, not bytes.
double loadElement(const void* base, DType dtype, std::int64_t index);

/// Write one element from a double.
void storeElement(void* base, DType dtype, std::int64_t index, double value);

}  // namespace milo::tensor

#endif  // MILO_TENSOR_CONVERT_HPP
