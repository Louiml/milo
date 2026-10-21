#include "tensor/convert.hpp"

#include <cstring>

namespace milo::tensor {

float halfToFloat(std::uint16_t half) {
  const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
  const std::uint32_t exponent = (half >> 10) & 0x1Fu;
  const std::uint32_t mantissa = half & 0x3FFu;

  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;  // signed zero
    } else {
      // A subnormal half: renormalise it into a normal float.
      std::uint32_t m = mantissa;
      std::uint32_t shift = 0;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++shift;
      }
      m &= 0x3FFu;
      bits = sign | ((127u - 15u - shift) << 23) | (m << 13);
    }
  } else if (exponent == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);  // infinity or NaN
  } else {
    bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
  }

  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t floatToHalf(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const std::uint32_t rawExponent = (bits >> 23) & 0xFFu;
  std::uint32_t mantissa = bits & 0x7FFFFFu;

  if (rawExponent == 0xFFu) {
    if (mantissa == 0) return static_cast<std::uint16_t>(sign | 0x7C00u);  // infinity
    std::uint16_t payload = static_cast<std::uint16_t>(mantissa >> 13);
    if (payload == 0) payload = 1;  // keep NaN a NaN
    return static_cast<std::uint16_t>(sign | 0x7C00u | payload);
  }

  std::int32_t exponent = static_cast<std::int32_t>(rawExponent) - 127 + 15;

  if (exponent >= 0x1F) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow to infinity
  }

  if (exponent <= 0) {
    if (exponent < -10) return sign;  // underflow to zero
    mantissa |= 0x800000u;
    const std::int32_t shift = 14 - exponent;
    std::uint32_t subnormal = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (subnormal & 1u) != 0)) {
      ++subnormal;
    }
    return static_cast<std::uint16_t>(sign | subnormal);
  }

  std::uint32_t result = mantissa >> 13;
  const std::uint32_t remainder = mantissa & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0)) {
    ++result;
    if (result == 0x400u) {
      result = 0;
      ++exponent;
      if (exponent >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | result);
}

float bfloat16ToFloat(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float result = 0.0f;
  std::memcpy(&result, &wide, sizeof(result));
  return result;
}

std::uint16_t floatToBfloat16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  // Round to nearest, with ties to even.
  const std::uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
  bits += rounding;
  return static_cast<std::uint16_t>(bits >> 16);
}

double loadElement(const void* base, DType dtype, std::int64_t index) {
  const auto* bytes = static_cast<const std::byte*>(base);
  switch (dtype) {
    case DType::F32: {
      float value = 0.0f;
      std::memcpy(&value, bytes + index * 4, sizeof(value));
      return value;
    }
    case DType::F64: {
      double value = 0.0;
      std::memcpy(&value, bytes + index * 8, sizeof(value));
      return value;
    }
    case DType::F16: {
      std::uint16_t raw = 0;
      std::memcpy(&raw, bytes + index * 2, sizeof(raw));
      return halfToFloat(raw);
    }
    case DType::BF16: {
      std::uint16_t raw = 0;
      std::memcpy(&raw, bytes + index * 2, sizeof(raw));
      return bfloat16ToFloat(raw);
    }
    case DType::I8: {
      std::int8_t value = 0;
      std::memcpy(&value, bytes + index, sizeof(value));
      return value;
    }
    case DType::U8: {
      std::uint8_t value = 0;
      std::memcpy(&value, bytes + index, sizeof(value));
      return value;
    }
    case DType::I32: {
      std::int32_t value = 0;
      std::memcpy(&value, bytes + index * 4, sizeof(value));
      return value;
    }
    case DType::I64: {
      std::int64_t value = 0;
      std::memcpy(&value, bytes + index * 8, sizeof(value));
      return static_cast<double>(value);
    }
    case DType::Bool: {
      std::uint8_t value = 0;
      std::memcpy(&value, bytes + index, sizeof(value));
      return value != 0 ? 1.0 : 0.0;
    }
  }
  return 0.0;
}

void storeElement(void* base, DType dtype, std::int64_t index, double value) {
  auto* bytes = static_cast<std::byte*>(base);
  switch (dtype) {
    case DType::F32: {
      const float converted = static_cast<float>(value);
      std::memcpy(bytes + index * 4, &converted, sizeof(converted));
      return;
    }
    case DType::F64: {
      std::memcpy(bytes + index * 8, &value, sizeof(value));
      return;
    }
    case DType::F16: {
      const std::uint16_t converted = floatToHalf(static_cast<float>(value));
      std::memcpy(bytes + index * 2, &converted, sizeof(converted));
      return;
    }
    case DType::BF16: {
      const std::uint16_t converted = floatToBfloat16(static_cast<float>(value));
      std::memcpy(bytes + index * 2, &converted, sizeof(converted));
      return;
    }
    case DType::I8: {
      const std::int8_t converted = static_cast<std::int8_t>(value);
      std::memcpy(bytes + index, &converted, sizeof(converted));
      return;
    }
    case DType::U8: {
      const std::uint8_t converted = static_cast<std::uint8_t>(value);
      std::memcpy(bytes + index, &converted, sizeof(converted));
      return;
    }
    case DType::I32: {
      const std::int32_t converted = static_cast<std::int32_t>(value);
      std::memcpy(bytes + index * 4, &converted, sizeof(converted));
      return;
    }
    case DType::I64: {
      const std::int64_t converted = static_cast<std::int64_t>(value);
      std::memcpy(bytes + index * 8, &converted, sizeof(converted));
      return;
    }
    case DType::Bool: {
      const std::uint8_t converted = value != 0.0 ? 1 : 0;
      std::memcpy(bytes + index, &converted, sizeof(converted));
      return;
    }
  }
}

}  // namespace milo::tensor
