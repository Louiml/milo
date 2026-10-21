#include "tensor/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "tensor/ops.hpp"

namespace milo::tensor {
namespace {

/// The NF4 codebook.
///
/// These are the sixteen values of the normal-float-4 codebook: the
/// quantiles of a standard normal distribution, scaled so that they lie
/// between -1 and 1.
constexpr float kNf4Codebook[16] = {
    -1.0f,           -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
    0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,   1.0f,
};

/// The index of the codebook entry nearest to `value`.
std::uint8_t nearestCode(float value) {
  std::uint8_t best = 0;
  float bestDistance = std::fabs(value - kNf4Codebook[0]);
  for (std::uint8_t i = 1; i < 16; ++i) {
    const float distance = std::fabs(value - kNf4Codebook[i]);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = i;
    }
  }
  return best;
}

}  // namespace

const float* nf4Codebook() { return kNf4Codebook; }

QuantizedTensor quantizeInt8(const Tensor& tensor) {
  if (tensor.shape().rank() == 0) {
    throw std::invalid_argument("a scalar cannot be quantised");
  }

  QuantizedTensor result;
  result.original = tensor.dtype();
  result.shape = tensor.shape();
  result.scheme = "int8";

  // One scale per row of the last axis: the output channel.
  const std::int64_t rows = tensor.numel() / tensor.shape().lastDimSize();
  const std::int64_t columns = tensor.shape().lastDimSize();

  result.codes.resize(static_cast<std::size_t>(tensor.numel()));
  result.scales.resize(static_cast<std::size_t>(rows));

  for (std::int64_t row = 0; row < rows; ++row) {
    double largest = 0.0;
    for (std::int64_t column = 0; column < columns; ++column) {
      largest = std::max(largest, std::fabs(tensor.getFlat(row * columns + column)));
    }
    const float scale = largest > 0.0 ? static_cast<float>(largest / 127.0) : 1.0f;
    result.scales[static_cast<std::size_t>(row)] = scale;

    for (std::int64_t column = 0; column < columns; ++column) {
      const double value = tensor.getFlat(row * columns + column);
      const double scaled = value / scale;
      const double clamped = std::clamp(scaled, -127.0, 127.0);
      const auto rounded = static_cast<std::int8_t>(std::llround(clamped));
      result.codes[static_cast<std::size_t>(row * columns + column)] =
          static_cast<std::byte>(rounded);
    }
  }

  return result;
}

QuantizedTensor quantizeNf4(const Tensor& tensor) {
  if (tensor.shape().rank() == 0) {
    throw std::invalid_argument("a scalar cannot be quantised");
  }

  QuantizedTensor result;
  result.original = tensor.dtype();
  result.shape = tensor.shape();
  result.scheme = "nf4";
  result.blockSize = kNf4BlockSize;

  const std::int64_t total = tensor.numel();
  const std::int64_t blocks = (total + kNf4BlockSize - 1) / kNf4BlockSize;

  // The codes are packed two to a byte.
  result.codes.resize(static_cast<std::size_t>((total + 1) / 2));
  result.scales.resize(static_cast<std::size_t>(blocks));
  result.absmax.resize(static_cast<std::size_t>(blocks));

  for (std::int64_t block = 0; block < blocks; ++block) {
    const std::int64_t start = block * kNf4BlockSize;
    const std::int64_t end = std::min(start + kNf4BlockSize, total);

    double largest = 0.0;
    for (std::int64_t i = start; i < end; ++i) {
      largest = std::max(largest, std::fabs(tensor.getFlat(i)));
    }
    const float absmax = static_cast<float>(largest);
    result.absmax[static_cast<std::size_t>(block)] = absmax;
    const float scale = absmax > 0.0f ? absmax : 1.0f;
    result.scales[static_cast<std::size_t>(block)] = scale;

    for (std::int64_t i = start; i < end; ++i) {
      const float normalised = static_cast<float>(tensor.getFlat(i)) / scale;
      const std::uint8_t code = nearestCode(normalised);
      const std::size_t index = static_cast<std::size_t>(i);
      if (index % 2 == 0) {
        // The low nibble.
        result.codes[index / 2] =
            static_cast<std::byte>(static_cast<std::uint8_t>(result.codes[index / 2]) |
                                   static_cast<std::uint8_t>(code & 0x0F));
      } else {
        // The high nibble.
        result.codes[index / 2] =
            static_cast<std::byte>(static_cast<std::uint8_t>(result.codes[index / 2]) |
                                   static_cast<std::uint8_t>((code & 0x0F) << 4));
      }
    }
  }

  return result;
}

namespace {

/// Read one NF4 code.
std::uint8_t codeAt(const QuantizedTensor& quantized, std::int64_t index) {
  const std::byte packed = quantized.codes[static_cast<std::size_t>(index / 2)];
  const auto raw = static_cast<std::uint8_t>(packed);
  return (index % 2 == 0) ? static_cast<std::uint8_t>(raw & 0x0F)
                          : static_cast<std::uint8_t>((raw >> 4) & 0x0F);
}

}  // namespace

Tensor dequantize(const QuantizedTensor& quantized) {
  Tensor result = Tensor::create(quantized.original, quantized.shape);
  const std::int64_t total = quantized.shape.numel();

  if (quantized.scheme == "int8") {
    const std::int64_t columns = quantized.shape.lastDimSize();
    const std::int64_t rows = total / columns;
    for (std::int64_t row = 0; row < rows; ++row) {
      const float scale = quantized.scales[static_cast<std::size_t>(row)];
      for (std::int64_t column = 0; column < columns; ++column) {
        const std::int64_t index = row * columns + column;
        const auto code = static_cast<std::int8_t>(quantized.codes[static_cast<std::size_t>(index)]);
        result.setFlat(index, static_cast<double>(code) * scale);
      }
    }
    return result;
  }

  // NF4.
  for (std::int64_t block = 0; block < static_cast<std::int64_t>(quantized.absmax.size()); ++block) {
    const std::int64_t start = block * quantized.blockSize;
    const std::int64_t end = std::min(start + quantized.blockSize, total);
    const float absmax = quantized.absmax[static_cast<std::size_t>(block)];
    for (std::int64_t i = start; i < end; ++i) {
      const std::uint8_t code = codeAt(quantized, i);
      result.setFlat(i, static_cast<double>(kNf4Codebook[code]) * absmax);
    }
  }
  return result;
}

Tensor matmulQuantized(const Tensor& x, const QuantizedTensor& weight) {
  if (weight.scheme != "nf4") {
    // The general path: dequantise the whole thing and use the ordinary
    // matrix product.
    return tensor::matmul(x, dequantize(weight));
  }

  if (weight.shape.rank() != 2 || x.shape().rank() != 2) {
    throw std::invalid_argument("the fused path works on two-dimensional tensors");
  }
  if (x.shape().dim(1) != weight.shape.dim(0)) {
    throw std::invalid_argument("the inner dimensions of these tensors do not match");
  }

  const std::int64_t m = x.shape().dim(0);
  const std::int64_t k = weight.shape.dim(0);
  const std::int64_t n = weight.shape.dim(1);

  Tensor result = Tensor::create(x.dtype(), Shape{std::vector<std::int64_t>{m, n}});

  // The weight is read one row at a time. The row is dequantised into a
  // small buffer, used, and thrown away: the whole dequantised matrix is
  // never in memory at once.
  std::vector<float> row(static_cast<std::size_t>(n), 0.0f);

  for (std::int64_t r = 0; r < m; ++r) {
    for (std::int64_t column = 0; column < n; ++column) {
      double total = 0.0;
      for (std::int64_t inner = 0; inner < k; ++inner) {
        // The element of the weight at [inner, column].
        const std::int64_t index = inner * n + column;
        const std::int64_t block = index / weight.blockSize;
        const float absmax = weight.absmax[static_cast<std::size_t>(block)];
        const std::uint8_t code = codeAt(weight, index);
        const float value = kNf4Codebook[code] * absmax;
        total += x.getFlat(r * k + inner) * value;
      }
      result.setFlat(r * n + column, total);
    }
  }

  return result;
}

double compressionRatio(const QuantizedTensor& quantized, const Tensor& original) {
  const std::size_t before = static_cast<std::size_t>(original.numel()) * dtypeSize(original.dtype());
  if (before == 0) return 1.0;
  return static_cast<double>(before) / static_cast<double>(quantized.byteSize());
}

}  // namespace milo::tensor
