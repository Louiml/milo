#include "tensor/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "tensor/convert.hpp"
#include "tensor/kernels.hpp"

namespace milo::tensor {
namespace {

double applyBinary(BinaryOp op, double a, double b) {
  switch (op) {
    case BinaryOp::Add:
      return a + b;
    case BinaryOp::Sub:
      return a - b;
    case BinaryOp::Mul:
      return a * b;
    case BinaryOp::Div:
      return a / b;
    case BinaryOp::Pow:
      return std::pow(a, b);
    case BinaryOp::Maximum:
      return a > b ? a : b;
    case BinaryOp::Minimum:
      return a < b ? a : b;
  }
  return 0.0;
}

double applyUnary(UnaryOp op, double x) {
  switch (op) {
    case UnaryOp::Neg:
      return -x;
    case UnaryOp::Abs:
      return std::fabs(x);
    case UnaryOp::Relu:
      return x > 0.0 ? x : 0.0;
    case UnaryOp::Exp:
      return std::exp(x);
    case UnaryOp::Log:
      return std::log(x);
    case UnaryOp::Sqrt:
      return std::sqrt(x);
    case UnaryOp::Tanh:
      return std::tanh(x);
    case UnaryOp::Sigmoid:
      return 1.0 / (1.0 + std::exp(-x));
    case UnaryOp::Gelu: {
      // The tanh approximation of the Gaussian error linear unit.
      const double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
      return 0.5 * x * (1.0 + std::tanh(inner));
    }
  }
  return x;
}

/// The per-output-axis strides for a broadcast operand.
std::vector<std::int64_t> broadcastStrides(const Shape& out, const Tensor& operand) {
  std::vector<std::int64_t> strides(out.rank(), 0);
  const std::size_t rank = operand.shape().rank();
  for (std::size_t i = 0; i < rank; ++i) {
    const std::size_t outAxis = out.rank() - rank + i;
    // A dimension of size one is broadcast: its stride becomes zero.
    if (operand.shape().dim(i) != 1) {
      strides[outAxis] = operand.strides()[i];
    }
  }
  return strides;
}

/// The float pointer for a contiguous f32 tensor.
const float* f32Data(const Tensor& tensor) {
  return reinterpret_cast<const float*>(static_cast<const std::byte*>(tensor.storage()->data()) +
                                       tensor.byteOffset());
}

float* f32DataMutable(Tensor& tensor) {
  return reinterpret_cast<float*>(static_cast<std::byte*>(tensor.storage()->data()) +
                                  tensor.byteOffset());
}

}  // namespace

DType promoteDType(DType a, DType b) {
  if (a == b) return a;
  // Float beats integer; wider beats narrower.
  const auto rank = [](DType dtype) -> int {
    switch (dtype) {
      case DType::Bool:
        return 0;
      case DType::U8:
      case DType::I8:
        return 1;
      case DType::I32:
        return 2;
      case DType::I64:
        return 3;
      case DType::BF16:
      case DType::F16:
        return 4;
      case DType::F32:
        return 5;
      case DType::F64:
        return 6;
    }
    return 0;
  };
  return rank(a) >= rank(b) ? a : b;
}

Tensor binaryOp(BinaryOp op, const Tensor& a, const Tensor& b) {
  if (a.storage()->device() != Device::Cpu || b.storage()->device() != Device::Cpu) {
    return deviceBinaryOp(op, a, b);
  }

  const Shape outShape = broadcastShapes(a.shape(), b.shape());
  const DType outDType = promoteDType(a.dtype(), b.dtype());
  Tensor out = Tensor::create(outDType, outShape);

  const std::int64_t total = out.numel();

  // The fast path: two contiguous f32 tensors of the same shape.
  if (a.dtype() == DType::F32 && b.dtype() == DType::F32 && outDType == DType::F32 &&
      a.isContiguous() && b.isContiguous() && a.shape() == b.shape() &&
      a.shape() == outShape && total > 0) {
    const auto* pa = f32Data(a);
    const auto* pb = f32Data(b);
    auto* po = f32DataMutable(out);
    const auto count = static_cast<std::size_t>(total);
    switch (op) {
      case BinaryOp::Add:
        kernels::table().addF32(pa, pb, po, count);
        break;
      case BinaryOp::Sub:
        kernels::table().subF32(pa, pb, po, count);
        break;
      case BinaryOp::Mul:
        kernels::table().mulF32(pa, pb, po, count);
        break;
      case BinaryOp::Div:
        kernels::table().divF32(pa, pb, po, count);
        break;
      default:
        for (std::size_t i = 0; i < count; ++i) {
          po[i] = static_cast<float>(applyBinary(op, pa[i], pb[i]));
        }
        break;
    }
    return out;
  }

  // The general path: strided operands and broadcasting.
  const std::vector<std::int64_t> aStrides = broadcastStrides(outShape, a);
  const std::vector<std::int64_t> bStrides = broadcastStrides(outShape, b);

  const auto* aBase =
      static_cast<const std::byte*>(a.storage()->data()) + a.byteOffset();
  const auto* bBase =
      static_cast<const std::byte*>(b.storage()->data()) + b.byteOffset();
  auto* outBase = static_cast<std::byte*>(out.storage()->data());

  std::vector<std::int64_t> indices(outShape.rank(), 0);
  for (std::int64_t i = 0; i < total; ++i) {
    std::int64_t aOffset = 0;
    std::int64_t bOffset = 0;
    for (std::size_t axis = 0; axis < outShape.rank(); ++axis) {
      aOffset += indices[axis] * aStrides[axis];
      bOffset += indices[axis] * bStrides[axis];
    }
    const double av = loadElement(aBase, a.dtype(), aOffset);
    const double bv = loadElement(bBase, b.dtype(), bOffset);
    storeElement(outBase, outDType, i, applyBinary(op, av, bv));

    for (std::size_t axis = outShape.rank(); axis-- > 0;) {
      if (++indices[axis] < outShape.dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return out;
}
Tensor binaryOp(BinaryOp op, const Tensor& a, double scalar) {
  const DType outDType = a.dtype();
  Tensor out = Tensor::create(outDType, a.shape());
  const std::int64_t total = a.numel();

  if (a.dtype() == DType::F32 && a.isContiguous() && total > 0) {
    const auto* pa = f32Data(a);
    auto* po = f32DataMutable(out);
    const auto count = static_cast<std::size_t>(total);
    switch (op) {
      case BinaryOp::Mul:
        kernels::table().scaleF32(pa, static_cast<float>(scalar), po, count);
        break;
      case BinaryOp::Add: {
        std::vector<float> broadcast(count, static_cast<float>(scalar));
        kernels::table().addF32(pa, broadcast.data(), po, count);
        break;
      }
      case BinaryOp::Sub: {
        std::vector<float> broadcast(count, static_cast<float>(scalar));
        kernels::table().subF32(pa, broadcast.data(), po, count);
        break;
      }
      default:
        for (std::size_t i = 0; i < count; ++i) {
          po[i] = static_cast<float>(applyBinary(op, pa[i], scalar));
        }
        break;
    }
    return out;
  }

  const auto* aBase = static_cast<const std::byte*>(a.storage()->data()) + a.byteOffset();
  auto* outBase = static_cast<std::byte*>(out.storage()->data());
  std::vector<std::int64_t> indices(a.shape().rank(), 0);
  for (std::int64_t i = 0; i < total; ++i) {
    std::int64_t offset = 0;
    for (std::size_t axis = 0; axis < a.shape().rank(); ++axis) {
      offset += indices[axis] * a.strides()[axis];
    }
    storeElement(outBase, outDType, i, applyBinary(op, loadElement(aBase, a.dtype(), offset), scalar));
    for (std::size_t axis = a.shape().rank(); axis-- > 0;) {
      if (++indices[axis] < a.shape().dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return out;
}

Tensor unaryOp(UnaryOp op, const Tensor& a) {
  if (a.storage()->device() != Device::Cpu) {
    return deviceUnaryOp(op, a);
  }

  Tensor out = Tensor::create(a.dtype(), a.shape());
  const std::int64_t total = a.numel();
  const auto* aBase = static_cast<const std::byte*>(a.storage()->data()) + a.byteOffset();
  auto* outBase = static_cast<std::byte*>(out.storage()->data());

  std::vector<std::int64_t> indices(a.shape().rank(), 0);
  for (std::int64_t i = 0; i < total; ++i) {
    std::int64_t offset = 0;
    for (std::size_t axis = 0; axis < a.shape().rank(); ++axis) {
      offset += indices[axis] * a.strides()[axis];
    }
    storeElement(outBase, a.dtype(), i, applyUnary(op, loadElement(aBase, a.dtype(), offset)));
    for (std::size_t axis = a.shape().rank(); axis-- > 0;) {
      if (++indices[axis] < a.shape().dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return out;
}

namespace {

/// Reduce every element to one number.
double reduceAll(const Tensor& a, BinaryOp op, bool useMinimum) {
  const std::int64_t total = a.numel();
  if (total == 0) {
    throw std::invalid_argument("this tensor is empty");
  }
  const auto* base = static_cast<const std::byte*>(a.storage()->data()) + a.byteOffset();

  if (a.isContiguous()) {
    double accumulator = loadElement(base, a.dtype(), 0);
    for (std::int64_t i = 1; i < total; ++i) {
      const double value = loadElement(base, a.dtype(), i);
      if (useMinimum) {
        accumulator = value < accumulator ? value : accumulator;
      } else if (op == BinaryOp::Maximum) {
        accumulator = value > accumulator ? value : accumulator;
      } else {
        accumulator = applyBinary(op, accumulator, value);
      }
    }
    return accumulator;
  }

  std::vector<std::int64_t> indices(a.shape().rank(), 0);
  double accumulator = loadElement(base, a.dtype(), 0);
  for (std::int64_t i = 1; i < total; ++i) {
    // Move to the next element before reading it: the accumulator already
    // holds the first one.
    for (std::size_t axis = a.shape().rank(); axis-- > 0;) {
      if (++indices[axis] < a.shape().dim(axis)) break;
      indices[axis] = 0;
    }
    std::int64_t offset = 0;
    for (std::size_t axis = 0; axis < a.shape().rank(); ++axis) {
      offset += indices[axis] * a.strides()[axis];
    }
    const double value = loadElement(base, a.dtype(), offset);
    if (useMinimum) {
      accumulator = value < accumulator ? value : accumulator;
    } else if (op == BinaryOp::Maximum) {
      accumulator = value > accumulator ? value : accumulator;
    } else {
      accumulator = applyBinary(op, accumulator, value);
    }
  }
  return accumulator;
}

}  // namespace

Tensor sum(const Tensor& a) {
  if (a.storage()->device() != Device::Cpu) {
    return deviceSum(a);
  }
  return Tensor::full(a.dtype(), Shape::scalar(), reduceAll(a, BinaryOp::Add, false));
}

Tensor sumOverAxis(const Tensor& a, std::int64_t axis) {
  return sumOverAxes(a, {axis});
}

Tensor sumOverAxes(const Tensor& a, const std::vector<std::int64_t>& axes) {
  if (a.shape().rank() == 0) {
    throw std::invalid_argument("a scalar has no axes to reduce");
  }

  std::vector<bool> reduced(a.shape().rank(), false);
  for (const std::int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<std::int64_t>(a.shape().rank())) {
      throw std::invalid_argument("this axis does not exist");
    }
    reduced[static_cast<std::size_t>(axis)] = true;
  }

  // The output keeps the axes that were not reduced.
  std::vector<std::int64_t> outDims;
  std::vector<std::size_t> kept;
  for (std::size_t axis = 0; axis < a.shape().rank(); ++axis) {
    if (!reduced[axis]) {
      kept.push_back(axis);
      outDims.push_back(a.shape().dim(axis));
    }
  }

  Tensor out = Tensor::zeros(a.dtype(), Shape(outDims));

  // Walk the input once, adding each element into the right place.
  const std::int64_t total = a.numel();
  std::vector<std::int64_t> indices(a.shape().rank(), 0);
  std::vector<std::int64_t> outIndices(kept.size(), 0);
  for (std::int64_t i = 0; i < total; ++i) {
    for (std::size_t k = 0; k < kept.size(); ++k) {
      outIndices[k] = indices[kept[k]];
    }
    const double value = a.get(indices);
    out.set(outIndices, out.get(outIndices) + value);

    for (std::size_t axis = a.shape().rank(); axis-- > 0;) {
      if (++indices[axis] < a.shape().dim(axis)) break;
      indices[axis] = 0;
    }
  }

  return out;
}

Tensor mean(const Tensor& a) {
  const double total = reduceAll(a, BinaryOp::Add, false);
  return Tensor::full(a.dtype(), Shape::scalar(), total / static_cast<double>(a.numel()));
}

Tensor max(const Tensor& a) {
  return Tensor::full(a.dtype(), Shape::scalar(), reduceAll(a, BinaryOp::Maximum, false));
}

Tensor min(const Tensor& a) {
  return Tensor::full(a.dtype(), Shape::scalar(), reduceAll(a, BinaryOp::Maximum, true));
}

Tensor argmax(const Tensor& a) {
  const std::int64_t total = a.numel();
  if (total == 0) {
    throw std::invalid_argument("this tensor is empty");
  }
  std::int64_t best = 0;
  double bestValue = 0.0;
  std::vector<std::int64_t> indices(a.shape().rank(), 0);
  for (std::int64_t i = 0; i < total; ++i) {
    const double value = a.get(indices);
    if (i == 0 || value > bestValue) {
      bestValue = value;
      best = i;
    }
    for (std::size_t axis = a.shape().rank(); axis-- > 0;) {
      if (++indices[axis] < a.shape().dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return Tensor::full(DType::I64, Shape::scalar(), static_cast<double>(best));
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  if (a.storage()->device() != Device::Cpu || b.storage()->device() != Device::Cpu) {
    return deviceMatmul(a, b);
  }

  const bool aIsVector = a.shape().rank() == 1;
  const bool bIsVector = b.shape().rank() == 1;
  if (a.shape().rank() > 2 || b.shape().rank() > 2) {
    throw std::invalid_argument("matmul works on one- and two-dimensional tensors");
  }
  if (a.shape().rank() == 0 || b.shape().rank() == 0) {
    throw std::invalid_argument("matmul does not work on a scalar");
  }

  const std::int64_t m = aIsVector ? 1 : a.shape().dim(0);
  const std::int64_t k = aIsVector ? a.shape().dim(0) : a.shape().dim(1);
  const std::int64_t k2 = bIsVector ? b.shape().dim(0) : b.shape().dim(0);
  const std::int64_t n = bIsVector ? 1 : b.shape().dim(1);

  if (k != k2) {
    throw std::invalid_argument("the inner dimensions of these tensors do not match");
  }

  const DType outDType = promoteDType(a.dtype(), b.dtype());

  std::vector<std::int64_t> outDims;
  if (aIsVector && bIsVector) {
    outDims = {};
  } else if (aIsVector) {
    outDims = {n};
  } else if (bIsVector) {
    outDims = {m};
  } else {
    outDims = {m, n};
  }
  Tensor out = Tensor::create(outDType, Shape(outDims));

  // Work in f32 when both inputs are f32; otherwise use the general path.
  if (a.dtype() == DType::F32 && b.dtype() == DType::F32) {
    const Tensor aContiguous = a.contiguous();
    const Tensor bContiguous = b.contiguous();
    const auto* pa = f32Data(aContiguous);
    const auto* pb = f32Data(bContiguous);
    auto* po = f32DataMutable(out);

    if (aIsVector && bIsVector) {
      po[0] = kernels::table().dotF32(pa, pb, static_cast<std::size_t>(k));
      return out;
    }
    if (aIsVector) {
      // [k] x [k, n] -> [n]
      for (std::int64_t column = 0; column < n; ++column) {
        float total = 0.0f;
        for (std::int64_t row = 0; row < k; ++row) {
          total += pa[row] * pb[row * n + column];
        }
        po[column] = total;
      }
      return out;
    }
    if (bIsVector) {
      // [m, k] x [k] -> [m]
      for (std::int64_t row = 0; row < m; ++row) {
        float total = 0.0f;
        for (std::int64_t column = 0; column < k; ++column) {
          total += pa[row * k + column] * pb[column];
        }
        po[row] = total;
      }
      return out;
    }

    // [m, k] x [k, n] -> [m, n], using the ikj formulation so that the
    // inner loop runs over contiguous memory.
    for (std::int64_t row = 0; row < m; ++row) {
      float* outRow = po + row * n;
      std::fill(outRow, outRow + n, 0.0f);
      for (std::int64_t inner = 0; inner < k; ++inner) {
        const float scale = pa[row * k + inner];
        if (scale == 0.0f) continue;
        kernels::table().axpyF32(scale, pb + inner * n, outRow, static_cast<std::size_t>(n));
      }
    }
    return out;
  }

  // The general path, for other element types.
  if (aIsVector && bIsVector) {
    double total = 0.0;
    for (std::int64_t i = 0; i < k; ++i) {
      total += a.get({i}) * b.get({i});
    }
    return Tensor::full(outDType, Shape::scalar(), total);
  }
  if (aIsVector) {
    Tensor out2 = Tensor::create(outDType, Shape{std::vector<std::int64_t>{n}});
    for (std::int64_t column = 0; column < n; ++column) {
      double total = 0.0;
      for (std::int64_t row = 0; row < k; ++row) {
        total += a.get({row}) * b.get({row, column});
      }
      out2.setFlat(column, total);
    }
    return out2;
  }
  if (bIsVector) {
    Tensor out2 = Tensor::create(outDType, Shape{std::vector<std::int64_t>{m}});
    for (std::int64_t row = 0; row < m; ++row) {
      double total = 0.0;
      for (std::int64_t column = 0; column < k; ++column) {
        total += a.get({row, column}) * b.get({column});
      }
      out2.setFlat(row, total);
    }
    return out2;
  }

  Tensor out2 = Tensor::create(outDType, Shape{std::vector<std::int64_t>{m, n}});
  for (std::int64_t row = 0; row < m; ++row) {
    for (std::int64_t column = 0; column < n; ++column) {
      double total = 0.0;
      for (std::int64_t inner = 0; inner < k; ++inner) {
        total += a.get({row, inner}) * b.get({inner, column});
      }
      out2.set({row, column}, total);
    }
  }
  return out2;
}

Tensor softmax(const Tensor& a) {
  if (a.shape().rank() == 0) {
    throw std::invalid_argument("softmax needs at least one dimension");
  }
  const std::int64_t rows = a.shape().numel() / a.shape().lastDimSize();
  const std::int64_t columns = a.shape().lastDimSize();

  Tensor out = Tensor::create(a.dtype(), a.shape());
  const Tensor source = a.contiguous();
  const auto* base = static_cast<const std::byte*>(source.storage()->data()) + source.byteOffset();
  auto* outBase = static_cast<std::byte*>(out.storage()->data());

  for (std::int64_t row = 0; row < rows; ++row) {
    // Subtract the maximum first, so that exp() cannot overflow.
    double largest = loadElement(base, a.dtype(), row * columns);
    for (std::int64_t column = 1; column < columns; ++column) {
      const double value = loadElement(base, a.dtype(), row * columns + column);
      if (value > largest) largest = value;
    }
    double total = 0.0;
    for (std::int64_t column = 0; column < columns; ++column) {
      const double value = std::exp(loadElement(base, a.dtype(), row * columns + column) - largest);
      storeElement(outBase, a.dtype(), row * columns + column, value);
      total += value;
    }
    for (std::int64_t column = 0; column < columns; ++column) {
      const double value = loadElement(outBase, a.dtype(), row * columns + column);
      storeElement(outBase, a.dtype(), row * columns + column, value / total);
    }
  }
  return out;
}

Tensor transpose(const Tensor& a) { return a.transpose(); }

Tensor reshape(const Tensor& a, const Shape& shape) { return a.reshape(shape); }

}  // namespace milo::tensor
