// Tensor operations: elementwise maths, reductions, matmul, shape changes.
#ifndef MILO_TENSOR_OPS_HPP
#define MILO_TENSOR_OPS_HPP

#include <vector>

#include "tensor/tensor.hpp"
#include "tensor/types.hpp"

namespace milo::tensor {

/// An elementwise binary operation.
enum class BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Maximum,
  Minimum,
};

/// An elementwise unary operation.
enum class UnaryOp {
  Neg,
  Abs,
  Relu,
  Exp,
  Log,
  Sqrt,
  Tanh,
  Sigmoid,
  Gelu,
};

/// The result type of an operation on two types.
DType promoteDType(DType a, DType b);

/// Elementwise binary operation, with broadcasting.
Tensor binaryOp(BinaryOp op, const Tensor& a, const Tensor& b);

/// Elementwise binary operation against a scalar.
Tensor binaryOp(BinaryOp op, const Tensor& a, double scalar);

/// Elementwise unary operation.
Tensor unaryOp(UnaryOp op, const Tensor& a);

/// The sum of every element, as a scalar tensor.
Tensor sum(const Tensor& a);

/// The sum over one axis, with that axis removed.
Tensor sumOverAxis(const Tensor& a, std::int64_t axis);

/// The sum over several axes, with those axes removed.
Tensor sumOverAxes(const Tensor& a, const std::vector<std::int64_t>& axes);

/// The mean of every element, as a scalar tensor.
Tensor mean(const Tensor& a);

/// The largest element, as a scalar tensor.
Tensor max(const Tensor& a);

/// The smallest element, as a scalar tensor.
Tensor min(const Tensor& a);

/// The index of the largest element, as a scalar tensor.
Tensor argmax(const Tensor& a);

/// The matrix product of two tensors.
///
/// Supported shapes: `[m, k] x [k, n]`, `[k] x [k, n]`, and
/// `[m, k] x [k]`.
Tensor matmul(const Tensor& a, const Tensor& b);

/// The softmax over the last axis.
Tensor softmax(const Tensor& a);

/// A view of `a` with the axes reversed.
Tensor transpose(const Tensor& a);

/// A view of `a` with a new shape.
Tensor reshape(const Tensor& a, const Shape& shape);

// --- the operations that run on a device ------------------------------------
//
// The tensor engine calls these when a tensor is not on the host. They fall
// back to the host for anything the device cannot do, so the answer is
// always right even when it is not fast.

Tensor deviceBinaryOp(BinaryOp op, const Tensor& a, const Tensor& b);
Tensor deviceMatmul(const Tensor& a, const Tensor& b);
Tensor deviceSum(const Tensor& a);
Tensor deviceUnaryOp(UnaryOp op, const Tensor& a);

}  // namespace milo::tensor

#endif  // MILO_TENSOR_OPS_HPP
