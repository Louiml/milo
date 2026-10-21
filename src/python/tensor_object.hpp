// The Python type that represents a Milo tensor.
//
// It exposes the buffer protocol, so that NumPy can view a Milo's memory
// without copying, and `__dlpack__`, so that PyTorch can do the same.
#ifndef MILO_PYTHON_TENSOR_OBJECT_HPP
#define MILO_PYTHON_TENSOR_OBJECT_HPP

#include <memory>

#include "tensor/tensor.hpp"

namespace milo::python {

/// Create a Python object that holds `tensor`.
///
/// The result is a new reference; the caller owns it.
void* tensorObjectCreate(const std::shared_ptr<tensor::Tensor>& tensor);

/// Prepare the Python type that holds Milo's tensors.
///
/// This is done when the module starts, rather than the first time a
/// tensor is made, so that the type is in place before any Python code
/// can see it.
void tensorObjectPrepare();

/// The tensor inside a Python object, or nullptr.
std::shared_ptr<tensor::Tensor> tensorObjectGet(void* object);

}  // namespace milo::python

#endif  // MILO_PYTHON_TENSOR_OBJECT_HPP
