#include "python/tensor_object.hpp"

#include <Python.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "python/dlpack.h"

namespace milo::python {
namespace {

/// The Python type that holds a Milo tensor.
struct TensorObject {
  PyObject_HEAD
  /// The tensor. Keeping it here means the memory stays alive as long as
  /// the Python object does.
  std::shared_ptr<tensor::Tensor>* tensor;
};

/// The buffer protocol's format string for a Milo's element type.
const char* bufferFormat(tensor::DType dtype) {
  switch (dtype) {
    case tensor::DType::F32:
      return "f";
    case tensor::DType::F64:
      return "d";
    case tensor::DType::I64:
      return "q";
    case tensor::DType::I32:
      return "i";
    case tensor::DType::I8:
      return "b";
    case tensor::DType::U8:
      return "B";
    case tensor::DType::Bool:
      return "?";
    default:
      return nullptr;
  }
}

/// The DLPack type code for a Milo's element type.
bool dlpackType(tensor::DType dtype, uint8_t& code, uint8_t& bits) {
  switch (dtype) {
    case tensor::DType::F32:
      code = 2;
      bits = 32;
      return true;
    case tensor::DType::F64:
      code = 2;
      bits = 64;
      return true;
    case tensor::DType::F16:
      code = 2;
      bits = 16;
      return true;
    case tensor::DType::BF16:
      code = 4;
      bits = 16;
      return true;
    case tensor::DType::I64:
      code = 0;
      bits = 64;
      return true;
    case tensor::DType::I32:
      code = 0;
      bits = 32;
      return true;
    case tensor::DType::I8:
      code = 0;
      bits = 8;
      return true;
    case tensor::DType::U8:
      code = 1;
      bits = 8;
      return true;
    case tensor::DType::Bool:
      code = 6;
      bits = 8;
      return true;
  }
  return false;
}

/// The view of a tensor's memory, as the buffer protocol wants it.
int tensorGetBuffer(PyObject* object, Py_buffer* view, int flags) {
  if (view == nullptr) {
    PyErr_SetString(PyExc_BufferError, "a view is required");
    return -1;
  }
  if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
    // The tensor's memory may be on a device or read-only.
    const TensorObject* self = reinterpret_cast<TensorObject*>(object);
    if (self->tensor == nullptr || !self->tensor->get()) {
      PyErr_SetString(PyExc_BufferError, "this tensor has no memory");
      return -1;
    }
    if (self->tensor->get()->storage()->readOnly()) {
      PyErr_SetString(PyExc_BufferError, "this tensor is read-only");
      return -1;
    }
  }

  const TensorObject* self = reinterpret_cast<TensorObject*>(object);
  if (self->tensor == nullptr || !self->tensor->get()) {
    PyErr_SetString(PyExc_BufferError, "this tensor has no memory");
    return -1;
  }
  const tensor::Tensor& tensor = *self->tensor->get();

  const char* format = bufferFormat(tensor.dtype());
  if (format == nullptr) {
    PyErr_SetString(PyExc_BufferError, "this element type has no buffer format");
    return -1;
  }

  view->buf = const_cast<std::byte*>(static_cast<const std::byte*>(tensor.storage()->data()) +
                                    tensor.byteOffset());
  view->obj = object;
  Py_INCREF(object);
  view->len = static_cast<Py_ssize_t>(tensor.numel() * tensor::dtypeSize(tensor.dtype()));
  view->readonly = tensor.storage()->readOnly() ? 1 : 0;
  view->itemsize = static_cast<Py_ssize_t>(tensor::dtypeSize(tensor.dtype()));
  view->format = const_cast<char*>(format);
  view->ndim = static_cast<int>(tensor.shape().rank());
  view->shape = nullptr;
  view->strides = nullptr;
  view->suboffsets = nullptr;
  view->internal = nullptr;

  if (view->ndim > 0) {
    view->shape = static_cast<Py_ssize_t*>(PyMem_Malloc(sizeof(Py_ssize_t) * view->ndim));
    view->strides =
        static_cast<Py_ssize_t*>(PyMem_Malloc(sizeof(Py_ssize_t) * view->ndim));
    if (view->shape == nullptr || view->strides == nullptr) {
      PyMem_Free(view->shape);
      PyMem_Free(view->strides);
      PyErr_NoMemory();
      return -1;
    }
    const std::size_t elementSize = tensor::dtypeSize(tensor.dtype());
    for (std::size_t axis = 0; axis < tensor.shape().rank(); ++axis) {
      view->shape[axis] = static_cast<Py_ssize_t>(tensor.shape().dim(axis));
      view->strides[axis] =
          static_cast<Py_ssize_t>(tensor.strides()[axis] * static_cast<std::int64_t>(elementSize));
    }
  }
  return 0;
}

void tensorReleaseBuffer(PyObject* object, Py_buffer* view) {
  (void)object;
  if (view == nullptr) return;
  // The shape and the strides were made for this view, so they go with
  // it. The reference to the object itself belongs to `PyBuffer_Release`,
  // which gives it back after this call.
  PyMem_Free(view->shape);
  PyMem_Free(view->strides);
  view->shape = nullptr;
  view->strides = nullptr;
}

/// The data behind a DLPack dlpack.
struct DlpackExport {
  DLManagedTensor managed;
  std::shared_ptr<tensor::Tensor> tensor;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

void dlpackDeleter(DLManagedTensor* self) {
  auto* dlpack = static_cast<DlpackExport*>(self->manager_ctx);
  delete dlpack;
}
PyObject* tensorDlpack(PyObject* object, PyObject* args, PyObject* keywords) {
  (void)args;
  (void)keywords;
  const TensorObject* self = reinterpret_cast<TensorObject*>(object);
  if (self->tensor == nullptr || !self->tensor->get()) {
    PyErr_SetString(PyExc_RuntimeError, "this tensor has no memory");
    return nullptr;
  }
  const tensor::Tensor& tensor = *self->tensor->get();

  auto* dlpack = new DlpackExport();
  dlpack->tensor = *self->tensor;
  dlpack->shape = tensor.shape().dims();
  // DLPack counts strides in elements, not in bytes.
  dlpack->strides.reserve(tensor.shape().rank());
  for (std::size_t axis = 0; axis < tensor.shape().rank(); ++axis) {
    dlpack->strides.push_back(tensor.strides()[axis]);
  }

  uint8_t code = 0;
  uint8_t bits = 0;
  if (!dlpackType(tensor.dtype(), code, bits)) {
    delete dlpack;
    PyErr_SetString(PyExc_TypeError, "this element type cannot be shared with DLPack");
    return nullptr;
  }

  dlpack->managed.dl_tensor.data = const_cast<std::byte*>(
      static_cast<const std::byte*>(tensor.storage()->data()) + tensor.byteOffset());
  dlpack->managed.dl_tensor.device.device_type = 1;  // kDLCPU
  dlpack->managed.dl_tensor.device.device_id = 0;
  dlpack->managed.dl_tensor.ndim = static_cast<int32_t>(dlpack->shape.size());
  dlpack->managed.dl_tensor.dtype.code = code;
  dlpack->managed.dl_tensor.dtype.bits = bits;
  dlpack->managed.dl_tensor.dtype.lanes = 1;
  dlpack->managed.dl_tensor.shape = dlpack->shape.data();
  // DLPack counts strides in elements, and it wants them even for a
  // contiguous tensor.
  dlpack->managed.dl_tensor.strides = dlpack->strides.data();
  dlpack->managed.dl_tensor.byte_offset = 0;
  dlpack->managed.manager_ctx = dlpack;
  dlpack->managed.deleter = &dlpackDeleter;

  PyObject* capsule = PyCapsule_New(&dlpack->managed, MILO_DLPACK_CAPSULE_NAME, nullptr);
  if (capsule == nullptr) {
    delete dlpack;
    return nullptr;
  }
  return capsule;
}

/// The device the tensor lives on, as DLPack wants it: (type, index).
///
/// The type is 1 for a CPU device, which is what DLPack calls kDLCPU.
PyObject* tensorDlpackDevice(PyObject* object, PyObject* /*args*/) {
  const TensorObject* self = reinterpret_cast<TensorObject*>(object);
  if (self->tensor == nullptr || !self->tensor->get()) {
    Py_RETURN_NONE;
  }
  return Py_BuildValue("(ii)", 1, 0);
}

PyObject* tensorGetShape(PyObject* object, void*) {
  const TensorObject* self = reinterpret_cast<TensorObject*>(object);
  if (self->tensor == nullptr || !self->tensor->get()) {
    return PyTuple_New(0);
  }
  const tensor::Shape& shape = self->tensor->get()->shape();
  PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(shape.rank()));
  if (tuple == nullptr) return nullptr;
  for (std::size_t axis = 0; axis < shape.rank(); ++axis) {
    PyObject* dimension = PyLong_FromLongLong(shape.dim(axis));
    if (dimension == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(axis), dimension);
  }
  return tuple;
}

PyObject* tensorGetDtype(PyObject* object, void*) {
  const TensorObject* self = reinterpret_cast<TensorObject*>(object);
  if (self->tensor == nullptr || !self->tensor->get()) {
    return PyUnicode_FromString("unknown");
  }
  return PyUnicode_FromString(tensor::dtypeName(self->tensor->get()->dtype()));
}

PyObject* tensorGetNdim(PyObject* object, void*) {
  const TensorObject* self = reinterpret_cast<TensorObject*>(object);
  if (self->tensor == nullptr || !self->tensor->get()) {
    return PyLong_FromLong(0);
  }
  return PyLong_FromLong(static_cast<long>(self->tensor->get()->shape().rank()));
}

PyObject* tensorRepr(PyObject* object) {
  const TensorObject* self = reinterpret_cast<TensorObject*>(object);
  if (self->tensor == nullptr || !self->tensor->get()) {
    return PyUnicode_FromString("<tensor (empty)>");
  }
  const tensor::Tensor& tensor = *self->tensor->get();
  return PyUnicode_FromFormat("<tensor %s %s>", tensor::dtypeName(tensor.dtype()),
                              tensor.shape().toString().c_str());
}

void tensorDealloc(PyObject* object) {
  TensorObject* self = reinterpret_cast<TensorObject*>(object);
  delete self->tensor;
  self->tensor = nullptr;
  Py_TYPE(object)->tp_free(object);
}

PyBufferProcs gBufferProcs = {};
PyGetSetDef gGetSet[] = {
    {const_cast<char*>("shape"), &tensorGetShape, nullptr, nullptr, nullptr},
    {const_cast<char*>("dtype"), &tensorGetDtype, nullptr, nullptr, nullptr},
    {const_cast<char*>("ndim"), &tensorGetNdim, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef gMethods[] = {
    {const_cast<char*>("__dlpack__"), reinterpret_cast<PyCFunction>(&tensorDlpack),
     METH_VARARGS | METH_KEYWORDS, nullptr},
    {const_cast<char*>("__dlpack_device__"), &tensorDlpackDevice, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyType_Slot gSlots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&tensorDealloc)},
    {Py_tp_repr, reinterpret_cast<void*>(&tensorRepr)},
    {Py_tp_methods, reinterpret_cast<void*>(gMethods)},
    {Py_tp_getset, reinterpret_cast<void*>(gGetSet)},
    {Py_bf_getbuffer, reinterpret_cast<void*>(&tensorGetBuffer)},
    {Py_bf_releasebuffer, reinterpret_cast<void*>(&tensorReleaseBuffer)},
    {0, nullptr},
};

PyType_Spec gTypeSpec = {
    "milo.MiloTensor",
    static_cast<int>(sizeof(TensorObject)),
    0,
    Py_TPFLAGS_DEFAULT,
    gSlots,
};

PyTypeObject* gTensorType = nullptr;

}  // namespace

void tensorObjectPrepare() {
  if (gTensorType != nullptr) return;
  PyObject* type = PyType_FromSpec(&gTypeSpec);
  if (type == nullptr) {
    throw std::runtime_error("python: the tensor type could not be prepared");
  }
  gTensorType = reinterpret_cast<PyTypeObject*>(type);
}

void* tensorObjectCreate(const std::shared_ptr<tensor::Tensor>& tensor) {
  if (gTensorType == nullptr) {
    tensorObjectPrepare();
  }

  PyObject* object = gTensorType->tp_alloc(gTensorType, 0);
  if (object == nullptr) return nullptr;
  auto* self = reinterpret_cast<TensorObject*>(object);
  self->tensor = new std::shared_ptr<tensor::Tensor>(tensor);
  return object;
}

std::shared_ptr<tensor::Tensor> tensorObjectGet(void* object) {
  if (object == nullptr) return nullptr;
  if (gTensorType == nullptr) return nullptr;
  if (!PyObject_TypeCheck(object, gTensorType)) return nullptr;
  const auto* self = reinterpret_cast<TensorObject*>(object);
  if (self->tensor == nullptr) return nullptr;
  return *self->tensor;
}

}  // namespace milo::python
