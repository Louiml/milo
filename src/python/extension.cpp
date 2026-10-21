// The `milo` Python extension module.
//
// It lets Python call Milo: run a program, load a model, and move tensors
// between the two languages without copying.
//
//     import milo
//     value = milo.run("let x = 21\nx * 2")
//     tensor = milo.load("model.safetensors", "weight")
//     import numpy as np
//     array = np.asarray(tensor)      # a view, not a copy
//     import torch
//     torch_tensor = torch.from_dlpack(tensor)
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "milo/milo.h"
#include "python/dlpack.h"
#include "python/python.hpp"
#include "python/tensor_object.hpp"
#include "safetensors/reader.hpp"
#include "tensor/tensor.hpp"

namespace {

/// The runtime a Milo program runs in.
///
/// Each run gets its own runtime, so that one call cannot see — or trip
/// over — the names another call declared.
struct Runtime {
  milo_runtime* handle = nullptr;

  Runtime() { handle = milo_runtime_create(); }
  ~Runtime() {
    if (handle != nullptr) milo_runtime_destroy(handle);
  }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
};

/// Raise a Python error with the last Milo error attached.
void raiseMiloError(const char* what) {
  char buffer[1024] = {0};
  milo_last_error(buffer, sizeof(buffer));
  PyErr_Format(PyExc_RuntimeError, "milo: %s: %s", what, buffer);
}

/// A Milo tensor, seen by Python without copying.
PyObject* tensorToPython(milo_value* value);

/// Turn a Milo value into a Python object.
///
/// The value handle is consumed: it is released here, or handed over to
/// the Python object when the value owns memory that Python will see.
PyObject* toPython(milo_value* value) {
  switch (milo_value_kind_of(value)) {
    case MILO_VALUE_NULL:
      milo_value_release(value);
      Py_RETURN_NONE;
    case MILO_VALUE_BOOL: {
      const int truth = milo_value_as_bool(value);
      milo_value_release(value);
      if (truth) {
        Py_RETURN_TRUE;
      }
      Py_RETURN_FALSE;
    }
    case MILO_VALUE_INT: {
      const std::int64_t number = milo_value_as_int(value);
      milo_value_release(value);
      return PyLong_FromLongLong(static_cast<Py_ssize_t>(number));
    }
    case MILO_VALUE_FLOAT: {
      const double number = milo_value_as_float(value);
      milo_value_release(value);
      return PyFloat_FromDouble(number);
    }
    case MILO_VALUE_STRING: {
      const char* text = milo_value_as_string(value);
      const std::size_t length = milo_value_string_length(value);
      PyObject* result = PyUnicode_FromStringAndSize(text != nullptr ? text : "",
                                                     static_cast<Py_ssize_t>(length));
      milo_value_release(value);
      return result;
    }
    case MILO_VALUE_TENSOR:
      return tensorToPython(value);
    default:
      milo_value_release(value);
      PyErr_SetString(PyExc_TypeError, "milo: this value cannot be shown in Python");
      return nullptr;
  }
}

/// The Milo element type behind a C ABI dtype.
milo::tensor::DType elementType(milo_dtype dtype) {
  switch (dtype) {
    case MILO_DTYPE_F32: return milo::tensor::DType::F32;
    case MILO_DTYPE_F16: return milo::tensor::DType::F16;
    case MILO_DTYPE_BF16: return milo::tensor::DType::BF16;
    case MILO_DTYPE_F64: return milo::tensor::DType::F64;
    case MILO_DTYPE_I8: return milo::tensor::DType::I8;
    case MILO_DTYPE_I32: return milo::tensor::DType::I32;
    case MILO_DTYPE_I64: return milo::tensor::DType::I64;
    case MILO_DTYPE_U8: return milo::tensor::DType::U8;
    case MILO_DTYPE_BOOL: return milo::tensor::DType::Bool;
  }
  return milo::tensor::DType::F32;
}

/// A Milo tensor, seen by Python without copying.
///
/// The Python object holds the value handle, so the memory stays alive for
/// as long as Python can see it.
PyObject* tensorToPython(milo_value* value) {
  const int rank = milo_value_tensor_rank(value);
  const void* data = milo_value_tensor_data(value);
  const std::size_t bytes = milo_value_tensor_bytes(value);

  if (data == nullptr) {
    milo_value_release(value);
    PyErr_SetString(PyExc_ValueError, "milo: this tensor has no memory");
    return nullptr;
  }

  std::vector<std::int64_t> dims;
  dims.reserve(static_cast<std::size_t>(rank));
  for (int axis = 0; axis < rank; ++axis) {
    dims.push_back(milo_value_tensor_dim(value, axis));
  }
  const milo::tensor::Shape shape(dims);
  const milo::tensor::DType dtype = elementType(milo_value_tensor_dtype(value));

  // The value owns the memory. Handing it to the storage keeps it alive.
  std::shared_ptr<void> owner(value, [](void* pointer) {
    milo_value_release(static_cast<milo_value*>(pointer));
  });
  auto storage = milo_value_tensor_writable(value) != 0
                     ? milo::tensor::Storage::wrapMutable(const_cast<void*>(data), bytes,
                                                         milo::tensor::Device::Cpu,
                                                         std::move(owner))
                     : milo::tensor::Storage::wrap(static_cast<const std::byte*>(data), bytes,
                                                   milo::tensor::Device::Cpu, std::move(owner));
  auto view = std::make_shared<milo::tensor::Tensor>(milo::tensor::Tensor::view(
      storage, dtype, shape, milo::tensor::Strides::contiguous(shape), 0));
  return static_cast<PyObject*>(milo::python::tensorObjectCreate(view));
}

/// `milo.run(source)` — run a Milo program and return its value.
PyObject* moduleRun(PyObject* /*self*/, PyObject* args) {
  const char* source = nullptr;
  Py_ssize_t length = 0;
  if (!PyArg_ParseTuple(args, "s#", &source, &length)) return nullptr;

  milo_value* result = nullptr;
  Runtime runtime;
  milo_status status = MILO_OK;
  {
    // Let other Python threads run while the program does its work. The
    // bridge takes the lock again when the program calls into Python.
    void* lock = milo::python::releasePythonLock();
    status = milo_runtime_run(runtime.handle, source, static_cast<size_t>(length), "<python>",
                              &result);
    milo::python::acquirePythonLock(lock);
  }
  if (status != MILO_OK) {
    raiseMiloError("the program failed");
    return nullptr;
  }
  if (result == nullptr) {
    Py_RETURN_NONE;
  }
  PyObject* object = toPython(result);
  return object;
}

/// `milo.load(path, name)` — read one tensor out of a safetensors file.
PyObject* moduleLoad(PyObject* /*self*/, PyObject* args) {
  const char* path = nullptr;
  const char* name = nullptr;
  if (!PyArg_ParseTuple(args, "ss", &path, &name)) return nullptr;

  try {
    milo::safetensors::File file = milo::safetensors::File::open(path);
    auto tensor = file.tensor(name);
    return static_cast<PyObject*>(milo::python::tensorObjectCreate(tensor));
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_OSError, error.what());
    return nullptr;
  }
}

/// `milo.version()` — the runtime's version.
PyObject* moduleVersion(PyObject* /*self*/, PyObject* /*args*/) {
  return PyUnicode_FromString(milo_version_string());
}

PyMethodDef gMethods[] = {
    {"run", &moduleRun, METH_VARARGS, "Run a Milo program and return its value."},
    {"load", &moduleLoad, METH_VARARGS, "Read a tensor out of a safetensors file."},
    {"version", &moduleVersion, METH_NOARGS, "The runtime's version."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef gModule = {
    PyModuleDef_HEAD_INIT,
    "milo",
    "Milo: a programming language for AI and data work.",
    -1,
    gMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit_milo(void) {
  milo::python::tensorObjectPrepare();
  return PyModule_Create(&gModule);
}
