#include "python/python.hpp"

#include <Python.h>

#include <cstring>
#include <sstream>
#include <stdexcept>

#include "python/dlpack.h"
#include "python/tensor_object.hpp"
#include "runtime/interp.hpp"

namespace milo::python {
namespace {

/// A Milo error carrying a Python traceback.
std::runtime_error pythonError(const std::string& message) {
  return std::runtime_error(message);
}

/// The dtype code and size for a Milo's element type.
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

/// The element type for a DLPack type.
bool typeFromDlpack(uint8_t code, uint8_t bits, tensor::DType& out) {
  if (code == 2 && bits == 32) {
    out = tensor::DType::F32;
  } else if (code == 2 && bits == 64) {
    out = tensor::DType::F64;
  } else if (code == 2 && bits == 16) {
    out = tensor::DType::F16;
  } else if (code == 4 && bits == 16) {
    out = tensor::DType::BF16;
  } else if (code == 0 && bits == 64) {
    out = tensor::DType::I64;
  } else if (code == 0 && bits == 32) {
    out = tensor::DType::I32;
  } else if (code == 0 && bits == 8) {
    out = tensor::DType::I8;
  } else if (code == 1 && bits == 8) {
    out = tensor::DType::U8;
  } else if (code == 6 && bits == 8) {
    out = tensor::DType::Bool;
  } else {
    return false;
  }
  return true;
}

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

/// A Milo element type for a buffer protocol format string.
bool typeFromBufferFormat(const char* format, tensor::DType& out) {
  if (format == nullptr) return false;
  if (std::strcmp(format, "f") == 0) {
    out = tensor::DType::F32;
  } else if (std::strcmp(format, "d") == 0) {
    out = tensor::DType::F64;
  } else if (std::strcmp(format, "q") == 0) {
    out = tensor::DType::I64;
  } else if (std::strcmp(format, "i") == 0) {
    out = tensor::DType::I32;
  } else if (std::strcmp(format, "b") == 0) {
    out = tensor::DType::I8;
  } else if (std::strcmp(format, "B") == 0) {
    out = tensor::DType::U8;
  } else if (std::strcmp(format, "?") == 0) {
    out = tensor::DType::Bool;
  } else {
    return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// The context
// ---------------------------------------------------------------------------

struct Context::Impl {
  /// The interpreter state, when it is a sub-interpreter.
  PyInterpreterState* interpreter = nullptr;
  /// The thread state that was current before the sub-interpreter took over.
  PyThreadState* savedThreadState = nullptr;
  /// Whether this context started the interpreter.
  bool ownsInterpreter = false;
  /// Whether the interpreter was already running.
  bool started = false;
  std::string description = "python";
  /// The text of the last error.
  std::string error;

  ~Impl() {
    if (interpreter != nullptr) {
      // Put the old thread state back, so that the finalisation happens in
      // the right interpreter.
      if (savedThreadState != nullptr) {
        PyThreadState_Swap(savedThreadState);
      }
      Py_FinalizeEx();
      interpreter = nullptr;
    }
  }
};

Context::Context() : impl_(std::make_unique<Impl>()) {
  if (!isAvailable()) {
    throw std::runtime_error("there is no Python in this build");
  }

  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
    impl_->ownsInterpreter = true;
  }
  impl_->started = true;

  // An isolated sub-interpreter has its own global state and its own lock,
  // which is nice — but some C extensions, NumPy's among them, refuse to
  // load in one. So it is opt-in: set MILO_PYTHON_ISOLATED=1 to ask for
  // it, and the program says which one it got.
  const char* isolated = std::getenv("MILO_PYTHON_ISOLATED");
  const bool wantIsolated = isolated != nullptr && std::strcmp(isolated, "0") != 0;

#if PY_VERSION_HEX >= 0x030D0000 && PY_VERSION_HEX < 0x030E0000
  if (wantIsolated) {
    // Python 3.13: an isolated sub-interpreter with its own global state
    // and its own lock.
    PyInterpreterConfig config;
    PyInterpreterConfig_InitIsolated(&config);
    config.gil_used = 0;
    config.use_main_objs = 0;
    const PyStatus status = Py_NewInterpreterFromConfig(&config);
    if (PyStatus_Exception(status)) {
      impl_->description = "python (main interpreter)";
    } else {
      impl_->description = "python (isolated sub-interpreter)";
    }
  } else {
    impl_->description = "python (main interpreter)";
  }
#elif PY_VERSION_HEX >= 0x030C0000 && PY_VERSION_HEX < 0x030D0000
  if (wantIsolated) {
    // Python 3.12: the same idea, with the older spelling.
    PyInterpreterConfig config = {};
    config.use_main_obmalloc = 0;
    config.allow_fork = 0;
    config.allow_exec = 0;
    config.allow_threads = 1;
    config.allow_daemon_threads = 0;
    config.check_multi_interp_extensions = 1;
    config.gil = PyInterpreterConfig_OWN_GIL;

    PyThreadState* threadState = nullptr;
    const PyStatus status = Py_NewInterpreterFromConfig(&threadState, &config);
    if (PyStatus_Exception(status) || threadState == nullptr) {
      impl_->description = "python (main interpreter)";
    } else {
      // The new thread state becomes the current one.
      impl_->savedThreadState = PyThreadState_Swap(threadState);
      impl_->description = "python (isolated sub-interpreter)";
    }
  } else {
    impl_->description = "python (main interpreter)";
  }
#else
  // Before 3.12 a sub-interpreter shares the main one's lock, and from
  // 3.14 on the configuration API changed again. The main interpreter is
  // the safe choice on both.
  impl_->description = "python (main interpreter)";
#endif
}

Context::~Context() = default;

bool Context::isAvailable() {
  // The interpreter is only started when one is wanted; asking whether
  // Python is there does not start it. Nothing here may run Python code:
  // the caller may have given up Python's lock, and running code without
  // it is not allowed.
  static const bool available = [] {
    if (Py_IsInitialized() == 0) {
      Py_InitializeEx(0);
    }
    return Py_IsInitialized() != 0;
  }();
  return available;
}

const std::string& Context::description() const { return impl_->description; }

std::string Context::lastError() const { return impl_->error; }

namespace {

/// The thread state a caller gave up, if it gave one up.
///
/// A non-null value means the lock is not held by this thread. The bridge
/// needs the state to take the lock again without building a second thread
/// state for the same thread, which Python does not allow.
thread_local PyThreadState* g_savedThreadState = nullptr;

/// Does this thread hold Python's lock?
///
/// The bridge takes the lock and gives it back, and the two have to match
/// up: a call inside a call must not take the lock twice.
thread_local bool g_lockHeld = true;

/// Hold Python's lock while Milo talks to Python.
///
/// Milo may run without the lock — a program started from Python gives it
/// up so that other threads keep running — and take it again when it needs
/// Python.
class GilGuard {
 public:
  GilGuard() {
    if (!g_lockHeld && g_savedThreadState != nullptr) {
      // The lock was given up: take it back with the state that was saved.
      PyEval_RestoreThread(g_savedThreadState);
      g_lockHeld = true;
      restored_ = true;
    }
  }

  ~GilGuard() {
    if (restored_) {
      // Give the lock back, but keep the state for next time.
      PyEval_SaveThread();
      g_lockHeld = false;
    }
  }

  GilGuard(const GilGuard&) = delete;
  GilGuard& operator=(const GilGuard&) = delete;

 private:
  bool restored_ = false;
};

/// Fetch the Python error, with its traceback, as text.
std::string fetchError() {
  if (!PyErr_Occurred()) return "an unknown error";

  PyObject* type = nullptr;
  PyObject* value = nullptr;
  PyObject* traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  if (traceback != nullptr) {
    PyException_SetTraceback(value, traceback);
  }

  std::ostringstream out;
  out << "python: ";
  if (value != nullptr) {
    PyObject* text = PyObject_Str(value);
    if (text != nullptr) {
      const char* utf8 = PyUnicode_AsUTF8(text);
      if (utf8 != nullptr) out << utf8;
      Py_DECREF(text);
    } else {
      out << "an exception";
    }
  } else {
    out << "an exception";
  }

  // The traceback, line by line.
  if (traceback != nullptr) {
    PyObject* module = PyImport_ImportModule("traceback");
    if (module != nullptr) {
      PyObject* lines = PyObject_CallMethod(module, "format_exception", "OOO", type ? type : Py_None,
                                            value ? value : Py_None, traceback);
      if (lines != nullptr) {
        const Py_ssize_t count = PySequence_Size(lines);
        for (Py_ssize_t i = 0; i < count; ++i) {
          PyObject* line = PySequence_GetItem(lines, i);
          if (line == nullptr) break;
          const char* utf8 = PyUnicode_AsUTF8(line);
          if (utf8 != nullptr) out << "\n  " << utf8;
          Py_DECREF(line);
        }
        Py_DECREF(lines);
      }
      Py_DECREF(module);
    }
  }

  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  return out.str();
}

}  // namespace

void* releasePythonLock() {
  if (g_lockHeld) {
    g_savedThreadState = PyEval_SaveThread();
    g_lockHeld = false;
  }
  return g_savedThreadState;
}

void acquirePythonLock(void* state) {
  if (state == nullptr || g_lockHeld) return;
  PyEval_RestoreThread(static_cast<PyThreadState*>(state));
  g_lockHeld = true;
  g_savedThreadState = nullptr;
}

runtime::Value Context::importModule(const std::string& name) {
  const GilGuard guard;
  PyObject* module = PyImport_ImportModule(name.c_str());
  if (module == nullptr && impl_->savedThreadState != nullptr) {
    // The module could not be imported in the isolated sub-interpreter.
    // Some C extensions — NumPy's among them — only work in the main
    // interpreter, so fall back to it rather than fail.
    PyErr_Clear();
    PyThreadState* subInterpreterState = PyThreadState_Swap(impl_->savedThreadState);
    (void)subInterpreterState;
    impl_->savedThreadState = nullptr;
    impl_->description = "python (main interpreter: a module needed it)";
    module = PyImport_ImportModule(name.c_str());
  }
  if (module == nullptr) {
    impl_->error = fetchError();
    throw pythonError(impl_->error);
  }

  runtime::Value value = runtime::Value::pythonModule(
      runtime::PythonObjectValue{name, runtime::PythonObjectValue::Kind::Module, nullptr});
  // Attach the module itself, so that it stays alive.
  runtime::PythonObjectValue& object =
      const_cast<runtime::PythonObjectValue&>(value.asPythonObject());
  object.handle = std::shared_ptr<void>(module, [](void* pointer) {
    Py_DECREF(static_cast<PyObject*>(pointer));
  });
  return value;
}

runtime::Value Context::getAttribute(const runtime::Value& object, const std::string& name) {
  const GilGuard guard;
  if (object.kind() != runtime::Value::Kind::Python) {
    throw pythonError("that is not a Python object");
  }
  const auto& python = object.asPythonObject();
  if (!python.handle) throw pythonError("that Python object is gone");

  PyObject* target = static_cast<PyObject*>(python.handle.get());
  PyObject* attribute = PyObject_GetAttrString(target, name.c_str());
  if (attribute == nullptr) {
    impl_->error = fetchError();
    throw pythonError(impl_->error);
  }
  runtime::Value result = fromPython(attribute);
  Py_DECREF(attribute);
  return result;
}

runtime::Value Context::callObject(const runtime::Value& object,
                                   const std::vector<runtime::Value>& args,
                                   const std::vector<std::pair<std::string, runtime::Value>>& kwargs) {
  const GilGuard guard;
  if (object.kind() != runtime::Value::Kind::Python) {
    throw pythonError("that is not a Python object");
  }
  const auto& python = object.asPythonObject();
  if (!python.handle) throw pythonError("that Python object is gone");

  PyObject* target = static_cast<PyObject*>(python.handle.get());

  std::vector<PyObject*> arguments;
  arguments.reserve(args.size());
  for (const runtime::Value& argument : args) {
    arguments.push_back(static_cast<PyObject*>(toPython(argument)));
  }

  PyObject* keywords = nullptr;
  if (!kwargs.empty()) {
    keywords = PyDict_New();
    for (const auto& entry : kwargs) {
      PyDict_SetItemString(keywords, entry.first.c_str(),
                           static_cast<PyObject*>(toPython(entry.second)));
    }
  }

  PyObject* argumentTuple = PyTuple_New(static_cast<Py_ssize_t>(arguments.size()));
  if (argumentTuple == nullptr) {
    for (PyObject* argument : arguments) {
      Py_XDECREF(argument);
    }
    Py_XDECREF(keywords);
    throw pythonError("python: there was no room for the arguments");
  }
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    PyTuple_SET_ITEM(argumentTuple, static_cast<Py_ssize_t>(i), arguments[i]);
  }

  PyObject* result = PyObject_Call(target, argumentTuple, keywords);
  Py_DECREF(argumentTuple);
  Py_XDECREF(keywords);

  if (result == nullptr) {
    impl_->error = fetchError();
    throw pythonError(impl_->error);
  }
  runtime::Value value = fromPython(result);
  Py_DECREF(result);
  return value;
}

runtime::Value Context::call(const runtime::Value& object, const std::string& name,
                             const std::vector<runtime::Value>& args,
                             const std::vector<std::pair<std::string, runtime::Value>>& kwargs) {
  runtime::Value attribute = getAttribute(object, name);
  return callObject(attribute, args, kwargs);
}

// ---------------------------------------------------------------------------
// Milo to Python
// ---------------------------------------------------------------------------

void* Context::toPython(const runtime::Value& value) {
  const GilGuard guard;
  switch (value.kind()) {
    case runtime::Value::Kind::Null:
      Py_RETURN_NONE;
    case runtime::Value::Kind::Bool:
      if (value.asBool()) {
        Py_RETURN_TRUE;
      }
      Py_RETURN_FALSE;
    case runtime::Value::Kind::Int:
      return PyLong_FromLongLong(static_cast<Py_ssize_t>(value.asInt()));
    case runtime::Value::Kind::Float:
      return PyFloat_FromDouble(value.asFloat());
    case runtime::Value::Kind::String:
      return PyUnicode_FromStringAndSize(value.asString().data(),
                                        static_cast<Py_ssize_t>(value.asString().size()));
    case runtime::Value::Kind::List: {
      PyObject* list = PyList_New(static_cast<Py_ssize_t>(value.asList().size()));
      if (list == nullptr) return nullptr;
      for (std::size_t i = 0; i < value.asList().size(); ++i) {
        PyObject* item = static_cast<PyObject*>(toPython(value.asList()[i]));
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), item);
      }
      return list;
    }
    case runtime::Value::Kind::Dict: {
      PyObject* dict = PyDict_New();
      if (dict == nullptr) return nullptr;
      for (const auto& entry : value.asDict()) {
        PyDict_SetItemString(dict, entry.first.c_str(),
                             static_cast<PyObject*>(toPython(entry.second)));
      }
      return dict;
    }
    case runtime::Value::Kind::Tensor: {
      // A tensor becomes an object that exposes the buffer protocol, so
      // that NumPy can view it without copying.
      const auto& tensor = value.asTensor();
      if (!tensor) throw pythonError("that tensor has no memory");
      const std::size_t bytes =
          static_cast<std::size_t>(tensor->numel()) * tensor::dtypeSize(tensor->dtype());
      const auto* data = static_cast<const std::byte*>(tensor->storage()->data()) +
                         tensor->byteOffset();
      // The Python object views the same memory; the tensor keeps it
      // alive for as long as Python can see it.
      std::shared_ptr<tensor::Storage> storage;
      if (tensor->storage()->readOnly()) {
        storage = tensor::Storage::wrap(data, bytes, tensor->device(), tensor->storage());
      } else {
        storage = tensor::Storage::wrapMutable(const_cast<std::byte*>(data), bytes,
                                                tensor->device(), tensor->storage());
      }
      auto view = std::make_shared<tensor::Tensor>(tensor::Tensor::view(
          storage, tensor->dtype(), tensor->shape(), tensor::Strides::contiguous(tensor->shape()),
          0));
      return tensorObjectCreate(view);
    }
    case runtime::Value::Kind::Python: {
      const auto& python = value.asPythonObject();
      if (!python.handle) {
        Py_RETURN_NONE;
      }
      Py_INCREF(static_cast<PyObject*>(python.handle.get()));
      return python.handle.get();
    }
    default:
      throw pythonError(std::string("a ") + value.typeName() +
                        " cannot be turned into a Python value");
  }
}

// ---------------------------------------------------------------------------
// Python to Milo
// ---------------------------------------------------------------------------

runtime::Value Context::fromPython(void* object) {
  const GilGuard guard;
  PyObject* value = static_cast<PyObject*>(object);
  if (value != nullptr && value != Py_None) {
  }
  if (value == nullptr || value == Py_None) {
    return runtime::Value::null();
  }

  if (PyBool_Check(value)) {
    return runtime::Value::boolean(value == Py_True);
  }
  if (PyLong_Check(value)) {
    return runtime::Value::integer(static_cast<std::int64_t>(PyLong_AsLongLong(value)));
  }
  if (PyFloat_Check(value)) {
    const double number = PyFloat_AsDouble(value);
    return runtime::Value::floating(number);
  }
  if (PyUnicode_Check(value)) {
    const char* utf8 = PyUnicode_AsUTF8(value);
    return runtime::Value::string(utf8 != nullptr ? utf8 : "");
  }
  if (PyList_Check(value) || PyTuple_Check(value)) {
    const Py_ssize_t count = PySequence_Size(value);
    runtime::Value::List items;
    items.reserve(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
      PyObject* item = PySequence_GetItem(value, i);
      if (item == nullptr) break;
      items.push_back(fromPython(item));
      Py_DECREF(item);
    }
    return runtime::Value::list(std::move(items));
  }
  if (PyDict_Check(value)) {
    PyObject* key = nullptr;
    PyObject* item = nullptr;
    Py_ssize_t position = 0;
    runtime::Value::Dict entries;
    while (PyDict_Next(value, &position, &key, &item)) {
      const char* text = PyUnicode_AsUTF8(key);
      entries.emplace_back(text != nullptr ? text : "", fromPython(item));
    }
    return runtime::Value::dict(std::move(entries));
  }

  // A buffer provider — a NumPy array, for one — becomes a tensor that
  // views the same memory.
  if (PyObject_CheckBuffer(value)) {
    Py_buffer view;
    if (PyObject_GetBuffer(value, &view, PyBUF_WRITABLE | PyBUF_FORMAT) == 0) {
      tensor::DType dtype = tensor::DType::F32;
      if (typeFromBufferFormat(view.format, dtype)) {
        const std::size_t elementSize = tensor::dtypeSize(dtype);
        const std::size_t count = static_cast<std::size_t>(view.len) / elementSize;

        // The shape comes from the object's own attributes, when it has
        // them; a plain buffer has none.
        std::vector<std::int64_t> shape;
        PyObject* shapeAttribute = PyObject_GetAttrString(value, "shape");
        if (shapeAttribute != nullptr) {
          if (PyTuple_Check(shapeAttribute) || PyList_Check(shapeAttribute)) {
            const Py_ssize_t dimensions = PySequence_Size(shapeAttribute);
            for (Py_ssize_t i = 0; i < dimensions; ++i) {
              PyObject* dimension = PySequence_GetItem(shapeAttribute, i);
              if (dimension == nullptr) break;
              shape.push_back(static_cast<std::int64_t>(PyLong_AsLongLong(dimension)));
              Py_DECREF(dimension);
            }
          }
          Py_DECREF(shapeAttribute);
        }
        if (shape.empty()) shape.push_back(static_cast<std::int64_t>(count));

        // Keep the Python object alive for as long as the tensor is: the
        // buffer is released and the reference given back when the storage
        // goes away.
        struct BufferOwner {
          Py_buffer view;
          PyObject* object = nullptr;
          ~BufferOwner() {
            // Releasing a buffer is a Python call, and the lock may have
            // been given up by then.
            const GilGuard guard;
            PyBuffer_Release(&view);
            if (object != nullptr) Py_DECREF(object);
          }
        };
        auto owner = std::make_shared<BufferOwner>();
        owner->view = view;
        owner->object = value;
        Py_INCREF(value);

        auto owned = std::shared_ptr<tensor::Storage>(
            tensor::Storage::wrap(static_cast<const std::byte*>(view.buf),
                                  static_cast<std::size_t>(view.len), tensor::Device::Cpu,
                                  std::shared_ptr<void>(owner, [](void*) {})));

        tensor::Tensor tensor = tensor::Tensor::view(
            owned, dtype, tensor::Shape(shape),
            tensor::Strides::contiguous(tensor::Shape(shape)), 0);
        return runtime::Value::tensor(std::make_shared<tensor::Tensor>(std::move(tensor)));
      }
      PyBuffer_Release(&view);
    }
  }

  // A DLPack provider — a PyTorch tensor, for one.
  if (PyObject_HasAttrString(value, "__dlpack__")) {
    PyObject* capsule = PyObject_CallMethod(value, "__dlpack__", nullptr);
    if (capsule != nullptr) {
      void* pointer = PyCapsule_GetPointer(capsule, MILO_DLPACK_CAPSULE_NAME);
      if (pointer == nullptr) {
        PyErr_Clear();
        Py_DECREF(capsule);
      } else {
        auto* managed = static_cast<DLManagedTensor*>(pointer);
        const DLTensor& source = managed->dl_tensor;
        tensor::DType dtype = tensor::DType::F32;
        if (typeFromDlpack(source.dtype.code, source.dtype.bits, dtype)) {
          std::vector<std::int64_t> shape;
          for (int32_t i = 0; i < source.ndim; ++i) {
            shape.push_back(source.shape[i]);
          }
          if (shape.empty()) shape.push_back(1);

          const std::size_t elementSize = tensor::dtypeSize(dtype);
          std::size_t total = 1;
          for (const std::int64_t dim : shape) {
            total *= static_cast<std::size_t>(dim);
          }

          auto storage = tensor::Storage::wrap(
              static_cast<const std::byte*>(source.data) + source.byte_offset,
              total * elementSize, tensor::Device::Cpu,
              std::shared_ptr<void>(managed, [](void* pointer) {
                // The producer's deleter is a Python call.
                const GilGuard guard;
                auto* tensor = static_cast<DLManagedTensor*>(pointer);
                if (tensor->deleter != nullptr) tensor->deleter(tensor);
              }));

          tensor::Tensor tensor = tensor::Tensor::view(
              storage, dtype, tensor::Shape(shape),
              tensor::Strides::contiguous(tensor::Shape(shape)), 0);
          Py_DECREF(capsule);
          return runtime::Value::tensor(std::make_shared<tensor::Tensor>(std::move(tensor)));
        }
        Py_DECREF(capsule);
      }
    }
  }

  // Anything else stays a Python object.
  Py_INCREF(value);
  runtime::Value result = runtime::Value::pythonObject(
      runtime::PythonObjectValue{"", runtime::PythonObjectValue::Kind::Object, nullptr});
  runtime::PythonObjectValue& python =
      const_cast<runtime::PythonObjectValue&>(result.asPythonObject());
  python.handle = std::shared_ptr<void>(value, [](void* pointer) {
    Py_DECREF(static_cast<PyObject*>(pointer));
  });
  return result;
}

}  // namespace milo::python
