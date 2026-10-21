// The Python bridge.
//
// Milo embeds CPython. A program can import a Python module and use it
// as if it were a Milo namespace:
//
//     use python "numpy" as np
//     let x = np.asarray([1.0, 2.0])
//
// Tensors cross the boundary without copying: NumPy arrays come through
// the buffer protocol, and PyTorch tensors through DLPack.
#ifndef MILO_PYTHON_PYTHON_HPP
#define MILO_PYTHON_PYTHON_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/value.hpp"

namespace milo::python {

/// Give up Python's lock, so that other threads can run.
///
/// The result has to be handed back to `acquirePythonLock` when the work
/// is done. A program started from Python gives the lock up like this, so
/// that a long run does not freeze the rest of the program; the bridge
/// takes the lock again when the program calls into Python.
void* releasePythonLock();

/// Take Python's lock again, with the state from `releasePythonLock`.
void acquirePythonLock(void* state);

/// A Python interpreter, and the objects Milo holds inside it.
///
/// The interpreter is started when the first one is made. Where the
/// runtime supports it, it is an isolated sub-interpreter with its own
/// global state; otherwise it is the main interpreter.
class Context {
 public:
  Context();
  ~Context();

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  /// Is there a Python at all?
  static bool isAvailable();

  /// A short description of the interpreter, for diagnostics.
  const std::string& description() const;

  /// Import a module and return it as a Milo value.
  ///
  /// Throws a `RuntimeError` when the import fails.
  runtime::Value importModule(const std::string& name);

  /// Read one attribute of a Python object.
  runtime::Value getAttribute(const runtime::Value& object, const std::string& name);

  /// Call something in a Python object.
  runtime::Value call(const runtime::Value& object, const std::string& name,
                      const std::vector<runtime::Value>& args,
                      const std::vector<std::pair<std::string, runtime::Value>>& kwargs);

  /// Call a Python object directly.
  runtime::Value callObject(const runtime::Value& object,
                            const std::vector<runtime::Value>& args,
                            const std::vector<std::pair<std::string, runtime::Value>>& kwargs);

  /// Turn a Milo value into a Python object.
  ///
  /// The result is a new reference; the caller owns it.
  void* toPython(const runtime::Value& value);

  /// Turn a Python object into a Milo value.
  ///
  /// The object is not stolen.
  runtime::Value fromPython(void* object);

  /// The text of the last Python error, traceback and all.
  std::string lastError() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace milo::python

#endif  // MILO_PYTHON_PYTHON_HPP
