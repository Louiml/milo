// The stable C ABI.
//
// Rules for this file:
//   * no C++ exception may escape into C
//   * no C++ type may appear in a function signature
//   * every entry point must work when called from a foreign language
#include "milo/milo.h"

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "runtime/interp.hpp"
#include "core/lexer.hpp"
#include "core/parser.hpp"

namespace {

/// The calling thread's last error message.
thread_local std::string g_last_error;

void setLastError(const std::string& message) { g_last_error = message; }

}  // namespace

// `milo_runtime` and `milo_value` are opaque in the header; their layouts
// live here.
struct milo_runtime {
  std::unique_ptr<milo::Interpreter> interpreter;
};

struct milo_value {
  std::shared_ptr<milo::runtime::Value> value;
};

namespace {

/// The tensor inside a value, or nullptr.
std::shared_ptr<milo::tensor::Tensor> tensorOf(const milo_value* v) {
  if (v == nullptr || !v->value) return nullptr;
  if (v->value->kind() != milo::runtime::Value::Kind::Tensor) return nullptr;
  return v->value->asTensor();
}

}  // namespace

extern "C" {

uint32_t milo_abi_version(void) { return MILO_ABI_VERSION; }

const char* milo_version_string(void) { return MILO_RUNTIME_VERSION; }

const char* milo_language_version(void) { return MILO_LANGUAGE_VERSION; }

const char* milo_status_to_string(milo_status status) {
  switch (status) {
    case MILO_OK:
      return "ok";
    case MILO_ERR_SYNTAX:
      return "syntax error";
    case MILO_ERR_TYPE:
      return "type error";
    case MILO_ERR_RUNTIME:
      return "runtime error";
    case MILO_ERR_IO:
      return "i/o error";
    case MILO_ERR_NOT_FOUND:
      return "not found";
    case MILO_ERR_UNSUPPORTED:
      return "unsupported";
    case MILO_ERR_NOMEM:
      return "out of memory";
    case MILO_ERR_INTERNAL:
      return "internal error";
  }
  return "unknown error";
}

size_t milo_last_error(char* buf, size_t cap) {
  const std::string& message = g_last_error;
  if (buf == nullptr || cap == 0) return message.size();
  const size_t written = message.size() < cap - 1 ? message.size() : cap - 1;
  std::memcpy(buf, message.data(), written);
  buf[written] = '\0';
  return message.size();
}

void milo_clear_error(void) { g_last_error.clear(); }

milo_runtime* milo_runtime_create(void) {
  try {
    auto* rt = new milo_runtime();
    rt->interpreter = std::make_unique<milo::Interpreter>();
    g_last_error.clear();
    return rt;
  } catch (const std::exception& e) {
    setLastError(e.what());
    return nullptr;
  } catch (...) {
    setLastError("unknown failure while creating a runtime");
    return nullptr;
  }
}

void milo_runtime_destroy(milo_runtime* rt) { delete rt; }

milo_value_kind milo_value_kind_of(const milo_value* v) {
  if (v == nullptr || !v->value) return MILO_VALUE_NULL;
  switch (v->value->kind()) {
    case milo::runtime::Value::Kind::Null:
      return MILO_VALUE_NULL;
    case milo::runtime::Value::Kind::Bool:
      return MILO_VALUE_BOOL;
    case milo::runtime::Value::Kind::Int:
      return MILO_VALUE_INT;
    case milo::runtime::Value::Kind::Float:
      return MILO_VALUE_FLOAT;
    case milo::runtime::Value::Kind::String:
      return MILO_VALUE_STRING;
    case milo::runtime::Value::Kind::List:
      return MILO_VALUE_LIST;
    case milo::runtime::Value::Kind::Dict:
      return MILO_VALUE_DICT;
    case milo::runtime::Value::Kind::Tensor:
    case milo::runtime::Value::Kind::Quantized:
      return MILO_VALUE_TENSOR;
    case milo::runtime::Value::Kind::Model:
      return MILO_VALUE_MODEL;
    case milo::runtime::Value::Kind::Dataset:
      return MILO_VALUE_DATASET;
    case milo::runtime::Value::Kind::Module:
      return MILO_VALUE_MODULE;
    case milo::runtime::Value::Kind::Function:
    case milo::runtime::Value::Kind::Stub:
      return MILO_VALUE_FUNCTION;
    case milo::runtime::Value::Kind::Builtin:
      return MILO_VALUE_BUILTIN;
  }
  return MILO_VALUE_NULL;
}

int milo_value_is_null(const milo_value* v) {
  if (v == nullptr || !v->value) return 1;
  return v->value->isNull() ? 1 : 0;
}

int milo_value_as_bool(const milo_value* v) {
  if (v == nullptr || !v->value) return 0;
  return v->value->asBool() ? 1 : 0;
}

int64_t milo_value_as_int(const milo_value* v) {
  if (v == nullptr || !v->value) return 0;
  return v->value->asInt();
}

double milo_value_as_float(const milo_value* v) {
  if (v == nullptr || !v->value) return 0.0;
  return v->value->asFloat();
}

const char* milo_value_as_string(const milo_value* v) {
  if (v == nullptr || !v->value) return nullptr;
  if (v->value->kind() != milo::runtime::Value::Kind::String) return nullptr;
  return v->value->asString().c_str();
}

size_t milo_value_string_length(const milo_value* v) {
  if (v == nullptr || !v->value) return 0;
  if (v->value->kind() != milo::runtime::Value::Kind::String) return 0;
  return v->value->asString().size();
}

int milo_value_tensor_rank(const milo_value* v) {
  const auto tensor = tensorOf(v);
  if (tensor == nullptr) return 0;
  return static_cast<int>(tensor->rank());
}

int64_t milo_value_tensor_dim(const milo_value* v, int axis) {
  const auto tensor = tensorOf(v);
  if (tensor == nullptr) return 0;
  if (axis < 0 || static_cast<std::size_t>(axis) >= tensor->rank()) return 0;
  return tensor->shape().dim(static_cast<std::size_t>(axis));
}

milo_dtype milo_value_tensor_dtype(const milo_value* v) {
  const auto tensor = tensorOf(v);
  if (tensor == nullptr) return MILO_DTYPE_F32;
  switch (tensor->dtype()) {
    case milo::tensor::DType::F32: return MILO_DTYPE_F32;
    case milo::tensor::DType::F16: return MILO_DTYPE_F16;
    case milo::tensor::DType::BF16: return MILO_DTYPE_BF16;
    case milo::tensor::DType::F64: return MILO_DTYPE_F64;
    case milo::tensor::DType::I8: return MILO_DTYPE_I8;
    case milo::tensor::DType::I32: return MILO_DTYPE_I32;
    case milo::tensor::DType::I64: return MILO_DTYPE_I64;
    case milo::tensor::DType::U8: return MILO_DTYPE_U8;
    case milo::tensor::DType::Bool: return MILO_DTYPE_BOOL;
  }
  return MILO_DTYPE_F32;
}

const void* milo_value_tensor_data(const milo_value* v) {
  const auto tensor = tensorOf(v);
  if (tensor == nullptr) return nullptr;
  const auto& storage = tensor->storage();
  if (storage == nullptr) return nullptr;
  const auto* base = static_cast<const std::byte*>(storage->data());
  return base + tensor->byteOffset();
}

size_t milo_value_tensor_bytes(const milo_value* v) {
  const auto tensor = tensorOf(v);
  if (tensor == nullptr) return 0;
  return static_cast<size_t>(tensor->numel()) * milo::tensor::dtypeSize(tensor->dtype());
}

int milo_value_tensor_writable(const milo_value* v) {
  const auto tensor = tensorOf(v);
  if (tensor == nullptr) return 0;
  const auto& storage = tensor->storage();
  if (storage == nullptr) return 0;
  return storage->readOnly() ? 0 : 1;
}

void milo_value_release(milo_value* v) { delete v; }

milo_status milo_runtime_check(milo_runtime* rt, const char* src, size_t len,
                                const char* filename) {
  if (rt == nullptr) {
    setLastError("no runtime");
    return MILO_ERR_INTERNAL;
  }
  try {
    const std::string source(src == nullptr ? "" : src, src == nullptr ? 0 : len);
    const std::string name(filename == nullptr ? "<string>" : filename);

    milo::Lexer lexer(source, name);
    std::vector<milo::Token> tokens = lexer.tokenize();
    if (lexer.diagnostics().hasErrors()) {
      setLastError(lexer.diagnostics().render(source, name));
      return MILO_ERR_SYNTAX;
    }

    milo::Parser parser(std::move(tokens), name);
    parser.parseProgram();
    if (parser.diagnostics().hasErrors()) {
      setLastError(parser.diagnostics().render(source, name));
      return MILO_ERR_SYNTAX;
    }
    g_last_error.clear();
    return MILO_OK;
  } catch (const std::exception& e) {
    setLastError(e.what());
    return MILO_ERR_INTERNAL;
  } catch (...) {
    setLastError("unknown failure while checking a program");
    return MILO_ERR_INTERNAL;
  }
}

milo_status milo_runtime_run(milo_runtime* rt, const char* src, size_t len, const char* filename,
                              milo_value** out_value) {
  if (rt == nullptr) {
    setLastError("no runtime");
    return MILO_ERR_INTERNAL;
  }
  const std::string source(src == nullptr ? "" : src, src == nullptr ? 0 : len);
  try {
    const std::string name(filename == nullptr ? "<string>" : filename);

    milo::Lexer lexer(source, name);
    std::vector<milo::Token> tokens = lexer.tokenize();
    if (lexer.diagnostics().hasErrors()) {
      setLastError(lexer.diagnostics().render(source, name));
      return MILO_ERR_SYNTAX;
    }

    milo::Parser parser(std::move(tokens), name);
    std::vector<milo::ast::StmtPtr> program = parser.parseProgram();
    if (parser.diagnostics().hasErrors()) {
      setLastError(parser.diagnostics().render(source, name));
      return MILO_ERR_SYNTAX;
    }

    milo::runtime::Value result = rt->interpreter->runProgram(program);

    if (out_value != nullptr) {
      auto handle = std::make_unique<milo_value>();
      handle->value = std::make_shared<milo::runtime::Value>(std::move(result));
      *out_value = handle.release();
    }
    g_last_error.clear();
    return MILO_OK;
  } catch (const milo::RuntimeError& error) {
    milo::Diagnostic diagnostic(milo::Severity::Error, error.message, error.span);
    if (!error.suggestion.empty()) diagnostic.label = error.suggestion;
    milo::DiagnosticBag bag;
    bag.add(std::move(diagnostic));
    const std::string name(filename == nullptr ? "<string>" : filename);
    setLastError(bag.render(source, name));
    return MILO_ERR_RUNTIME;
  } catch (const std::exception& e) {
    setLastError(e.what());
    return MILO_ERR_INTERNAL;
  } catch (...) {
    setLastError("unknown failure while running a program");
    return MILO_ERR_INTERNAL;
  }
}

milo_status milo_runtime_run_and_print(milo_runtime* rt, const char* src, size_t len,
                                        const char* filename) {
  milo_value* value = nullptr;
  const milo_status status = milo_runtime_run(rt, src, len, filename, &value);
  if (status != MILO_OK) {
    return status;
  }
  if (value != nullptr) {
    if (milo_value_kind_of(value) != MILO_VALUE_NULL) {
      switch (milo_value_kind_of(value)) {
        case MILO_VALUE_BOOL:
          std::cout << (milo_value_as_bool(value) ? "true" : "false");
          break;
        case MILO_VALUE_INT:
          std::cout << milo_value_as_int(value);
          break;
        case MILO_VALUE_FLOAT:
          std::cout << milo_value_as_float(value);
          break;
        case MILO_VALUE_STRING:
          std::cout << milo_value_as_string(value);
          break;
        default:
          std::cout << "<value>";
          break;
      }
      std::cout << '\n';
    }
    milo_value_release(value);
  }
  return MILO_OK;
}

}  // extern "C"
