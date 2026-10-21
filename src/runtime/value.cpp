#include "runtime/value.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace milo::runtime {
namespace {

std::string formatDouble(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
  std::ostringstream out;
  out.precision(17);
  out << value;
  std::string text = out.str();
  // Prefer the shortest representation that round-trips.
  for (int precision = 1; precision < 17; ++precision) {
    std::ostringstream shorter;
    shorter.precision(precision);
    shorter << value;
    if (std::stod(shorter.str()) == value) {
      text = shorter.str();
      break;
    }
  }
  return text;
}

}  // namespace

Value Value::null() { return Value(); }

Value Value::boolean(bool value) {
  Value v;
  v.kind_ = Kind::Bool;
  v.bool_ = value;
  return v;
}

Value Value::integer(std::int64_t value) {
  Value v;
  v.kind_ = Kind::Int;
  v.int_ = value;
  return v;
}

Value Value::floating(double value) {
  Value v;
  v.kind_ = Kind::Float;
  v.float_ = value;
  return v;
}

Value Value::string(std::string value) {
  Value v;
  v.kind_ = Kind::String;
  v.string_ = std::move(value);
  return v;
}

Value Value::list(List value) {
  Value v;
  v.kind_ = Kind::List;
  v.list_ = std::make_shared<List>(std::move(value));
  return v;
}

Value Value::dict(Dict value) {
  Value v;
  v.kind_ = Kind::Dict;
  v.dict_ = std::make_shared<Dict>(std::move(value));
  return v;
}

Value Value::model(ModelValue value) {
  Value v;
  v.kind_ = Kind::Model;
  v.model_ = std::make_shared<ModelValue>(std::move(value));
  return v;
}

std::int64_t ModelValue::parameterCount() const {
  if (!tensors) return 0;
  std::int64_t total = 0;
  for (const auto& entry : *tensors) {
    total += entry.second ? entry.second->numel() : 0;
  }
  return total;
}

Value Value::dataset(DatasetValue value) {
  Value v;
  v.kind_ = Kind::Dataset;
  v.dataset_ = std::make_shared<DatasetValue>(std::move(value));
  return v;
}

Value Value::pythonModule(PythonObjectValue value) {
  Value v;
  v.kind_ = Kind::Python;
  v.python_object_ = std::make_shared<PythonObjectValue>(std::move(value));
  return v;
}

Value Value::pythonObject(PythonObjectValue value) {
  Value v;
  v.kind_ = Kind::Python;
  v.python_object_ = std::make_shared<PythonObjectValue>(std::move(value));
  return v;
}

Value Value::layer(ModuleValue value) {
  Value v;
  v.kind_ = Kind::Module;
  v.module_ = std::make_shared<ModuleValue>(std::move(value));
  return v;
}

Value Value::optimizer(OptimizerValue value) {
  Value v;
  v.kind_ = Kind::Optimizer;
  v.optimizer_ = std::make_shared<OptimizerValue>(std::move(value));
  return v;
}

Value Value::stub(StubValue value) {
  Value v;
  v.kind_ = Kind::Stub;
  v.stub_ = std::make_shared<StubValue>(std::move(value));
  return v;
}

Value Value::tensor(std::shared_ptr<tensor::Tensor> value) {
  Value v;
  v.kind_ = Kind::Tensor;
  v.tensor_ = std::move(value);
  return v;
}

Value Value::quantized(std::shared_ptr<tensor::QuantizedTensor> value) {
  Value v;
  v.kind_ = Kind::Quantized;
  v.quantized_ = std::move(value);
  return v;
}

Value Value::iterator(std::shared_ptr<IteratorValue> value) {
  Value v;
  v.kind_ = Kind::Iterator;
  v.iterator_ = std::move(value);
  return v;
}

Value Value::machineFunction(std::shared_ptr<void> value) {
  Value v;
  v.kind_ = Kind::MachineFunction;
  v.machine_function_ = std::move(value);
  return v;
}

Value Value::function(std::shared_ptr<FunctionValue> value) {
  Value v;
  v.kind_ = Kind::Function;
  v.function_ = std::move(value);
  return v;
}

Value Value::builtin(std::shared_ptr<BuiltinValue> value) {
  Value v;
  v.kind_ = Kind::Builtin;
  v.builtin_ = std::move(value);
  return v;
}

bool Value::isTruthy() const {
  switch (kind_) {
    case Kind::Null:
      return false;
    case Kind::Bool:
      return bool_;
    case Kind::Int:
      return int_ != 0;
    case Kind::Float:
      return float_ != 0.0;
    case Kind::String:
      return !string_.empty();
    case Kind::List:
      return !list_->empty();
    case Kind::Dict:
      return !dict_->empty();
    default:
      return true;
  }
}

const char* Value::typeName() const {
  switch (kind_) {
    case Kind::Null:
      return "null";
    case Kind::Bool:
      return "bool";
    case Kind::Int:
      return "int";
    case Kind::Float:
      return "float";
    case Kind::String:
      return "str";
    case Kind::List:
      return "list";
    case Kind::Dict:
      return "dict";
    case Kind::Model:
      return "model";
    case Kind::Dataset:
      return "dataset";
    case Kind::Python:
      return "module";
    case Kind::Module:
      return "module";
    case Kind::Optimizer:
      return "optimizer";
    case Kind::Stub:
      return "function";
    case Kind::Tensor:
      return "tensor";
    case Kind::Quantized:
      return "quantized";
    case Kind::Iterator:
      return "iterator";
    case Kind::MachineFunction:
      return "function";
    case Kind::Function:
      return "function";
    case Kind::Builtin:
      return "builtin";
  }
  return "any";
}

const Value* Value::find(const std::string& key) const {
  if (kind_ != Kind::Dict) return nullptr;
  for (const auto& entry : *dict_) {
    if (entry.first == key) return &entry.second;
  }
  return nullptr;
}

std::string Value::toReprString() const {
  switch (kind_) {
    case Kind::Null:
      return "null";
    case Kind::Bool:
      return bool_ ? "true" : "false";
    case Kind::Int:
      return std::to_string(int_);
    case Kind::Float:
      return formatDouble(float_);
    case Kind::String:
      return "\"" + string_ + "\"";
    case Kind::List: {
      std::ostringstream out;
      out << '[';
      for (std::size_t i = 0; i < list_->size(); ++i) {
        if (i != 0) out << ", ";
        out << (*list_)[i].toReprString();
      }
      out << ']';
      return out.str();
    }
    case Kind::Dict: {
      std::ostringstream out;
      out << '{';
      for (std::size_t i = 0; i < dict_->size(); ++i) {
        if (i != 0) out << ", ";
        out << '"' << (*dict_)[i].first << "\": " << (*dict_)[i].second.toReprString();
      }
      out << '}';
      return out.str();
    }
    case Kind::Model:
      return "<model " + model_->name + ">";
    case Kind::Dataset:
      return "<dataset " + dataset_->name + ">";
    case Kind::Python:
      return "<python object>";
    case Kind::Module:
      return "<module " + module_->kind + ">";
    case Kind::Optimizer:
      return "<optimizer " + optimizer_->kind + ">";
    case Kind::Stub:
      return "<function " + stub_->name + ">";
    case Kind::Tensor:
      return tensor_->toString();
    case Kind::Quantized:
      return "<quantized " + quantized_->scheme + " " + quantized_->shape.toString() + ">";
    case Kind::Function:
      return "<function " + function_->name + ">";
    case Kind::Builtin:
      return "<builtin " + builtin_->name + ">";
  }
  return "any";
}

std::string Value::toDisplayString() const {
  if (kind_ == Kind::String) return string_;
  if (kind_ == Kind::Tensor) return tensor_->toDisplayString();
  return toReprString();
}

}  // namespace runtime
