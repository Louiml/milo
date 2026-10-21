// The tensor builtins: construction, elementwise maths, and shape changes.
#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include "runtime/builtins.hpp"
#include "autograd/ops.hpp"
#include "safetensors/writer.hpp"
#include "tensor/backend.hpp"
#include "tensor/ops.hpp"
#include "tensor/random.hpp"
#include "tensor/types.hpp"

namespace milo::runtime {
namespace {

using tensor::BinaryOp;
using tensor::DType;
using tensor::Shape;
using tensor::Tensor;
using tensor::UnaryOp;


/// Turn the arguments of a call into a shape.
///
/// Both `zeros(2, 3)` and `zeros([2, 3])` work.
Shape shapeFromArgs(const std::vector<Value>& args, std::size_t start) {
  std::vector<std::int64_t> dims;
  if (args.size() == start + 1 && args[start].kind() == Value::Kind::List) {
    for (const Value& item : args[start].asList()) {
      if (item.kind() != Value::Kind::Int) {
        throw RuntimeError{"a shape must be made of whole numbers", SourceSpan::start(), {}};
      }
      dims.push_back(item.asInt());
    }
    return Shape(std::move(dims));
  }
  for (std::size_t i = start; i < args.size(); ++i) {
    if (args[i].kind() != Value::Kind::Int) {
      throw RuntimeError{"a shape must be made of whole numbers", SourceSpan::start(), {}};
    }
    dims.push_back(args[i].asInt());
  }
  return Shape(std::move(dims));
}

/// The dtype an operation should produce, from an optional string argument.
DType dtypeFromArgs(const std::vector<Value>& args, std::size_t start, DType fallback) {
  for (std::size_t i = start; i < args.size(); ++i) {
    if (args[i].kind() == Value::Kind::String) {
      DType dtype = DType::F32;
      if (!tensor::dtypeFromName(args[i].asString(), dtype)) {
        throw RuntimeError{"'" + args[i].asString() + "' is not a dtype",
                           SourceSpan::start(), {}};
      }
      return dtype;
    }
  }
  return fallback;
}

/// Fetch a tensor argument.
std::shared_ptr<Tensor> wantTensor(const Value& value, const char* what) {
  if (value.kind() != Value::Kind::Tensor) {
    throw RuntimeError{std::string(what) + " needs a tensor, not a " + value.typeName(),
                       SourceSpan::start(), {}};
  }
  return value.asTensor();
}

/// A tensor or a plain number, as a tensor.
/// Wrap a tensor in a value. The tensor is *not* copied: the value shares
/// the tensor, so that the gradient graph stays connected.
Value makeTensor(std::shared_ptr<Tensor> tensor) {
  return Value::tensor(std::move(tensor));
}

// --- construction ---------------------------------------------------------

Value builtinZeros(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  if (args.empty()) {
    throw RuntimeError{"zeros needs a shape", SourceSpan::start(), {}};
  }
  const Shape shape = shapeFromArgs(args, 0);
  const DType dtype = dtypeFromArgs(args, args.size(), DType::F32);
  return makeTensor(std::make_shared<Tensor>(Tensor::zeros(dtype, shape)));
}

Value builtinOnes(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  if (args.empty()) {
    throw RuntimeError{"ones needs a shape", SourceSpan::start(), {}};
  }
  const Shape shape = shapeFromArgs(args, 0);
  const DType dtype = dtypeFromArgs(args, args.size(), DType::F32);
  return makeTensor(std::make_shared<Tensor>(Tensor::ones(dtype, shape)));
}

Value builtinFull(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() < 2) {
    throw RuntimeError{"full needs a shape and a value", SourceSpan::start(), {}};
  }
  const Shape shape = shapeFromArgs(args, 0);
  const Value& fill = args[args.size() - 1];
  if (fill.kind() != Value::Kind::Int && fill.kind() != Value::Kind::Float) {
    throw RuntimeError{"full needs a number to fill with", SourceSpan::start(), {}};
  }
  const DType dtype = dtypeFromArgs(args, 1, DType::F32);
  const double value = fill.kind() == Value::Kind::Int ? static_cast<double>(fill.asInt())
                                                        : fill.asFloat();
  return makeTensor(std::make_shared<Tensor>(Tensor::full(dtype, shape, value)));
}

Value builtinRandn(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  if (args.empty()) {
    throw RuntimeError{"randn needs a shape", SourceSpan::start(), {}};
  }
  const Shape shape = shapeFromArgs(args, 0);
  const DType dtype = dtypeFromArgs(args, args.size(), DType::F32);

  Tensor tensor = Tensor::create(dtype, shape);
  for (std::int64_t i = 0; i < tensor.numel(); ++i) {
    tensor.setFlat(i, tensor::randomNormal());
  }
  return makeTensor(std::make_shared<Tensor>(std::move(tensor)));
}

Value builtinRand(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  if (args.empty()) {
    throw RuntimeError{"rand needs a shape", SourceSpan::start(), {}};
  }
  const Shape shape = shapeFromArgs(args, 0);
  const DType dtype = dtypeFromArgs(args, args.size(), DType::F32);

  Tensor tensor = Tensor::create(dtype, shape);
  for (std::int64_t i = 0; i < tensor.numel(); ++i) {
    tensor.setFlat(i, tensor::randomUniform());
  }
  return makeTensor(std::make_shared<Tensor>(std::move(tensor)));
}

Value builtinArange(Interpreter&, const std::vector<Value>& args,
                    const std::vector<std::pair<std::string, Value>>&) {
  if (args.empty() || args.size() > 3) {
    throw RuntimeError{"arange takes one, two, or three numbers", SourceSpan::start(), {}};
  }
  for (const Value& argument : args) {
    if (argument.kind() != Value::Kind::Int && argument.kind() != Value::Kind::Float) {
      throw RuntimeError{"arange works with numbers only", SourceSpan::start(), {}};
    }
  }
  const double start = args.size() == 1 ? 0.0
                                         : (args[0].kind() == Value::Kind::Int
                                                ? static_cast<double>(args[0].asInt())
                                                : args[0].asFloat());
  const double stop = args.size() == 1 ? (args[0].kind() == Value::Kind::Int
                                              ? static_cast<double>(args[0].asInt())
                                              : args[0].asFloat())
                                        : (args[1].kind() == Value::Kind::Int
                                               ? static_cast<double>(args[1].asInt())
                                               : args[1].asFloat());
  const double step = args.size() == 3 ? (args[2].kind() == Value::Kind::Int
                                              ? static_cast<double>(args[2].asInt())
                                              : args[2].asFloat())
                                        : 1.0;
  if (step == 0.0) {
    throw RuntimeError{"arange cannot step by zero", SourceSpan::start(), {}};
  }

  const auto count = static_cast<std::int64_t>(std::ceil((stop - start) / step));
  Tensor tensor = Tensor::create(DType::F32, Shape{std::vector<std::int64_t>{std::max<std::int64_t>(
                                                          count, 0)}});
  for (std::int64_t i = 0; i < tensor.numel(); ++i) {
    tensor.setFlat(i, start + static_cast<double>(i) * step);
  }
  return makeTensor(std::make_shared<Tensor>(std::move(tensor)));
}

Value builtinEye(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  if (args.empty() || args[0].kind() != Value::Kind::Int) {
    throw RuntimeError{"eye needs a size", SourceSpan::start(), {}};
  }
  const auto size = static_cast<std::int64_t>(args[0].asInt());
  if (size < 0) {
    throw RuntimeError{"eye needs a size that is not negative", SourceSpan::start(), {}};
  }
  const DType dtype = dtypeFromArgs(args, 1, DType::F32);

  Tensor tensor = Tensor::zeros(dtype, Shape{std::vector<std::int64_t>{size, size}});
  for (std::int64_t i = 0; i < size; ++i) {
    tensor.set({i, i}, 1.0);
  }
  return makeTensor(std::make_shared<Tensor>(std::move(tensor)));
}

/// Flatten a (possibly nested) list into a shape and a list of numbers.
bool flattenList(const Value& list, std::vector<std::int64_t>& shape,
                 std::vector<double>& values) {
  if (list.kind() != Value::Kind::List) return false;
  shape.push_back(static_cast<std::int64_t>(list.asList().size()));
  if (list.asList().empty()) return true;

  const bool nested = list.asList().front().kind() == Value::Kind::List;
  if (!nested) {
    for (const Value& item : list.asList()) {
      if (item.kind() == Value::Kind::Int) {
        values.push_back(static_cast<double>(item.asInt()));
      } else if (item.kind() == Value::Kind::Float) {
        values.push_back(item.asFloat());
      } else {
        return false;
      }
    }
    return true;
  }

  // Every row must have the same shape.
  std::vector<std::int64_t> rowShape;
  for (const Value& row : list.asList()) {
    std::vector<std::int64_t> currentShape;
    if (!flattenList(row, currentShape, values)) return false;
    if (rowShape.empty()) {
      rowShape = currentShape;
    } else if (rowShape != currentShape) {
      return false;
    }
  }
  for (std::int64_t dim : rowShape) shape.push_back(dim);
  return true;
}

Value builtinTensor(Interpreter&, const std::vector<Value>& args,
                    const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"tensor takes one argument, a list of numbers", SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::List) {
    throw RuntimeError{"tensor takes a list of numbers", SourceSpan::start(), {}};
  }

  std::vector<std::int64_t> shape;
  std::vector<double> values;
  if (!flattenList(args[0], shape, values)) {
    throw RuntimeError{"tensor takes a list of numbers, or a list of lists of the same shape",
                       SourceSpan::start(), {}};
  }

  DType dtype = DType::F32;
  if (!args.empty() && args.back().kind() == Value::Kind::String) {
    if (!tensor::dtypeFromName(args.back().asString(), dtype)) {
      throw RuntimeError{"'" + args.back().asString() + "' is not a dtype",
                         SourceSpan::start(), {}};
    }
  }

  return makeTensor(std::make_shared<Tensor>(Tensor::fromDoubles(Shape(shape), values, dtype)));
}

Value builtinSeed(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1 || args[0].kind() != Value::Kind::Int) {
    throw RuntimeError{"seed needs one whole number", SourceSpan::start(), {}};
  }
  tensor::setRandomSeed(static_cast<std::uint32_t>(args[0].asInt()));
  return Value::null();
}

// --- elementwise ---------------------------------------------------------

Value builtinAdd(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"add takes two arguments", SourceSpan::start(), {}};
  }
  if (args[0].kind() == Value::Kind::Tensor && args[1].kind() == Value::Kind::Tensor) {
    return makeTensor(autograd::add(args[0].asTensor(), args[1].asTensor()));
  }
  if (args[0].kind() == Value::Kind::Tensor) {
    const double scalar = args[1].kind() == Value::Kind::Int
                              ? static_cast<double>(args[1].asInt())
                              : args[1].asFloat();
    return makeTensor(autograd::add(args[0].asTensor(), scalar));
  }
  throw RuntimeError{"add needs a tensor", SourceSpan::start(), {}};
}

Value builtinSub(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"sub takes two arguments", SourceSpan::start(), {}};
  }
  if (args[0].kind() == Value::Kind::Tensor && args[1].kind() == Value::Kind::Tensor) {
    return makeTensor(autograd::sub(args[0].asTensor(), args[1].asTensor()));
  }
  if (args[0].kind() == Value::Kind::Tensor) {
    const double scalar = args[1].kind() == Value::Kind::Int
                              ? static_cast<double>(args[1].asInt())
                              : args[1].asFloat();
    return makeTensor(autograd::sub(args[0].asTensor(), scalar));
  }
  throw RuntimeError{"sub needs a tensor", SourceSpan::start(), {}};
}

Value builtinMul(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"mul takes two arguments", SourceSpan::start(), {}};
  }
  if (args[0].kind() == Value::Kind::Tensor && args[1].kind() == Value::Kind::Tensor) {
    return makeTensor(autograd::mul(args[0].asTensor(), args[1].asTensor()));
  }
  if (args[0].kind() == Value::Kind::Tensor) {
    const double scalar = args[1].kind() == Value::Kind::Int
                              ? static_cast<double>(args[1].asInt())
                              : args[1].asFloat();
    return makeTensor(autograd::mul(args[0].asTensor(), scalar));
  }
  throw RuntimeError{"mul needs a tensor", SourceSpan::start(), {}};
}

Value builtinDiv(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"div takes two arguments", SourceSpan::start(), {}};
  }
  if (args[0].kind() == Value::Kind::Tensor && args[1].kind() == Value::Kind::Tensor) {
    return makeTensor(autograd::div(args[0].asTensor(), args[1].asTensor()));
  }
  if (args[0].kind() == Value::Kind::Tensor) {
    const double scalar = args[1].kind() == Value::Kind::Int
                              ? static_cast<double>(args[1].asInt())
                              : args[1].asFloat();
    return makeTensor(autograd::div(args[0].asTensor(), scalar));
  }
  throw RuntimeError{"div needs a tensor", SourceSpan::start(), {}};
}

Value builtinMatmul(Interpreter&, const std::vector<Value>& args,
                    const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"matmul takes two tensors", SourceSpan::start(), {}};
  }
  const auto a = wantTensor(args[0], "matmul");
  const auto b = wantTensor(args[1], "matmul");
  return makeTensor(autograd::matmul(a, b));
}

/// Build one of the unary tensor operations.
template <typename Fn>
Value unaryBuiltin(Fn fn, const std::vector<Value>& args, const char* name) {
  if (args.size() != 1) {
    throw RuntimeError{std::string(name) + " takes one tensor", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], name);
  return makeTensor(fn(tensor));
}

Value builtinRelu(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  return unaryBuiltin(
      [](const std::shared_ptr<tensor::Tensor>& t) { return autograd::relu(t); }, args, "relu");
}

Value builtinGelu(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  return unaryBuiltin(
      [](const std::shared_ptr<tensor::Tensor>& t) { return autograd::gelu(t); }, args, "gelu");
}

Value builtinSigmoid(Interpreter&, const std::vector<Value>& args,
                     const std::vector<std::pair<std::string, Value>>&) {
  return unaryBuiltin(
      [](const std::shared_ptr<tensor::Tensor>& t) { return autograd::sigmoid(t); }, args,
      "sigmoid");
}

Value builtinTanh(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  return unaryBuiltin(
      [](const std::shared_ptr<tensor::Tensor>& t) { return autograd::tanh(t); }, args, "tanh");
}

Value builtinExp(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  return unaryBuiltin(
      [](const std::shared_ptr<tensor::Tensor>& t) { return autograd::exp(t); }, args, "exp");
}

Value builtinLog(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  return unaryBuiltin(
      [](const std::shared_ptr<tensor::Tensor>& t) { return autograd::log(t); }, args, "log");
}

Value builtinSqrt(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  return unaryBuiltin(
      [](const std::shared_ptr<tensor::Tensor>& t) { return autograd::sqrt(t); }, args, "sqrt");
}

// --- reductions and shapes ------------------------------------------------

Value builtinMean(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"mean takes one tensor", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "mean");
  return makeTensor(autograd::mean(tensor));
}

Value builtinArgmax(Interpreter&, const std::vector<Value>& args,
                    const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"argmax takes one tensor", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "argmax");
  return makeTensor(autograd::argmax(tensor));
}

Value builtinSoftmax(Interpreter&, const std::vector<Value>& args,
                     const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"softmax takes one tensor", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "softmax");
  return makeTensor(autograd::softmax(tensor));
}

Value builtinReshape(Interpreter&, const std::vector<Value>& args,
                     const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() < 2) {
    throw RuntimeError{"reshape needs a tensor and a shape", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "reshape");
  const Shape shape = shapeFromArgs(args, 1);
  return makeTensor(autograd::reshape(tensor, shape));
}

Value builtinTranspose(Interpreter&, const std::vector<Value>& args,
                       const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"transpose takes one tensor", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "transpose");
  return makeTensor(autograd::transpose(tensor));
}

Value builtinContiguous(Interpreter&, const std::vector<Value>& args,
                        const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"contiguous takes one tensor", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "contiguous");
  return makeTensor(autograd::contiguous(tensor));
}

Value builtinShape(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"shape takes one tensor", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "shape");
  Value::List dims;
  for (const std::int64_t dim : tensor->shape().dims()) {
    dims.push_back(Value::integer(dim));
  }
  return Value::list(std::move(dims));
}

Value builtinDtype(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"dtype takes one tensor", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "dtype");
  return Value::string(tensor::dtypeName(tensor->dtype()));
}

Value builtinTo(Interpreter&, const std::vector<Value>& args,
                const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"to takes a tensor and a dtype", SourceSpan::start(), {}};
  }
  const auto tensor = wantTensor(args[0], "to");
  if (args[1].kind() != Value::Kind::String) {
    throw RuntimeError{"to needs a dtype name", SourceSpan::start(), {}};
  }
  DType dtype = DType::F32;
  if (!tensor::dtypeFromName(args[1].asString(), dtype)) {
    throw RuntimeError{"'" + args[1].asString() + "' is not a dtype", SourceSpan::start(), {}};
  }
  return makeTensor(std::make_shared<Tensor>(tensor->to(dtype)));
}

Value builtinSave(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"save takes a dictionary of tensors and a path", SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::Dict) {
    throw RuntimeError{"save takes a dictionary of tensors", SourceSpan::start(), {}};
  }
  if (args[1].kind() != Value::Kind::String) {
    throw RuntimeError{"save needs a path", SourceSpan::start(), {}};
  }
  const std::string path = args[1].asString();

  try {
    safetensors::Writer writer(path);
    for (const auto& entry : args[0].asDict()) {
      if (entry.second.kind() != Value::Kind::Tensor) {
        throw RuntimeError{"every value in the dictionary must be a tensor",
                           SourceSpan::start(), {}};
      }
      writer.add(entry.first, *entry.second.asTensor());
    }
    writer.write();
  } catch (const safetensors::Error& error) {
    throw RuntimeError{"I could not save '" + path + "': " + error.what(),
                       SourceSpan::start(), {}};
  }
  return Value::null();
}

Value builtinToDevice(Interpreter&, const std::vector<Value>& args,
                      const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 2) {
    throw RuntimeError{"to_device needs a tensor and a device name", SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::Tensor) {
    throw RuntimeError{"to_device needs a tensor", SourceSpan::start(), {}};
  }
  if (args[1].kind() != Value::Kind::String) {
    throw RuntimeError{"to_device needs a device name: 'cpu' or 'cuda'", SourceSpan::start(), {}};
  }

  const std::string name = args[1].asString();
  tensor::Device device = tensor::Device::Cpu;
  if (name == "cuda" || name == "gpu") {
    device = tensor::Device::Cuda;
  } else if (name != "cpu") {
    throw RuntimeError{"there is no device called '" + name + "'", SourceSpan::start(), {}};
  }

  if (device == tensor::Device::Cuda && tensor::backendFor(device) == nullptr) {
    throw RuntimeError{"this build has no GPU backend", SourceSpan::start(), {}};
  }

  try {
    return Value::tensor(
        std::make_shared<tensor::Tensor>(args[0].asTensor()->toDevice(device)));
  } catch (const std::exception& error) {
    throw RuntimeError{error.what(), SourceSpan::start(), {}};
  }
}

Value builtinDevice(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() != 1) {
    throw RuntimeError{"device needs one tensor", SourceSpan::start(), {}};
  }
  if (args[0].kind() != Value::Kind::Tensor) {
    throw RuntimeError{"device needs a tensor", SourceSpan::start(), {}};
  }
  const tensor::Device device = args[0].asTensor()->storage()->device();
  return Value::string(device == tensor::Device::Cuda ? "cuda" : "cpu");
}

Value builtinGpuInfo(Interpreter&, const std::vector<Value>& /*args*/,
                     const std::vector<std::pair<std::string, Value>>&) {
  Value::Dict info;
  tensor::Backend* backend = tensor::backendFor(tensor::Device::Cuda);
  if (backend == nullptr) {
    info.emplace_back("available", Value::boolean(false));
    return Value::dict(std::move(info));
  }
  info.emplace_back("available", Value::boolean(true));
  info.emplace_back("name", Value::string(backend->name()));
  info.emplace_back("total", Value::integer(static_cast<std::int64_t>(backend->totalMemory())));
  info.emplace_back("free", Value::integer(static_cast<std::int64_t>(backend->freeMemory())));
  return Value::dict(std::move(info));
}

}  // namespace

/// Install the tensor builtins. Called by `installBuiltins`.
void installTensorBuiltins(Interpreter& interpreter) {
  interpreter.defineBuiltin("zeros", &builtinZeros);
  interpreter.defineBuiltin("ones", &builtinOnes);
  interpreter.defineBuiltin("full", &builtinFull);
  interpreter.defineBuiltin("randn", &builtinRandn);
  interpreter.defineBuiltin("rand", &builtinRand);
  interpreter.defineBuiltin("arange", &builtinArange);
  interpreter.defineBuiltin("eye", &builtinEye);
  interpreter.defineBuiltin("tensor", &builtinTensor);
  interpreter.defineBuiltin("seed", &builtinSeed);

  interpreter.defineBuiltin("add", &builtinAdd);
  interpreter.defineBuiltin("sub", &builtinSub);
  interpreter.defineBuiltin("mul", &builtinMul);
  interpreter.defineBuiltin("div", &builtinDiv);
  interpreter.defineBuiltin("matmul", &builtinMatmul);

  interpreter.defineBuiltin("relu", &builtinRelu);
  interpreter.defineBuiltin("gelu", &builtinGelu);
  interpreter.defineBuiltin("sigmoid", &builtinSigmoid);
  interpreter.defineBuiltin("tanh", &builtinTanh);
  interpreter.defineBuiltin("exp", &builtinExp);
  interpreter.defineBuiltin("log", &builtinLog);
  interpreter.defineBuiltin("sqrt", &builtinSqrt);

  interpreter.defineBuiltin("mean", &builtinMean);
  interpreter.defineBuiltin("argmax", &builtinArgmax);
  interpreter.defineBuiltin("softmax", &builtinSoftmax);

  interpreter.defineBuiltin("reshape", &builtinReshape);
  interpreter.defineBuiltin("transpose", &builtinTranspose);
  interpreter.defineBuiltin("contiguous", &builtinContiguous);
  interpreter.defineBuiltin("shape", &builtinShape);
  interpreter.defineBuiltin("dtype", &builtinDtype);
  interpreter.defineBuiltin("to", &builtinTo);
  interpreter.defineBuiltin("to_device", &builtinToDevice);
  interpreter.defineBuiltin("device", &builtinDevice);
  interpreter.defineBuiltin("gpu_info", &builtinGpuInfo);
  interpreter.defineBuiltin("save", &builtinSave);
}

}  // namespace milo::runtime
