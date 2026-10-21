#include "runtime/builtins.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "autograd/ops.hpp"
#include "tensor/ops.hpp"

namespace milo::runtime {
namespace {

using Value = runtime::Value;

/// The signature every builtin shares.
using Builtin = Value (*)(Interpreter&, const std::vector<Value>&,
                           const std::vector<std::pair<std::string, Value>>&);

std::size_t expectArgs(const std::vector<Value>& args, std::size_t count, const char* name) {
  if (args.size() != count) {
    std::ostringstream out;
    out << name << " takes " << count << (count == 1 ? " argument" : " arguments") << ", not "
        << args.size();
    throw RuntimeError{out.str(), SourceSpan::start(), {}};
  }
  return count;
}

Value builtinPrint(Interpreter& interpreter, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  if (args.size() > 1) {
    throw RuntimeError{"print takes one argument, or a list of them", SourceSpan::start(), {}};
  }
  if (args.size() == 1) {
    // A list prints as its repr, so that `print [1, 2]` shows the brackets.
    interpreter.out() << args[0].toDisplayString() << '\n';
  }
  return Value::null();
}

Value builtinLen(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "len");
  const Value& value = args.front();
  switch (value.kind()) {
    case Value::Kind::String:
      return Value::integer(static_cast<std::int64_t>(value.asString().size()));
    case Value::Kind::List:
      return Value::integer(static_cast<std::int64_t>(value.asList().size()));
    case Value::Kind::Dict:
      return Value::integer(static_cast<std::int64_t>(value.asDict().size()));
    default:
      throw RuntimeError{"len does not work on a " + std::string(value.typeName()),
                         SourceSpan::start(), {}};
  }
}

Value builtinRange(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  if (args.empty() || args.size() > 3) {
    throw RuntimeError{"range takes one, two, or three arguments", SourceSpan::start(), {}};
  }
  for (const Value& argument : args) {
    if (argument.kind() != Value::Kind::Int) {
      throw RuntimeError{"range works with whole numbers only", SourceSpan::start(), {}};
    }
  }

  std::int64_t start = 0;
  std::int64_t stop = 0;
  std::int64_t step = 1;
  if (args.size() == 1) {
    stop = args[0].asInt();
  } else {
    start = args[0].asInt();
    stop = args[1].asInt();
    if (args.size() == 3) step = args[2].asInt();
  }
  if (step == 0) {
    throw RuntimeError{"range cannot step by zero", SourceSpan::start(), {}};
  }

  Value::List items;
  if (step > 0) {
    for (std::int64_t i = start; i < stop; i += step) items.push_back(Value::integer(i));
  } else {
    for (std::int64_t i = start; i > stop; i += step) items.push_back(Value::integer(i));
  }
  return Value::list(std::move(items));
}

Value builtinStr(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "str");
  return Value::string(args.front().toDisplayString());
}

Value builtinInt(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "int");
  const Value& value = args.front();
  if (value.kind() == Value::Kind::Int) return value;
  if (value.kind() == Value::Kind::Float) return Value::integer(static_cast<std::int64_t>(value.asFloat()));
  if (value.kind() == Value::Kind::Bool) return Value::integer(value.asBool() ? 1 : 0);
  if (value.kind() == Value::Kind::String) {
    try {
      std::size_t consumed = 0;
      const int parsed = std::stoi(value.asString(), &consumed);
      if (consumed == value.asString().size()) return Value::integer(parsed);
    } catch (const std::exception&) {
      // fall through to the error below
    }
  }
  throw RuntimeError{"int cannot turn a " + std::string(value.typeName()) + " into a whole number",
                     SourceSpan::start(), {}};
}

Value builtinFloat(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "float");
  const Value& value = args.front();
  if (value.kind() == Value::Kind::Float) return value;
  if (value.kind() == Value::Kind::Int) return Value::floating(static_cast<double>(value.asInt()));
  if (value.kind() == Value::Kind::Bool) return Value::floating(value.asBool() ? 1.0 : 0.0);
  if (value.kind() == Value::Kind::String) {
    try {
      std::size_t consumed = 0;
      const double parsed = std::stod(value.asString(), &consumed);
      if (consumed == value.asString().size()) return Value::floating(parsed);
    } catch (const std::exception&) {
      // fall through to the error below
    }
  }
  throw RuntimeError{"float cannot turn a " + std::string(value.typeName()) + " into a number",
                     SourceSpan::start(), {}};
}

Value builtinBool(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "bool");
  return Value::boolean(args.front().isTruthy());
}

Value builtinAbs(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "abs");
  const Value& value = args.front();
  if (value.kind() == Value::Kind::Int) return Value::integer(std::abs(value.asInt()));
  if (value.kind() == Value::Kind::Float) return Value::floating(std::fabs(value.asFloat()));
  throw RuntimeError{"abs does not work on a " + std::string(value.typeName()),
                     SourceSpan::start(), {}};
}

Value builtinRound(Interpreter&, const std::vector<Value>& args,
                   const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "round");
  const Value& value = args.front();
  if (value.kind() == Value::Kind::Tensor) {
    // Round every element.
    const auto tensor = value.asTensor();
    auto out = std::make_shared<tensor::Tensor>(tensor::Tensor::zeros(tensor->dtype(),
                                                                    tensor->shape()));
    const std::int64_t total = tensor->numel();
    std::vector<std::int64_t> indices(tensor->shape().rank(), 0);
    for (std::int64_t i = 0; i < total; ++i) {
      out->setFlat(i, static_cast<double>(
                          static_cast<std::int64_t>(std::llround(tensor->getFlat(i)))));
      for (std::size_t axis = tensor->shape().rank(); axis-- > 0;) {
        if (++indices[axis] < tensor->shape().dim(axis)) break;
        indices[axis] = 0;
      }
    }
    return Value::tensor(out);
  }
  if (value.kind() == Value::Kind::Int) return value;
  if (value.kind() == Value::Kind::Float) {
    return Value::integer(static_cast<std::int64_t>(std::llround(value.asFloat())));
  }
  throw RuntimeError{"round does not work on a " + std::string(value.typeName()),
                     SourceSpan::start(), {}};
}

Value builtinSum(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "sum");
  const Value& value = args.front();
  if (value.kind() == Value::Kind::Tensor) {
    return Value::tensor(std::make_shared<tensor::Tensor>(*autograd::sum(value.asTensor())));
  }
  if (value.kind() != Value::Kind::List) {
    throw RuntimeError{"sum works on a list or a tensor", SourceSpan::start(), {}};
  }
  bool allInts = true;
  double total = 0.0;
  std::int64_t integerTotal = 0;
  for (const Value& item : value.asList()) {
    if (item.kind() == Value::Kind::Int) {
      integerTotal += item.asInt();
      total += static_cast<double>(item.asInt());
    } else if (item.kind() == Value::Kind::Float) {
      allInts = false;
      total += item.asFloat();
    } else {
      throw RuntimeError{"sum works on a list of numbers", SourceSpan::start(), {}};
    }
  }
  return allInts ? Value::integer(integerTotal) : Value::floating(total);
}

Value builtinMin(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "min");
  const Value& value = args.front();
  if (value.kind() == Value::Kind::Tensor) {
    return Value::tensor(std::make_shared<tensor::Tensor>(*autograd::min(value.asTensor())));
  }
  if (value.kind() != Value::Kind::List || value.asList().empty()) {
    throw RuntimeError{"min works on a list with at least one number",
                       SourceSpan::start(), {}};
  }
  Value best = value.asList().front();
  for (const Value& item : value.asList()) {
    if (item.kind() == Value::Kind::Int && best.kind() == Value::Kind::Float) {
      if (static_cast<double>(item.asInt()) < best.asFloat()) best = item;
    } else if (best.kind() == Value::Kind::Int && item.kind() == Value::Kind::Float) {
      if (item.asFloat() < static_cast<double>(best.asInt())) best = item;
    } else if (item.kind() == Value::Kind::Int && best.kind() == Value::Kind::Int) {
      if (item.asInt() < best.asInt()) best = item;
    } else if (item.kind() == Value::Kind::Float && best.kind() == Value::Kind::Float) {
      if (item.asFloat() < best.asFloat()) best = item;
    }
  }
  return best;
}

Value builtinMax(Interpreter&, const std::vector<Value>& args,
                 const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "max");
  const Value& value = args.front();
  if (value.kind() == Value::Kind::Tensor) {
    return Value::tensor(std::make_shared<tensor::Tensor>(*autograd::max(value.asTensor())));
  }
  if (value.kind() != Value::Kind::List || value.asList().empty()) {
    throw RuntimeError{"max works on a list with at least one number",
                       SourceSpan::start(), {}};
  }
  Value best = value.asList().front();
  for (const Value& item : value.asList()) {
    if (item.kind() == Value::Kind::Int && best.kind() == Value::Kind::Float) {
      if (static_cast<double>(item.asInt()) > best.asFloat()) best = item;
    } else if (best.kind() == Value::Kind::Int && item.kind() == Value::Kind::Float) {
      if (item.asFloat() > static_cast<double>(best.asInt())) best = item;
    } else if (item.kind() == Value::Kind::Int && best.kind() == Value::Kind::Int) {
      if (item.asInt() > best.asInt()) best = item;
    } else if (item.kind() == Value::Kind::Float && best.kind() == Value::Kind::Float) {
      if (item.asFloat() > best.asFloat()) best = item;
    }
  }
  return best;
}

Value builtinSorted(Interpreter&, const std::vector<Value>& args,
                    const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "sorted");
  const Value& value = args.front();
  if (value.kind() != Value::Kind::List) {
    throw RuntimeError{"sorted works on a list", SourceSpan::start(), {}};
  }
  Value::List items = value.asList();
  std::stable_sort(items.begin(), items.end(), [](const Value& a, const Value& b) {
    if (a.kind() == Value::Kind::Int && b.kind() == Value::Kind::Int) {
      return a.asInt() < b.asInt();
    }
    if (a.kind() == Value::Kind::String && b.kind() == Value::Kind::String) {
      return a.asString() < b.asString();
    }
    if (a.kind() == Value::Kind::Float && b.kind() == Value::Kind::Float) {
      return a.asFloat() < b.asFloat();
    }
    return false;
  });
  return Value::list(std::move(items));
}

Value builtinType(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "type");
  return Value::string(args.front().typeName());
}

Value builtinCopy(Interpreter&, const std::vector<Value>& args,
                  const std::vector<std::pair<std::string, Value>>&) {
  expectArgs(args, 1, "copy");
  const Value& value = args.front();
  if (value.kind() == Value::Kind::List) {
    return Value::list(value.asList());
  }
  if (value.kind() == Value::Kind::Dict) {
    return Value::dict(value.asDict());
  }
  return value;
}

}  // namespace

void installBuiltins(Interpreter& interpreter) {
  interpreter.defineBuiltin("print", &builtinPrint);
  interpreter.defineBuiltin("len", &builtinLen);
  interpreter.defineBuiltin("range", &builtinRange);
  interpreter.defineBuiltin("str", &builtinStr);
  interpreter.defineBuiltin("int", &builtinInt);
  interpreter.defineBuiltin("float", &builtinFloat);
  interpreter.defineBuiltin("bool", &builtinBool);
  interpreter.defineBuiltin("abs", &builtinAbs);
  interpreter.defineBuiltin("round", &builtinRound);
  interpreter.defineBuiltin("sum", &builtinSum);
  interpreter.defineBuiltin("min", &builtinMin);
  interpreter.defineBuiltin("max", &builtinMax);
  interpreter.defineBuiltin("sorted", &builtinSorted);
  interpreter.defineBuiltin("type", &builtinType);
  interpreter.defineBuiltin("copy", &builtinCopy);

  installTensorBuiltins(interpreter);
  installTrainingBuiltins(interpreter);
}

}  // namespace milo::runtime
