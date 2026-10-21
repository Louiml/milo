#include "runtime/interp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>

#include "core/lexer.hpp"
#include "autograd/ops.hpp"
#include "autograd/tape.hpp"
#include "python/python.hpp"
#include "runtime/builtins.hpp"
#include "runtime/modules.hpp"
#include "runtime/sentences.hpp"
#include "safetensors/reader.hpp"
#include "safetensors/writer.hpp"

namespace milo {
namespace {

using runtime::Value;

/// Control-flow signals, implemented as exceptions so that the statement
/// executor stays readable.
struct ReturnSignal {
  Value value;
};

struct BreakSignal {};

struct ContinueSignal {};

/// The maximum depth of nested calls. A runaway recursion should fail with
/// a clear message rather than a stack overflow.
constexpr int kMaxCallDepth = 512;

/// Compare two values for equality. Different types are never equal.
bool valuesEqual(const Value& lhs, const Value& rhs) {
  if (lhs.kind() == rhs.kind()) {
    switch (lhs.kind()) {
      case Value::Kind::Null:
        return true;
      case Value::Kind::Bool:
        return lhs.asBool() == rhs.asBool();
      case Value::Kind::Int:
        return lhs.asInt() == rhs.asInt();
      case Value::Kind::Float:
        return lhs.asFloat() == rhs.asFloat();
      case Value::Kind::Tensor:
        return lhs.asTensor() == rhs.asTensor();
      case Value::Kind::String:
        return lhs.asString() == rhs.asString();
      case Value::Kind::List:
        return lhs.asList().size() == rhs.asList().size() &&
               std::equal(lhs.asList().begin(), lhs.asList().end(), rhs.asList().begin(),
                          valuesEqual);
      case Value::Kind::Model:
        return lhs.asModel().path == rhs.asModel().path;
      case Value::Kind::Dataset:
        return lhs.asDataset().path == rhs.asDataset().path;
    case Value::Kind::Python:
      return lhs.asPythonObject().handle == rhs.asPythonObject().handle;
      case Value::Kind::Function:
        return lhs.asFunction() == rhs.asFunction();
      case Value::Kind::Builtin:
        return lhs.asBuiltin() == rhs.asBuiltin();
      case Value::Kind::Stub:
        return lhs.asStub().name == rhs.asStub().name;
      case Value::Kind::Dict:
        break;
    }
  }
  // An int and a float with the same value are equal; every other
  // cross-type comparison is false.
  if (lhs.kind() == Value::Kind::Int && rhs.kind() == Value::Kind::Float) {
    return static_cast<double>(lhs.asInt()) == rhs.asFloat();
  }
  if (lhs.kind() == Value::Kind::Float && rhs.kind() == Value::Kind::Int) {
    return lhs.asFloat() == static_cast<double>(rhs.asInt());
  }
  if (lhs.kind() == Value::Kind::Dict && rhs.kind() == Value::Kind::Dict) {
    if (lhs.asDict().size() != rhs.asDict().size()) return false;
    for (const auto& entry : lhs.asDict()) {
      const Value* other = rhs.find(entry.first);
      if (other == nullptr || !valuesEqual(entry.second, *other)) return false;
    }
    return true;
  }
  return false;
}

/// Ordering comparison. Returns false when the values are not comparable.
bool valuesCompare(const Value& lhs, const Value& rhs, int& result) {
  if (lhs.kind() == Value::Kind::Int && rhs.kind() == Value::Kind::Int) {
    result = lhs.asInt() < rhs.asInt() ? -1 : (lhs.asInt() > rhs.asInt() ? 1 : 0);
    return true;
  }
  if ((lhs.kind() == Value::Kind::Int || lhs.kind() == Value::Kind::Float) &&
      (rhs.kind() == Value::Kind::Int || rhs.kind() == Value::Kind::Float)) {
    const double a = lhs.kind() == Value::Kind::Int ? static_cast<double>(lhs.asInt())
                                                    : lhs.asFloat();
    const double b = rhs.kind() == Value::Kind::Int ? static_cast<double>(rhs.asInt())
                                                    : rhs.asFloat();
    result = a < b ? -1 : (a > b ? 1 : 0);
    return true;
  }
  if (lhs.kind() == Value::Kind::String && rhs.kind() == Value::Kind::String) {
    const int comparison = lhs.asString().compare(rhs.asString());
    result = comparison < 0 ? -1 : (comparison > 0 ? 1 : 0);
    return true;
  }
  return false;
}

/// Hand a scope over to a function that is leaving it.
///
/// A function closes over the scope it was written in, and that scope
/// holds the function: the two keep each other alive for ever, and the
/// scope is never given back. When the function leaves, it takes the scope
/// with it and the scope lets go of the function, so that neither of them
/// is kept alive by the other.
void releaseReturnedFunction(const Value& value,
                             const std::shared_ptr<runtime::Environment>& scope) {
  if (value.kind() != Value::Kind::Function) return;
  const auto& function = value.asFunction();
  auto closed = function->closure.lock();
  if (!closed || closed.get() != scope.get()) return;
  function->keepAlive = closed;
  scope->release(function->name);
}

}  // namespace

Interpreter::Interpreter() : globals_(std::make_shared<runtime::Environment>()) {
  output_ = &std::cout;
  notes_ = &std::cerr;
  runtime::installBuiltins(*this);
}

std::ostream& Interpreter::out() const { return output_ != nullptr ? *output_ : std::cout; }

std::ostream& Interpreter::notes() const { return notes_ != nullptr ? *notes_ : std::cerr; }

void Interpreter::defineBuiltin(
    const std::string& name,
    runtime::Value (*fn)(Interpreter&, const std::vector<runtime::Value>&,
                         const std::vector<std::pair<std::string, runtime::Value>>&)) {
  auto builtin = std::make_shared<runtime::BuiltinValue>();
  builtin->name = name;
  builtin->fn = fn;
  globals_->define(name, runtime::Value::builtin(std::move(builtin)));
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

void Interpreter::raise(const std::string& message, SourceSpan span) {
  throw RuntimeError{message, span, {}};
}

void Interpreter::raiseUnknownName(const std::string& name, SourceSpan span) {
  RuntimeError error;
  error.message = "I cannot find anything called '" + name + "'";
  error.span = span;

  std::vector<std::string> candidates = globals_->visibleNames();
  const std::vector<std::string>& keywords = Lexer::keywords();
  candidates.insert(candidates.end(), keywords.begin(), keywords.end());

  const std::string suggestion = closestMatch(name, candidates);
  if (!suggestion.empty()) {
    error.suggestion = "did you mean '" + suggestion + "'?";
  }
  throw error;
}

void Interpreter::raiseWrongType(const std::string& what, const Value& value, SourceSpan span) {
  raise(what + " but got a " + value.typeName(), span);
}

// ---------------------------------------------------------------------------
// Program execution
// ---------------------------------------------------------------------------

Value Interpreter::runProgram(const std::vector<ast::StmtPtr>& program) {
  return execute(program, globals_);
}

Value Interpreter::execute(const std::vector<ast::StmtPtr>& body,
                           const std::shared_ptr<runtime::Environment>& env) {
  Value last = Value::null();
  for (const ast::StmtPtr& statement : body) {
    last = exec(*statement, env);
  }
  return last;
}

Value Interpreter::exec(ast::Stmt& stmt, const std::shared_ptr<runtime::Environment>& env) {
  switch (stmt.kind()) {
    case ast::StmtKind::Let:
      execLet(static_cast<const ast::LetStmt&>(stmt), env);
      return Value::null();
    case ast::StmtKind::Assign:
      execAssign(static_cast<const ast::AssignStmt&>(stmt), env);
      return Value::null();
    case ast::StmtKind::Expr: {
      const auto& node = static_cast<const ast::ExprStmt&>(stmt);
      return eval(*node.expr, env);
    }
    case ast::StmtKind::Return: {
      const auto& node = static_cast<const ast::ReturnStmt&>(stmt);
      Value value = Value::null();
      if (node.value) value = eval(*node.value, env);
      throw ReturnSignal{std::move(value)};
    }
    case ast::StmtKind::Break:
      throw BreakSignal{};
    case ast::StmtKind::Continue:
      throw ContinueSignal{};
    case ast::StmtKind::If: {
      const auto& node = static_cast<const ast::IfStmt&>(stmt);
      for (const auto& branch : node.branches) {
        if (eval(*branch.first, env).isTruthy()) {
          return execute(branch.second, env);
        }
      }
      if (node.has_else) {
        return execute(node.else_body, env);
      }
      return Value::null();
    }
    case ast::StmtKind::While: {
      const auto& node = static_cast<const ast::WhileStmt&>(stmt);
      // A loop body shares the scope around it, so that a value it
      // computes is still there when the loop ends.
      const bool previous = env->allowRedeclaration();
      env->setAllowRedeclaration(true);
      while (eval(*node.condition, env).isTruthy()) {
        try {
          execute(node.body, env);
        } catch (const BreakSignal&) {
          break;
        } catch (const ContinueSignal&) {
          continue;
        }
      }
      env->setAllowRedeclaration(previous);
      return Value::null();
    }
    case ast::StmtKind::For: {
      const auto& node = static_cast<const ast::ForStmt&>(stmt);
      const Value iterable = eval(*node.iterable, env);
      std::vector<Value> items;
      if (iterable.kind() == Value::Kind::List) {
        items = iterable.asList();
      } else if (iterable.kind() == Value::Kind::String) {
        for (const char c : iterable.asString()) {
          items.push_back(Value::string(std::string(1, c)));
        }
      } else {
        raiseWrongType("I can only loop over a list or a string", iterable,
                       node.iterable->span);
      }

      const bool previous = env->allowRedeclaration();
      env->setAllowRedeclaration(true);
      for (Value& item : items) {
        env->define(node.name, item);
        try {
          execute(node.body, env);
        } catch (const BreakSignal&) {
          break;
        } catch (const ContinueSignal&) {
          continue;
        }
      }
      env->setAllowRedeclaration(previous);
      return Value::null();
    }
    case ast::StmtKind::Fn: {
      auto& node = static_cast<ast::FnStmt&>(stmt);
      auto function = std::make_shared<runtime::FunctionValue>();
      function->name = node.name;
      function->closure = env;
      function->body = std::move(node.body);
      function->return_type = node.return_type;
      for (const ast::FunctionParam& param : node.params) {
        function->params.push_back(param.name);
      }
      env->define(node.name, Value::function(function));
      return Value::null();
    }
    case ast::StmtKind::Import: {
      const auto& node = static_cast<const ast::ImportStmt&>(stmt);
      const std::string alias = node.alias.empty() ? node.module : node.alias;
      // An import names a Python module.
      try {
        env->define(alias, pythonContext().importModule(node.module));
      } catch (const std::exception& error) {
        raise(error.what(), stmt.span);
      }
      return Value::null();
    }
    case ast::StmtKind::Sentence:
      execSentence(static_cast<ast::SentenceStmt&>(stmt), env);
      return Value::null();
  }
  return Value::null();
}

void Interpreter::execLet(const ast::LetStmt& node,
                          const std::shared_ptr<runtime::Environment>& env) {
  if (env->findLocal(node.name) != nullptr && !env->allowRedeclaration()) {
    raise("'" + node.name + "' is already declared in this scope", node.span);
  }
  Value value = Value::null();
  if (node.has_value) {
    value = eval(*node.value, env);
  }
  env->define(node.name, std::move(value));
}

void Interpreter::execAssign(const ast::AssignStmt& node,
                             const std::shared_ptr<runtime::Environment>& env) {
  Value value = eval(*node.value, env);
  if (!env->assign(node.name, std::move(value))) {
    raiseUnknownName(node.name, node.span);
  }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

Interpreter::~Interpreter() = default;

python::Context& Interpreter::pythonContext() {
  if (!python_) {
    python_ = std::make_unique<python::Context>();
  }
  return *python_;
}

Value Interpreter::eval(const ast::Expr& expr, const std::shared_ptr<runtime::Environment>& env) {
  switch (expr.kind()) {
    case ast::ExprKind::Int:
      return Value::integer(static_cast<const ast::IntLiteral&>(expr).value);
    case ast::ExprKind::Float:
      return Value::floating(static_cast<const ast::FloatLiteral&>(expr).value);
    case ast::ExprKind::String:
      return Value::string(static_cast<const ast::StringLiteral&>(expr).value);
    case ast::ExprKind::Bool:
      return Value::boolean(static_cast<const ast::BoolLiteral&>(expr).value);
    case ast::ExprKind::Null:
      return Value::null();
    case ast::ExprKind::Name: {
      const auto& node = static_cast<const ast::NameExpr&>(expr);
      const Value* value = env->find(node.name);
      if (value == nullptr) raiseUnknownName(node.name, node.span);
      return *value;
    }
    case ast::ExprKind::List: {
      const auto& node = static_cast<const ast::ListExpr&>(expr);
      runtime::Value::List items;
      items.reserve(node.elements.size());
      for (const ast::ExprPtr& element : node.elements) {
        items.push_back(eval(*element, env));
      }
      return Value::list(std::move(items));
    }
    case ast::ExprKind::Dict: {
      const auto& node = static_cast<const ast::DictExpr&>(expr);
      runtime::Value::Dict entries;
      for (const auto& entry : node.entries) {
        const Value key = eval(*entry.first, env);
        entries.emplace_back(key.toDisplayString(), eval(*entry.second, env));
      }
      return Value::dict(std::move(entries));
    }
    case ast::ExprKind::Unary: {
      const auto& node = static_cast<const ast::UnaryExpr&>(expr);
      if (node.op == "not") {
        return Value::boolean(!eval(*node.operand, env).isTruthy());
      }
      const Value operand = eval(*node.operand, env);
      if (node.op == "-") {
        if (operand.kind() == Value::Kind::Int) return Value::integer(-operand.asInt());
        if (operand.kind() == Value::Kind::Float) return Value::floating(-operand.asFloat());
      }
      if (node.op == "+") return operand;
      raise("I cannot apply '" + node.op + "' to a " + operand.typeName(), node.span);
    }
    case ast::ExprKind::Binary:
      return evalBinary(static_cast<const ast::BinaryExpr&>(expr), env);
    case ast::ExprKind::Call:
      return evalCall(static_cast<const ast::CallExpr&>(expr), env);
    case ast::ExprKind::Index:
      return evalIndex(static_cast<const ast::IndexExpr&>(expr), env);
    case ast::ExprKind::Member:
      return evalMember(static_cast<const ast::MemberExpr&>(expr), env);
    case ast::ExprKind::TypeTest: {
      const auto& node = static_cast<const ast::TypeTestExpr&>(expr);
      const Value operand = eval(*node.operand, env);
      return typeTestOperation(operand, node.type_name);
    }
  }
  return Value::null();
}

Value Interpreter::typeTestOperation(const Value& operand, const std::string& type) {
  if (type == "int") return Value::boolean(operand.kind() == Value::Kind::Int);
  if (type == "float") return Value::boolean(operand.kind() == Value::Kind::Float);
  if (type == "bool") return Value::boolean(operand.kind() == Value::Kind::Bool);
  if (type == "str") return Value::boolean(operand.kind() == Value::Kind::String);
  if (type == "list") return Value::boolean(operand.kind() == Value::Kind::List);
  if (type == "dict") return Value::boolean(operand.kind() == Value::Kind::Dict);
  if (type == "tensor") {
    return Value::boolean(operand.kind() == Value::Kind::Tensor);
  }
  if (type == "model") {
    return Value::boolean(operand.kind() == Value::Kind::Model);
  }
  if (type == "dataset") return Value::boolean(operand.kind() == Value::Kind::Dataset);
  if (type == "any") return Value::boolean(true);
  if (type == "null") return Value::boolean(operand.kind() == Value::Kind::Null);
  return Value::boolean(false);
}

Value Interpreter::evalBinary(const ast::BinaryExpr& node,
                              const std::shared_ptr<runtime::Environment>& env) {
  const std::string& op = node.op;

  if (op == "and") {
    const Value lhs = eval(*node.lhs, env);
    if (!lhs.isTruthy()) return lhs;
    return eval(*node.rhs, env);
  }
  if (op == "or") {
    const Value lhs = eval(*node.lhs, env);
    if (lhs.isTruthy()) return lhs;
    return eval(*node.rhs, env);
  }

  const Value lhs = eval(*node.lhs, env);
  const Value rhs = eval(*node.rhs, env);
  return binaryOperation(op, lhs, rhs, node.span);
}

Value Interpreter::binaryOperation(const std::string& op, const Value& lhs, const Value& rhs,
                                   SourceSpan span) {
  // A tensor on either side makes this a tensor operation.
  if (lhs.kind() == Value::Kind::Tensor || rhs.kind() == Value::Kind::Tensor) {
    return tensorBinary(op, lhs, rhs, span);
  }

  if (op == "==") return Value::boolean(valuesEqual(lhs, rhs));
  if (op == "!=") return Value::boolean(!valuesEqual(lhs, rhs));

  if (op == "<" || op == "<=" || op == ">" || op == ">=") {
    int comparison = 0;
    if (!valuesCompare(lhs, rhs, comparison)) {
      raise("I cannot compare a " + std::string(lhs.typeName()) + " with a " +
                std::string(rhs.typeName()),
            span);
    }
    if (op == "<") return Value::boolean(comparison < 0);
    if (op == "<=") return Value::boolean(comparison <= 0);
    if (op == ">") return Value::boolean(comparison > 0);
    return Value::boolean(comparison >= 0);
  }

  const bool lhsIsInt = lhs.kind() == Value::Kind::Int;
  const bool rhsIsInt = rhs.kind() == Value::Kind::Int;
  const bool bothNumeric = (lhsIsInt || lhs.kind() == Value::Kind::Float) &&
                           (rhsIsInt || rhs.kind() == Value::Kind::Float);

  if (op == "+" && lhs.kind() == Value::Kind::String && rhs.kind() == Value::Kind::String) {
    return Value::string(lhs.asString() + rhs.asString());
  }
  if (op == "+" && lhs.kind() == Value::Kind::List && rhs.kind() == Value::Kind::List) {
    runtime::Value::List items = lhs.asList();
    items.insert(items.end(), rhs.asList().begin(), rhs.asList().end());
    return Value::list(std::move(items));
  }

  if (!bothNumeric) {
    raise("I cannot apply '" + op + "' to a " + std::string(lhs.typeName()) + " and a " +
              std::string(rhs.typeName()),
          span);
  }

  if (lhsIsInt && rhsIsInt) {
    const std::int64_t a = lhs.asInt();
    const std::int64_t b = rhs.asInt();
    if (op == "+") return Value::integer(a + b);
    if (op == "-") return Value::integer(a - b);
    if (op == "*") return Value::integer(a * b);
    if (op == "/") {
      if (b == 0) raise("I cannot divide by zero", span);
      return Value::floating(static_cast<double>(a) / static_cast<double>(b));
    }
    if (op == "//") {
      if (b == 0) raise("I cannot divide by zero", span);
      const std::int64_t quotient = a / b;
      return Value::integer((a % b != 0) && ((a < 0) != (b < 0)) ? quotient - 1 : quotient);
    }
    if (op == "%") {
      if (b == 0) raise("I cannot divide by zero", span);
      const std::int64_t remainder = a % b;
      return Value::integer(remainder != 0 && ((remainder < 0) != (b < 0)) ? remainder + b
                                                                         : remainder);
    }
    if (op == "**") {
      if (b < 0) {
        return Value::floating(std::pow(static_cast<double>(a), static_cast<double>(b)));
      }
      std::int64_t result = 1;
      for (std::int64_t i = 0; i < b; ++i) result *= a;
      return Value::integer(result);
    }
  }

  const double a = lhs.kind() == Value::Kind::Int ? static_cast<double>(lhs.asInt()) : lhs.asFloat();
  const double b = rhs.kind() == Value::Kind::Int ? static_cast<double>(rhs.asInt()) : rhs.asFloat();
  if (op == "+") return Value::floating(a + b);
  if (op == "-") return Value::floating(a - b);
  if (op == "*") return Value::floating(a * b);
  if (op == "/") {
    if (b == 0.0) raise("I cannot divide by zero", span);
    return Value::floating(a / b);
  }
  if (op == "//") {
    if (b == 0.0) raise("I cannot divide by zero", span);
    return Value::floating(std::floor(a / b));
  }
  if (op == "%") {
    if (b == 0.0) raise("I cannot divide by zero", span);
    return Value::floating(std::fmod(a, b));
  }
  if (op == "**") return Value::floating(std::pow(a, b));

  raise("I do not know the operator '" + op + "'", span);
}

Value Interpreter::evalCall(const ast::CallExpr& node,
                            const std::shared_ptr<runtime::Environment>& env) {
  const Value callee = eval(*node.callee, env);

  std::vector<Value> args;
  args.reserve(node.args.size());
  for (const ast::ExprPtr& argument : node.args) {
    args.push_back(eval(*argument, env));
  }

  std::vector<std::pair<std::string, Value>> kwargs;
  kwargs.reserve(node.kwargs.size());
  for (const auto& keyword : node.kwargs) {
    kwargs.emplace_back(keyword.first, eval(*keyword.second, env));
  }

  if (callee.kind() == Value::Kind::Stub) {
    // A placeholder for something that does not exist yet.
    return Value::null();
  }

  return call(callee, args, kwargs);
}

Value Interpreter::call(Value callee, const std::vector<Value>& args,
                        const std::vector<std::pair<std::string, Value>>& kwargs) {
  Value out = Value::null();
  if (callSpecialValue(callee, args, kwargs, out)) {
    return out;
  }

  if (callee.kind() == Value::Kind::Function) {
    if (depth_ >= kMaxCallDepth) {
      raise("this call went too deep; is a function calling itself without end?",
            SourceSpan::start());
    }
    const auto& function = callee.asFunction();

    auto scope = std::make_shared<runtime::Environment>(function->closure.lock());
    for (std::size_t i = 0; i < function->params.size(); ++i) {
      if (i < args.size()) {
        scope->define(function->params[i], args[i]);
      } else {
        scope->define(function->params[i], Value::null());
      }
    }
    // The function can call itself.
    scope->define(function->name, callee);

    ++depth_;
    try {
      Value result = Value::null();
      for (const ast::StmtPtr& statement : function->body) {
        result = exec(*statement, scope);
      }
      releaseReturnedFunction(result, scope);
      --depth_;
      return result;
    } catch (const ReturnSignal& signal) {
      releaseReturnedFunction(signal.value, scope);
      --depth_;
      return signal.value;
    } catch (...) {
      --depth_;
      throw;
    }
  }

  if (callee.kind() == Value::Kind::Model || callee.kind() == Value::Kind::Dataset ||
      callee.kind() == Value::Kind::Python) {
    raise(std::string("I cannot call a ") + callee.typeName(), SourceSpan::start());
  }

  raise("this is not something I can call", SourceSpan::start());
}

bool Interpreter::callSpecialValue(const Value& callee, const std::vector<Value>& args,
                                   const std::vector<std::pair<std::string, Value>>& kwargs,
                                   Value& out) {
  if (callee.kind() == Value::Kind::Builtin) {
    out = callee.asBuiltin()->fn(*this, args, kwargs);
    return true;
  }

  if (callee.kind() == Value::Kind::Stub) {
    // A placeholder for something that does not exist yet.
    out = Value::null();
    return true;
  }

  if (callee.kind() == Value::Kind::Module) {
    // Calling a layer runs its forward pass.
    if (args.size() != 1) {
      raise("a layer takes one input", SourceSpan::start());
    }
    out = moduleForward(callee.asModule(), args.front());
    return true;
  }

  if (callee.kind() == Value::Kind::Optimizer) {
    // Calling an optimiser takes one step.
    optimizerStep(const_cast<runtime::OptimizerValue&>(callee.asOptimizer()));
    out = Value::null();
    return true;
  }

  if (callee.kind() == Value::Kind::Python) {
    // Calling a Python object.
    try {
      out = pythonContext().callObject(callee, args, kwargs);
    } catch (const std::exception& error) {
      raise(error.what(), SourceSpan::start());
    }
    return true;
  }

  return false;
}

Value Interpreter::evalMember(const ast::MemberExpr& node,
                              const std::shared_ptr<runtime::Environment>& env) {
  const Value target = eval(*node.target, env);
  return memberOperation(target, node.member, node.span);
}

Value Interpreter::memberOperation(const Value& target, const std::string& name, SourceSpan span) {
  switch (target.kind()) {
    case Value::Kind::Python: {
      // A Python object: read one of its attributes.
      try {
        return pythonContext().getAttribute(target, name);
      } catch (const std::exception& error) {
        raise(error.what(), span);
      }
      break;
    }
    default:
      break;
  }

  switch (target.kind()) {
    case Value::Kind::Model: {
      const auto& model = target.asModel();
      if (name == "name") return Value::string(model.name);
      if (name == "path") return Value::string(model.path);
      if (name == "kind") return Value::string(model.kind);
      if (name == "parameters") return Value::integer(model.parameterCount());
      if (name == "tensors") {
        runtime::Value::Dict entries;
        if (model.tensors) {
          for (const auto& entry : *model.tensors) {
            entries.emplace_back(entry.first, Value::tensor(entry.second));
          }
        }
        return Value::dict(std::move(entries));
      }
      if (name == "metadata") {
        runtime::Value::Dict entries;
        for (const auto& entry : model.metadata) {
          entries.emplace_back(entry.first, Value::string(entry.second));
        }
        return Value::dict(std::move(entries));
      }
      break;
    }
    case Value::Kind::Dataset: {
      const auto& dataset = target.asDataset();
      if (name == "name") return Value::string(dataset.name);
      if (name == "path") return Value::string(dataset.path);
      if (name == "kind") return Value::string(dataset.kind);
      break;
    }
    case Value::Kind::Python: {
      const auto& object = target.asPythonObject();
      return Value::stub(runtime::StubValue{object.name + "." + name});
    }
    case Value::Kind::Module: {
      const auto& module = target.asModule();
      if (name == "kind") return Value::string(module.kind);
      if (name == "parameters") return moduleParameters(module);
      if (name == "buffers") {
        runtime::Value::Dict entries;
        for (const auto& entry : module.buffers) {
          entries.emplace_back(entry.first, Value::tensor(entry.second));
        }
        return Value::dict(std::move(entries));
      }
      break;
    }
    case Value::Kind::Optimizer: {
      // The optimiser's methods are bound to the optimiser itself.
      const auto optimizer = std::make_shared<runtime::OptimizerValue>(target.asOptimizer());
      if (name == "kind") return Value::string(optimizer->kind);
      if (name == "step") {
        auto method = std::make_shared<runtime::BuiltinValue>();
        method->name = "step";
        method->fn = [optimizer](Interpreter&, const std::vector<Value>&,
                                const std::vector<std::pair<std::string, Value>>&) {
          optimizerStep(*optimizer);
          return Value::null();
        };
        return Value::builtin(method);
      }
      if (name == "zero_grad") {
        auto method = std::make_shared<runtime::BuiltinValue>();
        method->name = "zero_grad";
        method->fn = [optimizer](Interpreter&, const std::vector<Value>&,
                                const std::vector<std::pair<std::string, Value>>&) {
          optimizerZeroGrad(*optimizer);
          return Value::null();
        };
        return Value::builtin(method);
      }
      if (name == "learning_rate") {
        return Value::floating(optimizer->learning_rate);
      }
      if (name == "step_count") {
        return Value::integer(optimizer->state->step);
      }
      break;
    }
    case Value::Kind::Quantized: {
      const auto& quantized = *target.asQuantized();
      if (name == "scheme") return Value::string(quantized.scheme);
      if (name == "shape") {
        runtime::Value::List dims;
        for (const std::int64_t dim : quantized.shape.dims()) {
          dims.push_back(Value::integer(dim));
        }
        return Value::list(std::move(dims));
      }
      if (name == "bytes") {
        return Value::integer(static_cast<std::int64_t>(quantized.byteSize()));
      }
      if (name == "original_dtype") {
        return Value::string(tensor::dtypeName(quantized.original));
      }
      if (name == "block_size") {
        return Value::integer(quantized.blockSize);
      }
      break;
    }
    case Value::Kind::Tensor: {
      const auto& tensor = *target.asTensor();
      if (name == "backward") {
        // `loss.backward()` runs the backward pass.
        const auto held = target.asTensor();
        auto method = std::make_shared<runtime::BuiltinValue>();
        method->name = "backward";
        method->fn = [held](Interpreter&, const std::vector<Value>&,
                           const std::vector<std::pair<std::string, Value>>&) {

          autograd::Tape::current().backward(held);
          return Value::null();
        };
        return Value::builtin(method);
      }
      if (name == "grad") {
        if (!tensor.grad()) return Value::null();
        return Value::tensor(tensor.grad());
      }
      if (name == "requires_grad") {
        return Value::boolean(tensor.requiresGrad());
      }
      if (name == "shape") {
        runtime::Value::List dims;
        for (const std::int64_t dim : tensor.shape().dims()) {
          dims.push_back(Value::integer(dim));
        }
        return Value::list(std::move(dims));
      }
      if (name == "dtype") {
        return Value::string(tensor::dtypeName(tensor.dtype()));
      }
      if (name == "rank") {
        return Value::integer(static_cast<std::int64_t>(tensor.rank()));
      }
      if (name == "size") {
        return Value::integer(tensor.numel());
      }
      if (name == "strides") {
        runtime::Value::List strides;
        for (const std::int64_t stride : tensor.strides().values()) {
          strides.push_back(Value::integer(stride));
        }
        return Value::list(std::move(strides));
      }
      if (name == "contiguous") {
        return Value::boolean(tensor.isContiguous());
      }
      break;
    }
    case Value::Kind::String:
      if (name == "length") {
        return Value::integer(static_cast<std::int64_t>(target.asString().size()));
      }
      break;
    case Value::Kind::List:
      if (name == "length") {
        return Value::integer(static_cast<std::int64_t>(target.asList().size()));
      }
      break;
    default:
      break;
  }

  // Models and datasets expose their members as placeholders until the
  // real implementations arrive.
  if (target.kind() == Value::Kind::Model || target.kind() == Value::Kind::Dataset) {
    return Value::stub(runtime::StubValue{std::string(target.typeName()) + "." + name});
  }

  raise("a " + std::string(target.typeName()) + " does not have a field called '" + name +
            "'",
        span);
}

Value Interpreter::evalIndex(const ast::IndexExpr& node,
                             const std::shared_ptr<runtime::Environment>& env) {
  const Value target = eval(*node.target, env);
  std::vector<Value> indices;
  indices.reserve(node.indices.size());
  for (const ast::ExprPtr& index : node.indices) {
    indices.push_back(eval(*index, env));
  }
  return indexOperation(target, indices, node.span);
}

Value Interpreter::unaryOperation(const std::string& op, const Value& value, SourceSpan span) {
  if (op == "not") {
    return Value::boolean(!value.isTruthy());
  }
  if (op == "+") {
    return value;
  }
  if (op == "-") {
    if (value.kind() == Value::Kind::Int) return Value::integer(-value.asInt());
    if (value.kind() == Value::Kind::Float) return Value::floating(-value.asFloat());
  }
  raise("I cannot apply '" + op + "' to a " + value.typeName(), span);
}

Value Interpreter::indexOperation(const Value& target, const std::vector<Value>& indices,
                                 SourceSpan span) {
  if (target.kind() == Value::Kind::Tensor) {
    return indexTensor(target.asTensor(), indices, span);
  }

  if (indices.size() != 1) {
    raise("I can only index with one position for now", span);
  }

  const Value index = indices.front();
  if (target.kind() == Value::Kind::List) {
    if (index.kind() != Value::Kind::Int) {
      raiseWrongType("a list index must be a whole number", index, span);
    }
    const auto& items = target.asList();
    std::int64_t position = index.asInt();
    if (position < 0) position += static_cast<std::int64_t>(items.size());
    if (position < 0 || position >= static_cast<std::int64_t>(items.size())) {
      raise("that position is past the end of the list", span);
    }
    return items[static_cast<std::size_t>(position)];
  }

  if (target.kind() == Value::Kind::String) {
    if (index.kind() != Value::Kind::Int) {
      raiseWrongType("a string index must be a whole number", index, span);
    }
    const auto& text = target.asString();
    std::int64_t position = index.asInt();
    if (position < 0) position += static_cast<std::int64_t>(text.size());
    if (position < 0 || position >= static_cast<std::int64_t>(text.size())) {
      raise("that position is past the end of the string", span);
    }
    return Value::string(std::string(1, text[static_cast<std::size_t>(position)]));
  }

  if (target.kind() == Value::Kind::Dict) {
    const std::string key = index.toDisplayString();
    const Value* found = target.find(key);
    if (found == nullptr) {
      raise("there is no key called '" + key + "' in this dictionary", span);
    }
    return *found;
  }

  raiseWrongType("I cannot index a " + std::string(target.typeName()), target, span);
}

// ---------------------------------------------------------------------------
// Tensor indexing
// ---------------------------------------------------------------------------

Value Interpreter::indexTensor(const std::shared_ptr<tensor::Tensor>& tensor,
                               const std::vector<Value>& indexValues, SourceSpan span) {
  const auto& shape = tensor->shape();

  std::vector<std::int64_t> indices;
  indices.reserve(indexValues.size());
  for (const Value& indexValue : indexValues) {
    if (indexValue.kind() != Value::Kind::Int) {
      raiseWrongType("a tensor index must be a whole number", indexValue, span);
    }
    std::int64_t position = indexValue.asInt();
    if (position < 0) position += shape.dim(indices.size());
    if (position < 0 || position >= shape.dim(indices.size())) {
      raise("that position is past the end of the tensor", span);
    }
    indices.push_back(position);
  }

  if (indices.size() > shape.rank()) {
    raise("that index has more dimensions than the tensor", span);
  }

  if (indices.size() == shape.rank()) {
    // A single element.
    return Value::floating(tensor->get(indices));
  }

  // A sub-view: slice along the axes that were named.
  tensor::Tensor view = *tensor;
  for (std::size_t axis = 0; axis < indices.size(); ++axis) {
    view = view.slice(axis, indices[axis], indices[axis] + 1);
  }
  return Value::tensor(std::make_shared<tensor::Tensor>(std::move(view)));
}

// ---------------------------------------------------------------------------
// Tensor operators
// ---------------------------------------------------------------------------

namespace {

/// A number on the right of a tensor operator.
double numberOf(const Value& value) {
  if (value.kind() == Value::Kind::Int) return static_cast<double>(value.asInt());
  if (value.kind() == Value::Kind::Float) return value.asFloat();
  return 0.0;
}

}  // namespace

Value Interpreter::tensorBinary(const std::string& op, const Value& lhs, const Value& rhs,
                               SourceSpan span) {
  const bool lhsIsTensor = lhs.kind() == Value::Kind::Tensor;
  const bool rhsIsTensor = rhs.kind() == Value::Kind::Tensor;

  if (lhsIsTensor && rhsIsTensor) {
    const auto a = lhs.asTensor();
    const auto b = rhs.asTensor();
    if (op == "+") return Value::tensor(autograd::add(a, b));
    if (op == "-") return Value::tensor(autograd::sub(a, b));
    if (op == "*") return Value::tensor(autograd::mul(a, b));
    if (op == "/") return Value::tensor(autograd::div(a, b));
    if (op == "==" || op == "!=" || op == "<" || op == "<=" || op == ">" || op == ">=") {
      // A comparison of two tensors gives a tensor of answers, which is not
      // yet a condition. A single-element tensor compares to a number.
      if (a->numel() == 1 && b->numel() == 1) {
        const double x = a->getFlat(0);
        const double y = b->getFlat(0);
        if (op == "==") return Value::boolean(x == y);
        if (op == "!=") return Value::boolean(x != y);
        if (op == "<") return Value::boolean(x < y);
        if (op == "<=") return Value::boolean(x <= y);
        if (op == ">") return Value::boolean(x > y);
        return Value::boolean(x >= y);
      }
      raise("comparing two tensors gives a tensor, which cannot be a condition yet", span);
    }
    raise("I cannot apply '" + op + "' to two tensors", span);
  }

  if (lhsIsTensor) {
    const auto a = lhs.asTensor();
    const double number = numberOf(rhs);
    if (op == "+") return Value::tensor(autograd::add(a, number));
    if (op == "-") return Value::tensor(autograd::sub(a, number));
    if (op == "*") return Value::tensor(autograd::mul(a, number));
    if (op == "/") return Value::tensor(autograd::div(a, number));
    if (op == "==" || op == "!=" || op == "<" || op == "<=" || op == ">" || op == ">=") {
      if (a->numel() != 1) {
        raise("comparing a tensor gives a tensor, which cannot be a condition yet", span);
      }
      const double x = a->getFlat(0);
      if (op == "==") return Value::boolean(x == number);
      if (op == "!=") return Value::boolean(x != number);
      if (op == "<") return Value::boolean(x < number);
      if (op == "<=") return Value::boolean(x <= number);
      if (op == ">") return Value::boolean(x > number);
      return Value::boolean(x >= number);
    }
    raise("I cannot apply '" + op + "' to a tensor", span);
  }

  // The tensor is on the right.
  const auto b = rhs.asTensor();
  const double number = numberOf(lhs);
  if (op == "+") return Value::tensor(autograd::add(b, number));
  if (op == "-") return Value::tensor(autograd::sub(b, -number));
  if (op == "*") return Value::tensor(autograd::mul(b, number));
  if (op == "/") return Value::tensor(autograd::div(b, number));
  raise("I cannot apply '" + op + "' to a tensor", span);
}

// ---------------------------------------------------------------------------
// Sentences
// ---------------------------------------------------------------------------
Value Interpreter::evalOptions(const std::vector<ast::SentenceOption>& clause_options,
                               const std::vector<ast::SentenceOption>& sentence_options,
                               const std::shared_ptr<runtime::Environment>& env) {
  std::map<std::string, Value> merged;
  for (const ast::SentenceOption& option : clause_options) {
    merged[option.key] = eval(*option.value, env);
  }
  // Sentence-wide options were written later, so they win.
  for (const ast::SentenceOption& option : sentence_options) {
    merged[option.key] = eval(*option.value, env);
  }

  runtime::Value::Dict result;
  result.reserve(merged.size());
  for (const auto& entry : merged) {
    result.emplace_back(entry.first, entry.second);
  }
  return Value::dict(std::move(result));
}

void Interpreter::execSentence(ast::SentenceStmt& node,
                               const std::shared_ptr<runtime::Environment>& env) {
  // Statements written in the sentence's block run first, so that they can
  // adjust settings before the action happens.
  for (const ast::StmtPtr& statement : node.body) {
    exec(*statement, env);
  }

  for (const ast::SentenceClause& clause : node.sentence.clauses) {
    execSentenceClause(clause, env, node.sentence.options);
  }
}

void Interpreter::execSentenceClause(const ast::SentenceClause& clause,
                                     const std::shared_ptr<runtime::Environment>& env,
                                     const std::vector<ast::SentenceOption>& extra_options) {
  // The slots and the options are gathered into one place, so that the
  // bytecode machine can do exactly the same thing.
  runtime::SentenceValues values;

  for (const auto& entry : clause.slots) {
    runtime::Value value;
    switch (entry.second->kind()) {
      case ast::ExprKind::String:
        value = runtime::Value::string(static_cast<const ast::StringLiteral&>(*entry.second).value);
        break;
      case ast::ExprKind::Int:
        value = runtime::Value::integer(static_cast<const ast::IntLiteral&>(*entry.second).value);
        break;
      case ast::ExprKind::Float:
        value = runtime::Value::floating(static_cast<const ast::FloatLiteral&>(*entry.second).value);
        break;
      case ast::ExprKind::Name:
        // A slot holds a name, not a reference to one.
        value = runtime::Value::string(static_cast<const ast::NameExpr&>(*entry.second).name);
        break;
      default:
        raise("this sentence cannot be run", clause.span);
        return;
    }
    values.slots.emplace_back(entry.first, std::move(value));
  }

  // The clause's own options, then the sentence-wide ones, which win.
  for (const ast::SentenceOption& option : clause.options) {
    values.options.emplace_back(option.key, eval(*option.value, env));
  }
  for (const ast::SentenceOption& option : extra_options) {
    bool replaced = false;
    for (auto& existing : values.options) {
      if (existing.first == option.key) {
        existing.second = eval(*option.value, env);
        replaced = true;
        break;
      }
    }
    if (!replaced) {
      values.options.emplace_back(option.key, eval(*option.value, env));
    }
  }

  runtime::executeSentence(*this, clause.verb, clause.span, values, env);
}
}  // namespace milo