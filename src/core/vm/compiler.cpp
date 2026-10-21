#include "core/vm/compiler.hpp"

#include <utility>

namespace milo::vm {
namespace {

/// The instruction a binary operator becomes.
bool binaryInstruction(const std::string& op, Op& out) {
  if (op == "+") out = Op::Add;
  else if (op == "-") out = Op::Sub;
  else if (op == "*") out = Op::Mul;
  else if (op == "/") out = Op::Div;
  else if (op == "%") out = Op::Mod;
  else if (op == "**") out = Op::Pow;
  else if (op == "==") out = Op::Eq;
  else if (op == "!=") out = Op::Ne;
  else if (op == "<") out = Op::Lt;
  else if (op == "<=") out = Op::Le;
  else if (op == ">") out = Op::Gt;
  else if (op == ">=") out = Op::Ge;
  else return false;
  return true;
}

/// The value of a literal, as a runtime value.
bool literalValue(const ast::Expr& expression, runtime::Value& out) {
  switch (expression.kind()) {
    case ast::ExprKind::String:
      out = runtime::Value::string(static_cast<const ast::StringLiteral&>(expression).value);
      return true;
    case ast::ExprKind::Int:
      out = runtime::Value::integer(static_cast<const ast::IntLiteral&>(expression).value);
      return true;
    case ast::ExprKind::Float:
      out = runtime::Value::floating(static_cast<const ast::FloatLiteral&>(expression).value);
      return true;
    case ast::ExprKind::Name:
      // A sentence's slot holds a name, not a reference to one.
      out = runtime::Value::string(static_cast<const ast::NameExpr&>(expression).name);
      return true;
    default:
      return false;
  }
}

}  // namespace

std::uint32_t Chunk::addConstant(runtime::Value value) {
  for (std::size_t i = 0; i < constants.size(); ++i) {
    if (constants[i].kind() == value.kind() &&
        constants[i].toReprString() == value.toReprString()) {
      return static_cast<std::uint32_t>(i);
    }
  }
  constants.push_back(std::move(value));
  return static_cast<std::uint32_t>(constants.size() - 1);
}

std::uint32_t Chunk::addName(const std::string& name) {
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) return static_cast<std::uint32_t>(i);
  }
  names.push_back(name);
  return static_cast<std::uint32_t>(names.size() - 1);
}

std::uint32_t Chunk::addKeywordGroup(std::vector<std::string> group) {
  keywordGroups.push_back(std::move(group));
  return static_cast<std::uint32_t>(keywordGroups.size());
}

// ---------------------------------------------------------------------------
// The compiler
// ---------------------------------------------------------------------------

bool Compiler::compile(const std::vector<ast::StmtPtr>& statements, Program& out,
                        DiagnosticBag& diagnostics) {
  Compiler compiler;
  compiler.program_ = &out;
  compiler.diagnostics_ = &diagnostics;
  compiler.chunk_ = &out.main;

  for (const ast::StmtPtr& statement : statements) {
    compiler.compileStatement(*statement);
  }
  // The program always leaves a value behind.
  compiler.emit(Op::Null);
  compiler.emit(Op::Return);

  return !diagnostics.hasErrors();
}

void Compiler::emit(Op op, std::uint32_t a, std::uint32_t b) {
  chunk_->code.push_back(Instruction{op, a, b});
}

std::size_t Compiler::emitJump(Op op) {
  chunk_->code.push_back(Instruction{op, 0, 0});
  return chunk_->code.size() - 1;
}

void Compiler::patchJump(std::size_t place) {
  chunk_->code[place].a = static_cast<std::uint32_t>(chunk_->code.size());
}

void Compiler::compileBlock(const std::vector<ast::StmtPtr>& body) {
  for (const ast::StmtPtr& statement : body) {
    compileStatement(*statement);
  }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

void Compiler::compileStatement(const ast::Stmt& statement) {
  switch (statement.kind()) {
    case ast::StmtKind::Let: {
      const auto& node = static_cast<const ast::LetStmt&>(statement);
      if (node.has_value) {
        compileExpression(*node.value);
      } else {
        emit(Op::Null);
        ++depth_;
      }
      emit(Op::DefineName, chunk_->addName(node.name));
      --depth_;
      return;
    }
    case ast::StmtKind::Assign: {
      const auto& node = static_cast<const ast::AssignStmt&>(statement);
      compileExpression(*node.value);
      emit(Op::StoreName, chunk_->addName(node.name));
      --depth_;
      return;
    }
    case ast::StmtKind::Expr: {
      const auto& node = static_cast<const ast::ExprStmt&>(statement);
      compileExpression(*node.expr);
      emit(Op::Pop);
      --depth_;
      return;
    }
    case ast::StmtKind::Return: {
      const auto& node = static_cast<const ast::ReturnStmt&>(statement);
      if (node.value) {
        compileExpression(*node.value);
      } else {
        emit(Op::Null);
        ++depth_;
      }
      emit(Op::Return);
      --depth_;
      return;
    }
    case ast::StmtKind::Break: {
      if (loops_.empty()) {
        diagnostics_->error("there is no loop to break out of", statement.span);
        return;
      }
      loops_.back().breakPatches.push_back(emitJump(Op::Jump));
      return;
    }
    case ast::StmtKind::Continue: {
      if (loops_.empty()) {
        diagnostics_->error("there is no loop to continue", statement.span);
        return;
      }
      Loop& loop = loops_.back();
      if (loop.continueTarget == 0 && !loop.continuePatches.empty()) {
        // The target is not known yet; patch it when the loop closes.
        loop.continuePatches.push_back(emitJump(Op::Jump));
      } else {
        emit(Op::Jump, static_cast<std::uint32_t>(loop.continueTarget));
      }
      return;
    }
    case ast::StmtKind::If:
      compileIf(static_cast<const ast::IfStmt&>(statement));
      return;
    case ast::StmtKind::While:
      compileWhile(static_cast<const ast::WhileStmt&>(statement));
      return;
    case ast::StmtKind::For:
      compileFor(static_cast<const ast::ForStmt&>(statement));
      return;
    case ast::StmtKind::Fn:
      compileFunction(static_cast<const ast::FnStmt&>(statement));
      return;
    case ast::StmtKind::Import: {
      const auto& node = static_cast<const ast::ImportStmt&>(statement);
      const std::string alias = node.alias.empty() ? node.module : node.alias;
      // The import statement names a Python module; the machine cannot hold
      // a live Python object in its constant pool, so the import is done
      // at run time by the runtime.
      emit(Op::Import, chunk_->addName(node.module));
      ++depth_;
      emit(Op::DefineName, chunk_->addName(alias));
      --depth_;
      return;
    }
    case ast::StmtKind::Sentence:
      compileSentence(static_cast<const ast::SentenceStmt&>(statement));
      return;
  }
}

void Compiler::compileIf(const ast::IfStmt& statement) {
  std::vector<std::size_t> endPatches;

  for (const auto& branch : statement.branches) {
    compileExpression(*branch.first);
    --depth_;
    const std::size_t otherwise = emitJump(Op::JumpIfFalse);
    compileBlock(branch.second);
    endPatches.push_back(emitJump(Op::Jump));
    patchJump(otherwise);
  }

  if (statement.has_else) {
    compileBlock(statement.else_body);
  }

  for (const std::size_t place : endPatches) {
    patchJump(place);
  }
}

void Compiler::compileWhile(const ast::WhileStmt& statement) {
  const std::size_t start = chunk_->code.size();

  compileExpression(*statement.condition);
  --depth_;
  const std::size_t otherwise = emitJump(Op::JumpIfFalse);
  // The body may declare the same name on every turn.
  emit(Op::AllowRedeclaration);

  Loop loop;
  loop.continueTarget = start;
  loops_.push_back(loop);

  compileBlock(statement.body);

  emit(Op::Jump, static_cast<std::uint32_t>(start));
  patchJump(otherwise);
  emit(Op::DisallowRedeclaration);

  Loop& open = loops_.back();
  for (const std::size_t place : open.breakPatches) {
    patchJump(place);
  }
  for (const std::size_t place : open.continuePatches) {
    chunk_->code[place].a = static_cast<std::uint32_t>(open.continueTarget);
  }
  loops_.pop_back();
}

void Compiler::compileFor(const ast::ForStmt& statement) {
  compileExpression(*statement.iterable);
  --depth_;
  emit(Op::GetIter);
  ++depth_;

  const std::size_t start = chunk_->code.size();
  const std::size_t endJump = emitJump(Op::ForIter);

  // The item comes off the stack and becomes the loop's variable.
  emit(Op::DefineVar, chunk_->addName(statement.name));
  --depth_;
  // The body may declare the same name on every turn.
  emit(Op::AllowRedeclaration);

  Loop loop;
  loop.continueTarget = start;
  loops_.push_back(loop);

  compileBlock(statement.body);

  emit(Op::Jump, static_cast<std::uint32_t>(start));
  patchJump(endJump);
  emit(Op::DisallowRedeclaration);

  Loop& open = loops_.back();
  for (const std::size_t place : open.breakPatches) {
    patchJump(place);
  }
  for (const std::size_t place : open.continuePatches) {
    chunk_->code[place].a = static_cast<std::uint32_t>(open.continueTarget);
  }
  loops_.pop_back();
}

void Compiler::compileFunction(const ast::FnStmt& statement) {
  auto function = std::make_shared<Function>();
  function->name = statement.name;
  function->arity = static_cast<std::uint32_t>(statement.params.size());
  for (const ast::FunctionParam& param : statement.params) {
    function->params.push_back(param.name);
  }

  Chunk* savedChunk = chunk_;
  chunk_ = &function->chunk;

  // The parameters are bound by the machine, which knows their names.

  compileBlock(statement.body);
  emit(Op::Return);

  chunk_ = savedChunk;

  program_->functions.push_back(function);
  emit(Op::Closure, static_cast<std::uint32_t>(program_->functions.size() - 1));
  ++depth_;
  // A function definition replaces any binding of the same name, so that a
  // program may shadow a builtin.
  emit(Op::DefineVar, chunk_->addName(statement.name));
  --depth_;
}

void Compiler::compileSentence(const ast::SentenceStmt& statement) {
  for (const ast::SentenceClause& clause : statement.sentence.clauses) {
    // The slots of a sentence are names and literals, so they become
    // constants.
    for (const auto& entry : clause.slots) {
      runtime::Value value;
      if (!literalValue(*entry.second, value)) {
        diagnostics_->error("this sentence cannot be compiled", entry.second->span);
        return;
      }
      emit(Op::Constant, chunk_->addConstant(std::move(value)));
      ++depth_;
    }

    // The options are literals too.
    for (const ast::SentenceOption& option : clause.options) {
      runtime::Value value;
      if (!literalValue(*option.value, value)) {
        diagnostics_->error("this option cannot be compiled", option.value->span);
        return;
      }
      emit(Op::Constant, chunk_->addConstant(std::move(value)));
      ++depth_;
    }

    CompiledSentence compiled;
    compiled.verb = clause.verb;
    for (const auto& entry : clause.slots) {
      compiled.slotNames.push_back(entry.first);
    }
    for (const ast::SentenceOption& option : clause.options) {
      compiled.optionKeys.push_back(option.key);
    }

    program_->sentences.push_back(std::move(compiled));
    emit(Op::Sentence, static_cast<std::uint32_t>(program_->sentences.size() - 1),
         static_cast<std::uint32_t>(clause.slots.size()));
    depth_ -= static_cast<int>(clause.slots.size() + clause.options.size());
  }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

void Compiler::compileExpression(const ast::Expr& expression) {
  switch (expression.kind()) {
    case ast::ExprKind::Int:
      emit(Op::Constant,
           chunk_->addConstant(runtime::Value::integer(
               static_cast<const ast::IntLiteral&>(expression).value)));
      ++depth_;
      return;
    case ast::ExprKind::Float:
      emit(Op::Constant,
           chunk_->addConstant(runtime::Value::floating(
               static_cast<const ast::FloatLiteral&>(expression).value)));
      ++depth_;
      return;
    case ast::ExprKind::String:
      emit(Op::Constant,
           chunk_->addConstant(runtime::Value::string(
               static_cast<const ast::StringLiteral&>(expression).value)));
      ++depth_;
      return;
    case ast::ExprKind::Bool:
      emit(static_cast<const ast::BoolLiteral&>(expression).value ? Op::True : Op::False);
      ++depth_;
      return;
    case ast::ExprKind::Null:
      emit(Op::Null);
      ++depth_;
      return;
    case ast::ExprKind::Name:
      emit(Op::LoadName, chunk_->addName(static_cast<const ast::NameExpr&>(expression).name));
      ++depth_;
      return;
    case ast::ExprKind::List: {
      const auto& node = static_cast<const ast::ListExpr&>(expression);
      for (const ast::ExprPtr& element : node.elements) {
        compileExpression(*element);
      }
      depth_ -= static_cast<int>(node.elements.size());
      emit(Op::MakeList, static_cast<std::uint32_t>(node.elements.size()));
      ++depth_;
      return;
    }
    case ast::ExprKind::Dict: {
      const auto& node = static_cast<const ast::DictExpr&>(expression);
      for (const auto& entry : node.entries) {
        compileExpression(*entry.first);
        compileExpression(*entry.second);
      }
      depth_ -= static_cast<int>(node.entries.size() * 2);
      emit(Op::MakeDict, static_cast<std::uint32_t>(node.entries.size()));
      ++depth_;
      return;
    }
    case ast::ExprKind::Unary: {
      const auto& node = static_cast<const ast::UnaryExpr&>(expression);
      compileExpression(*node.operand);
      if (node.op == "not") {
        emit(Op::Not);
      } else if (node.op == "-") {
        emit(Op::Neg);
      }
      return;
    }
    case ast::ExprKind::Binary: {
      const auto& node = static_cast<const ast::BinaryExpr&>(expression);
      if (node.op == "and" || node.op == "or") {
        // Short-circuit evaluation, with Python's rule that the result is
        // the value that decided the answer.
        compileExpression(*node.lhs);
        const std::size_t otherwise =
            emitJump(node.op == "and" ? Op::JumpIfFalseKeep : Op::JumpIfTrueKeep);
        emit(Op::Pop);
        --depth_;
        compileExpression(*node.rhs);
        const std::size_t end = emitJump(Op::Jump);
        patchJump(otherwise);
        patchJump(end);
        ++depth_;
        return;
      }
      compileExpression(*node.lhs);
      compileExpression(*node.rhs);
      depth_ -= 2;
      Op op = Op::Nop;
      if (binaryInstruction(node.op, op)) {
        emit(op);
      } else {
        diagnostics_->error("I do not know the operator '" + node.op + "'", expression.span);
      }
      ++depth_;
      return;
    }
    case ast::ExprKind::Call: {
      const auto& node = static_cast<const ast::CallExpr&>(expression);
      compileExpression(*node.callee);
      for (const ast::ExprPtr& argument : node.args) {
        compileExpression(*argument);
      }
      for (const auto& keyword : node.kwargs) {
        compileExpression(*keyword.second);
      }
      const std::uint32_t count =
          static_cast<std::uint32_t>(node.args.size() + node.kwargs.size());
      depth_ -= static_cast<int>(count);

      // The names of the keyword arguments, so that the machine can put
      // them back where they belong.
      std::uint32_t group = 0;
      if (!node.kwargs.empty()) {
        std::vector<std::string> names;
        names.reserve(node.kwargs.size());
        for (const auto& keyword : node.kwargs) {
          names.push_back(keyword.first);
        }
        group = chunk_->addKeywordGroup(std::move(names));
      }

      emit(Op::Call, count, group);
      ++depth_;
      return;
    }
    case ast::ExprKind::Index: {
      const auto& node = static_cast<const ast::IndexExpr&>(expression);
      compileExpression(*node.target);
      for (const ast::ExprPtr& index : node.indices) {
        compileExpression(*index);
      }
      depth_ -= static_cast<int>(node.indices.size());
      emit(Op::Index, static_cast<std::uint32_t>(node.indices.size()));
      ++depth_;
      return;
    }
    case ast::ExprKind::Member: {
      const auto& node = static_cast<const ast::MemberExpr&>(expression);
      compileExpression(*node.target);
      emit(Op::GetAttr, chunk_->addName(node.member));
      return;
    }
    case ast::ExprKind::TypeTest: {
      const auto& node = static_cast<const ast::TypeTestExpr&>(expression);
      compileExpression(*node.operand);
      emit(Op::TypeTest, chunk_->addName(node.type_name));
      return;
    }
  }
}

}  // namespace milo::vm
