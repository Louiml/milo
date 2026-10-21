// The Milo abstract syntax tree.
//
// Nodes own their children through `std::unique_ptr` and carry a source
// span. The tree is deliberately close to the surface syntax: the
// desugaring of sentences into ordinary calls happens in the parser, so
// by the time the tree exists there is exactly one kind of statement.
#ifndef MILO_CORE_AST_HPP
#define MILO_CORE_AST_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/span.hpp"

namespace milo::ast {

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A type as written by the user.
struct TypeRef {
  /// The base name: `int`, `float`, `bool`, `str`, `bytes`, `any`, `null`,
  /// `tensor`, `model`, `dataset`, `module`, `list`, `dict`.
  std::string name;

  /// Tensor shape, if given. `has_shape` distinguishes `tensor` from
  /// `tensor[2, 3]`.
  bool has_shape = false;
  std::vector<std::int64_t> shape;

  /// Tensor element dtype, if given.
  std::string dtype;

  /// Element type of a `list`, or value type of a `dict`.
  std::shared_ptr<TypeRef> element;

  /// Value type of a `dict` (`dict[str, int]`).
  std::shared_ptr<TypeRef> value_type;

  std::string toString() const;
};

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

/// The kind of an expression node.
enum class ExprKind {
  Int,
  Float,
  String,
  Bool,
  Null,
  Name,
  List,
  Dict,
  Unary,
  Binary,
  Call,
  Index,
  Member,
  TypeTest,
};

struct Expr {
  SourceSpan span;
  virtual ~Expr() = default;

  virtual ExprKind kind() const = 0;

  /// A short description used in error messages.
  virtual std::string describe() const = 0;
};

struct IntLiteral : Expr {
  std::int64_t value = 0;
  ExprKind kind() const override { return ExprKind::Int; }
  std::string describe() const override;
};

struct FloatLiteral : Expr {
  double value = 0.0;
  ExprKind kind() const override { return ExprKind::Float; }
  std::string describe() const override;
};

struct StringLiteral : Expr {
  std::string value;
  ExprKind kind() const override { return ExprKind::String; }
  std::string describe() const override;
};

struct BoolLiteral : Expr {
  bool value = false;
  ExprKind kind() const override { return ExprKind::Bool; }
  std::string describe() const override;
};

struct NullLiteral : Expr {
  ExprKind kind() const override { return ExprKind::Null; }
  std::string describe() const override;
};

struct NameExpr : Expr {
  std::string name;
  ExprKind kind() const override { return ExprKind::Name; }
  std::string describe() const override;
};

struct ListExpr : Expr {
  std::vector<ExprPtr> elements;
  ExprKind kind() const override { return ExprKind::List; }
  std::string describe() const override;
};

struct DictExpr : Expr {
  std::vector<std::pair<ExprPtr, ExprPtr>> entries;
  ExprKind kind() const override { return ExprKind::Dict; }
  std::string describe() const override;
};

struct UnaryExpr : Expr {
  std::string op;  // "-", "+", "not"
  ExprPtr operand;
  ExprKind kind() const override { return ExprKind::Unary; }
  std::string describe() const override;
};

struct BinaryExpr : Expr {
  std::string op;  // "+", "-", "*", "/", "//", "%", "**", "==", "!=", ...
  ExprPtr lhs;
  ExprPtr rhs;
  ExprKind kind() const override { return ExprKind::Binary; }
  std::string describe() const override;
};

struct CallExpr : Expr {
  ExprPtr callee;
  std::vector<ExprPtr> args;
  std::vector<std::pair<std::string, ExprPtr>> kwargs;
  ExprKind kind() const override { return ExprKind::Call; }
  std::string describe() const override;
};

struct IndexExpr : Expr {
  ExprPtr target;
  std::vector<ExprPtr> indices;
  ExprKind kind() const override { return ExprKind::Index; }
  std::string describe() const override;
};

struct MemberExpr : Expr {
  ExprPtr target;
  std::string member;
  ExprKind kind() const override { return ExprKind::Member; }
  std::string describe() const override;
};

/// `value is tensor`
struct TypeTestExpr : Expr {
  ExprPtr operand;
  std::string type_name;
  ExprKind kind() const override { return ExprKind::TypeTest; }
  std::string describe() const override;
};

// ---------------------------------------------------------------------------
// Sentences
// ---------------------------------------------------------------------------

/// One option of a sentence: `key = value`.
struct SentenceOption {
  std::string key;
  ExprPtr value;
  SourceSpan span;
};

/// One clause of a sentence: a verb and the slots it filled.
struct SentenceClause {
  std::string verb;
  /// Slot name -> value, in the order the template declares them.
  std::vector<std::pair<std::string, ExprPtr>> slots;
  std::vector<SentenceOption> options;
  SourceSpan span;
};

/// A whole sentence: one or more clauses joined by `and`.
struct Sentence {
  std::vector<SentenceClause> clauses;
  std::vector<SentenceOption> options;
  SourceSpan span;
};

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

struct Stmt;
using StmtPtr = std::unique_ptr<Stmt>;

/// The kind of a statement node.
enum class StmtKind {
  Let,
  Assign,
  Expr,
  Return,
  Break,
  Continue,
  If,
  While,
  For,
  Fn,
  Import,
  Sentence,
};

struct Stmt {
  SourceSpan span;
  virtual ~Stmt() = default;

  virtual StmtKind kind() const = 0;
};

struct LetStmt : Stmt {
  std::string name;
  std::shared_ptr<TypeRef> type;
  ExprPtr value;  // may be null
  bool has_value = false;
  StmtKind kind() const override { return StmtKind::Let; }
};

struct AssignStmt : Stmt {
  std::string name;
  ExprPtr value;
  StmtKind kind() const override { return StmtKind::Assign; }
};

struct ExprStmt : Stmt {
  ExprPtr expr;
  StmtKind kind() const override { return StmtKind::Expr; }
};

struct ReturnStmt : Stmt {
  ExprPtr value;  // may be null
  StmtKind kind() const override { return StmtKind::Return; }
};

struct BreakStmt : Stmt {
  StmtKind kind() const override { return StmtKind::Break; }
};

struct ContinueStmt : Stmt {
  StmtKind kind() const override { return StmtKind::Continue; }
};

struct ImportStmt : Stmt {
  std::string module;
  std::string alias;
  std::vector<std::string> members;
  bool is_from = false;
  StmtKind kind() const override { return StmtKind::Import; }
};

struct IfStmt : Stmt {
  /// Condition/body pairs. The last entry is the `else` branch when
  /// `has_else` is set.
  std::vector<std::pair<ExprPtr, std::vector<StmtPtr>>> branches;
  bool has_else = false;
  std::vector<StmtPtr> else_body;
  StmtKind kind() const override { return StmtKind::If; }
};

struct WhileStmt : Stmt {
  ExprPtr condition;
  std::vector<StmtPtr> body;
  StmtKind kind() const override { return StmtKind::While; }
};

struct ForStmt : Stmt {
  std::string name;
  ExprPtr iterable;
  std::vector<StmtPtr> body;
  StmtKind kind() const override { return StmtKind::For; }
};

struct FunctionParam {
  std::string name;
  std::shared_ptr<TypeRef> type;
  SourceSpan span;
};

struct FnStmt : Stmt {
  std::string name;
  std::vector<FunctionParam> params;
  std::shared_ptr<TypeRef> return_type;
  std::vector<StmtPtr> body;
  StmtKind kind() const override { return StmtKind::Fn; }
};

struct SentenceStmt : Stmt {
  Sentence sentence;
  /// Statements written in the sentence's indented block. They run before
  /// the sentence's own action.
  std::vector<StmtPtr> body;
  StmtKind kind() const override { return StmtKind::Sentence; }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

}  // namespace milo::ast

#endif  // MILO_CORE_AST_HPP
