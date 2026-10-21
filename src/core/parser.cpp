#include "core/parser.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

#include "core/directives.hpp"
#include "core/lexer.hpp"

namespace milo {
namespace {

/// Is `text` a hard keyword? Hard keywords may not be used as names.
bool isHardKeyword(const std::string& text) { return Lexer::isKeyword(text); }

std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

/// The words a verb template expects, for error messages.
std::string shapeFor(const core::VerbSpec& spec) {
  const auto& parts = spec.alternatives.front();
  std::string out = spec.verb;
  for (const core::TemplatePart& part : parts) {
    out += ' ';
    if (part.literal) {
      out += part.literal_text;
    } else if (part.slot == core::SlotKind::Kind) {
      out += "<";
      for (std::size_t i = 0; i < part.allowed.size(); ++i) {
        if (i != 0) out += "|";
        out += part.allowed[i];
      }
      out += ">";
    } else if (part.slot == core::SlotKind::Path) {
      out += "\"path\"";
    } else if (part.slot == core::SlotKind::Number) {
      out += "<number>";
    } else {
      out += "<name>";
    }
  }
  return out;
}

}  // namespace

Parser::Parser(std::vector<Token> tokens, std::string filename)
    : tokens_(std::move(tokens)), filename_(std::move(filename)) {
  if (tokens_.empty()) {
    Token end(TokenKind::EndOfFile, SourceSpan::start());
    tokens_.push_back(std::move(end));
  }
}

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

const Token& Parser::peek(std::size_t offset) const {
  const std::size_t index = index_ + offset;
  if (index >= tokens_.size()) return tokens_.back();
  return tokens_[index];
}

const Token& Parser::previous() const {
  if (index_ == 0) return tokens_.front();
  return tokens_[index_ - 1];
}

bool Parser::atEnd() const { return peek().kind == TokenKind::EndOfFile; }

bool Parser::check(TokenKind kind) const { return peek().kind == kind; }

bool Parser::checkAt(std::size_t offset, TokenKind kind) const {
  return peek(offset).kind == kind;
}

bool Parser::match(TokenKind kind) {
  if (!check(kind)) return false;
  ++index_;
  return true;
}

const Token& Parser::advance() {
  if (!atEnd()) ++index_;
  return previous();
}

bool Parser::isWordAt(std::size_t offset, const std::string& word) const {
  const Token& token = peek(offset);
  if (token.kind != TokenKind::Identifier) return false;
  return toLower(token.text) == toLower(word);
}

bool Parser::isWord(const std::string& word) const { return isWordAt(0, word); }

bool Parser::matchWord(const std::string& word) {
  if (!isWord(word)) return false;
  ++index_;
  return true;
}

Token Parser::expect(TokenKind kind, const std::string& what) {
  if (check(kind)) return advance();
  error("expected " + what + " but found " + peek().describe(), peek().span);
  synchronizeToLineEnd();
  return Token(TokenKind::Unknown, peek().span);
}

std::string Parser::expectIdentifier(const std::string& what) {
  if (check(TokenKind::Identifier) && !isHardKeyword(peek().text)) {
    return advance().text;
  }
  if (check(TokenKind::Identifier) && isHardKeyword(peek().text)) {
    error("`" + peek().text + "` is a keyword, so it cannot be used as " + what,
          peek().span);
    synchronizeToLineEnd();
    return {};
  }
  error("expected " + what + " but found " + peek().describe(), peek().span);
  synchronizeToLineEnd();
  return {};
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

void Parser::error(const std::string& message, SourceSpan span) {
  if (error_count_ > 0) return;  // one error at a time keeps messages useful
  ++error_count_;
  diagnostics_.error(message, span);
}

void Parser::errorWithSuggestion(const std::string& message, SourceSpan span,
                                  const std::string& word,
                                  const std::vector<std::string>& candidates) {
  if (error_count_ > 0) return;
  ++error_count_;
  Diagnostic diagnostic(Severity::Error, message, span);
  const std::string suggestion = closestMatch(word, candidates);
  if (!suggestion.empty()) {
    diagnostic.label = "did you mean `" + suggestion + "`?";
  }
  diagnostics_.add(std::move(diagnostic));
}

void Parser::synchronizeToLineEnd() {
  while (!atEnd() && !check(TokenKind::Newline) && !check(TokenKind::Indent) &&
         !check(TokenKind::Dedent)) {
    ++index_;
  }
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

std::vector<ast::StmtPtr> Parser::parseProgram() {
  std::vector<ast::StmtPtr> program;

  while (!atEnd() && error_count_ == 0) {
    if (match(TokenKind::Newline)) continue;
    if (check(TokenKind::Dedent)) {
      error("this block is closed but nothing opened it", peek().span);
      ++index_;
      continue;
    }
    ast::StmtPtr statement = parseStatement();
    if (!statement) break;
    program.push_back(std::move(statement));
  }

  return program;
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

ast::StmtPtr Parser::parseStatement() {
  if (check(TokenKind::Identifier)) {
    // A sentence begins with a verb — but a verb can also be the name of
    // a function, as in `save(weights, "out.safetensors")`. Only treat it
    // as a sentence when the verb's template actually fits.
    if (core::findVerb(peek().text) != nullptr && canParseSentence()) {
      return parseSentence();
    }
  }

  if (isWord("let")) return parseLet();
  if (isWord("set")) return parseSet();
  if (isWord("if")) return parseIf();
  if (isWord("while")) return parseWhile();
  if (isWord("for")) return parseFor();
  if (isWord("fn") || isWord("function")) return parseFn();
  if (isWord("return")) return parseReturn();
  if (isWord("break")) return parseBreak();
  if (isWord("continue")) return parseContinue();
  if (isWord("import")) return parseImport();
  if (isWord("from")) return parseFromImport();

  return parseAssignmentOrExpression();
}

ast::StmtPtr Parser::parseLet() {
  const Token& start = advance();  // `let`

  auto statement = std::make_unique<ast::LetStmt>();
  statement->span = start.span;
  statement->name = expectIdentifier("a variable name");

  if (match(TokenKind::Colon)) {
    statement->type = parseType();
  }

  if (isWord("be") || check(TokenKind::Equal)) {
    advance();
    statement->value = parseExpression();
    statement->has_value = true;
  }

  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseSet() {
  const Token& start = advance();  // `set`

  auto statement = std::make_unique<ast::AssignStmt>();
  statement->span = start.span;
  statement->name = expectIdentifier("a variable name");
  matchWord("to");
  statement->value = parseExpression();
  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseAssignmentOrExpression() {
  // `name = value`
  if (check(TokenKind::Identifier) && checkAt(1, TokenKind::Equal) &&
      !isHardKeyword(peek().text)) {
    auto statement = std::make_unique<ast::AssignStmt>();
    statement->span = peek().span;
    statement->name = advance().text;
    advance();  // `=`
    statement->value = parseExpression();
    statement->span = statement->span.join(previous().span);
    return statement;
  }

  // `print value` — the one builtin that may be called without brackets.
  if (isWord("print") && !checkAt(1, TokenKind::LeftParen)) {
    const Token& start = advance();  // `print`
    auto call = std::make_unique<ast::CallExpr>();
    call->span = start.span;
    auto callee = std::make_unique<ast::NameExpr>();
    callee->span = start.span;
    callee->name = "print";
    call->callee = std::move(callee);
    call->args.push_back(parseExpression());
    call->span = call->span.join(previous().span);

    auto statement = std::make_unique<ast::ExprStmt>();
    statement->span = call->span;
    statement->expr = std::move(call);
    return statement;
  }

  auto statement = std::make_unique<ast::ExprStmt>();
  statement->span = peek().span;
  statement->expr = parseExpression();
  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseIf() {
  const Token& start = advance();  // `if`

  auto statement = std::make_unique<ast::IfStmt>();
  statement->span = start.span;

  while (true) {
    ast::ExprPtr condition = parseExpression();
    expect(TokenKind::Colon, "':' after the condition");
    std::vector<ast::StmtPtr> body = parseBlock("if");
    statement->branches.emplace_back(std::move(condition), std::move(body));

    if (isWord("elif")) {
      advance();
      continue;
    }
    break;
  }

  if (isWord("else")) {
    advance();
    expect(TokenKind::Colon, "':' after 'else'");
    statement->else_body = parseBlock("else");
    statement->has_else = true;
  }

  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseWhile() {
  const Token& start = advance();  // `while`

  auto statement = std::make_unique<ast::WhileStmt>();
  statement->span = start.span;
  statement->condition = parseExpression();
  expect(TokenKind::Colon, "':' after the condition");
  statement->body = parseBlock("while");
  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseFor() {
  const Token& start = advance();  // `for`

  auto statement = std::make_unique<ast::ForStmt>();
  statement->span = start.span;
  statement->name = expectIdentifier("a loop variable");
  if (!matchWord("in")) {
    error("expected 'in' after the loop variable but found " + peek().describe(), peek().span);
    synchronizeToLineEnd();
    return statement;
  }
  statement->iterable = parseExpression();
  expect(TokenKind::Colon, "':' after the iterable");
  statement->body = parseBlock("for");
  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseFn() {
  const Token& start = advance();  // `fn` / `function`

  auto statement = std::make_unique<ast::FnStmt>();
  statement->span = start.span;
  statement->name = expectIdentifier("a function name");

  if (match(TokenKind::LeftParen)) {
    if (!check(TokenKind::RightParen)) {
      while (true) {
        ast::FunctionParam param;
        param.span = peek().span;
        param.name = expectIdentifier("a parameter name");
        if (match(TokenKind::Colon)) {
          param.type = parseType();
        }
        statement->params.push_back(std::move(param));
        if (!match(TokenKind::Comma)) break;
        if (check(TokenKind::RightParen)) break;
      }
    }
    expect(TokenKind::RightParen, "')' after the parameters");
  }

  if (match(TokenKind::Arrow)) {
    statement->return_type = parseType();
  }

  expect(TokenKind::Colon, "':' before the function body");
  statement->body = parseBlock("function");
  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseReturn() {
  const Token& start = advance();  // `return`

  auto statement = std::make_unique<ast::ReturnStmt>();
  statement->span = start.span;
  if (!check(TokenKind::Newline) && !check(TokenKind::EndOfFile) && !check(TokenKind::Semicolon)) {
    statement->value = parseExpression();
  }
  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseBreak() {
  const Token& start = advance();
  auto statement = std::make_unique<ast::BreakStmt>();
  statement->span = start.span;
  return statement;
}

ast::StmtPtr Parser::parseContinue() {
  const Token& start = advance();
  auto statement = std::make_unique<ast::ContinueStmt>();
  statement->span = start.span;
  return statement;
}

ast::StmtPtr Parser::parseImport() {
  const Token& start = advance();  // `import`

  auto statement = std::make_unique<ast::ImportStmt>();
  statement->span = start.span;

  std::string module = expectIdentifier("a module name");
  while (match(TokenKind::Dot)) {
    module += ".";
    module += expectIdentifier("a module name");
  }
  statement->module = module;

  if (matchWord("as")) {
    statement->alias = expectIdentifier("an alias");
  }

  statement->span = statement->span.join(previous().span);
  return statement;
}

ast::StmtPtr Parser::parseFromImport() {
  const Token& start = advance();  // `from`

  auto statement = std::make_unique<ast::ImportStmt>();
  statement->span = start.span;
  statement->is_from = true;

  std::string module = expectIdentifier("a module name");
  while (match(TokenKind::Dot)) {
    module += ".";
    module += expectIdentifier("a module name");
  }
  statement->module = module;

  if (!matchWord("import")) {
    error("expected 'import' but found " + peek().describe(), peek().span);
    synchronizeToLineEnd();
    return statement;
  }

  while (true) {
    statement->members.push_back(expectIdentifier("a name to import"));
    if (!match(TokenKind::Comma)) break;
  }

  statement->span = statement->span.join(previous().span);
  return statement;
}

std::vector<ast::StmtPtr> Parser::parseBlock(const std::string& owner) {
  std::vector<ast::StmtPtr> body;

  if (!match(TokenKind::Newline)) {
    error("expected a newline and an indented block after '" + owner + "' but found " +
              peek().describe(),
          peek().span);
    synchronizeToLineEnd();
    return body;
  }

  if (!match(TokenKind::Indent)) {
    error("the body of '" + owner + "' needs to be indented by four spaces", peek().span);
    return body;
  }

  while (!check(TokenKind::Dedent) && !atEnd() && error_count_ == 0) {
    if (match(TokenKind::Newline)) continue;
    ast::StmtPtr statement = parseStatement();
    if (!statement) break;
    body.push_back(std::move(statement));
  }

  if (!match(TokenKind::Dedent)) {
    error("this block is never closed", peek().span);
  }

  return body;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

std::shared_ptr<ast::TypeRef> Parser::parseType() {
  auto type = std::make_shared<ast::TypeRef>();
  type->name = toLower(expectIdentifier("a type name"));

  if (type->name == "tensor") {
    if (match(TokenKind::LeftBracket)) {
      type->has_shape = true;
      while (true) {
        if (check(TokenKind::Question)) {
          advance();
          type->shape.push_back(-1);
        } else {
          const Token& number = expect(TokenKind::Integer, "a size");
          type->shape.push_back(number.int_value);
        }
        if (!match(TokenKind::Comma)) break;
      }
      expect(TokenKind::RightBracket, "']' after the shape");
    }
    if (matchWord("of")) {
      type->dtype = toLower(expectIdentifier("a dtype"));
    }
    return type;
  }

  if (type->name == "list") {
    if (matchWord("of")) {
      type->element = parseType();
    }
    return type;
  }

  if (type->name == "dict") {
    if (match(TokenKind::LeftBracket)) {
      type->element = parseType();
      expect(TokenKind::Comma, "',' between the key and value types");
      type->value_type = parseType();
      expect(TokenKind::RightBracket, "']' after the types");
    }
    return type;
  }

  return type;
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

ast::ExprPtr Parser::parseExpression() { return parseOr(); }

ast::ExprPtr Parser::parseOr() {
  ast::ExprPtr left = parseAnd();
  while (isWord("or")) {
    advance();
    ast::ExprPtr right = parseAnd();
    auto node = std::make_unique<ast::BinaryExpr>();
    node->span = left->span.join(right->span);
    node->op = "or";
    node->lhs = std::move(left);
    node->rhs = std::move(right);
    left = std::move(node);
  }
  return left;
}

ast::ExprPtr Parser::parseAnd() {
  ast::ExprPtr left = parseNot();
  while (isWord("and")) {
    advance();
    ast::ExprPtr right = parseNot();
    auto node = std::make_unique<ast::BinaryExpr>();
    node->span = left->span.join(right->span);
    node->op = "and";
    node->lhs = std::move(left);
    node->rhs = std::move(right);
    left = std::move(node);
  }
  return left;
}

ast::ExprPtr Parser::parseNot() {
  if (isWord("not")) {
    const Token& op = advance();
    ast::ExprPtr operand = parseNot();
    auto node = std::make_unique<ast::UnaryExpr>();
    node->span = op.span.join(operand->span);
    node->op = "not";
    node->operand = std::move(operand);
    return node;
  }
  return parseComparison();
}

ast::ExprPtr Parser::parseComparison() {
  ast::ExprPtr left = parseAdditive();

  if (isWord("is")) {
    advance();
    ast::ExprPtr right = parseAdditive();
    auto node = std::make_unique<ast::TypeTestExpr>();
    node->span = left->span.join(right->span);
    node->operand = std::move(left);
    if (right->kind() == ast::ExprKind::Name) {
      node->type_name = static_cast<ast::NameExpr&>(*right).name;
    } else {
      node->type_name = "";
    }
    return node;
  }

  const TokenKind kind = peek().kind;
  const bool isComparison = kind == TokenKind::EqualEqual || kind == TokenKind::BangEqual ||
                            kind == TokenKind::Less || kind == TokenKind::LessEqual ||
                            kind == TokenKind::Greater || kind == TokenKind::GreaterEqual;
  if (!isComparison) return left;

  const Token& op = advance();
  ast::ExprPtr right = parseAdditive();
  auto node = std::make_unique<ast::BinaryExpr>();
  node->span = left->span.join(right->span);
  node->op = op.text.empty() ? std::string(to_string(op.kind)) : op.text;
  node->lhs = std::move(left);
  node->rhs = std::move(right);
  return node;
}

ast::ExprPtr Parser::parseAdditive() {
  ast::ExprPtr left = parseMultiplicative();
  while (check(TokenKind::Plus) || check(TokenKind::Minus)) {
    const Token& op = advance();
    ast::ExprPtr right = parseMultiplicative();
    auto node = std::make_unique<ast::BinaryExpr>();
    node->span = left->span.join(right->span);
    node->op = op.kind == TokenKind::Plus ? "+" : "-";
    node->lhs = std::move(left);
    node->rhs = std::move(right);
    left = std::move(node);
  }
  return left;
}

ast::ExprPtr Parser::parseMultiplicative() {
  ast::ExprPtr left = parseUnary();
  while (check(TokenKind::Star) || check(TokenKind::Slash) || check(TokenKind::SlashSlash) ||
         check(TokenKind::Percent)) {
    const Token& op = advance();
    ast::ExprPtr right = parseUnary();
    auto node = std::make_unique<ast::BinaryExpr>();
    node->span = left->span.join(right->span);
    switch (op.kind) {
      case TokenKind::Star:
        node->op = "*";
        break;
      case TokenKind::Slash:
        node->op = "/";
        break;
      case TokenKind::SlashSlash:
        node->op = "//";
        break;
      default:
        node->op = "%";
        break;
    }
    node->lhs = std::move(left);
    node->rhs = std::move(right);
    left = std::move(node);
  }
  return left;
}

ast::ExprPtr Parser::parseUnary() {
  if (check(TokenKind::Minus) || check(TokenKind::Plus)) {
    const Token& op = advance();
    ast::ExprPtr operand = parseUnary();
    auto node = std::make_unique<ast::UnaryExpr>();
    node->span = op.span.join(operand->span);
    node->op = op.kind == TokenKind::Minus ? "-" : "+";
    node->operand = std::move(operand);
    return node;
  }
  return parsePower();
}

ast::ExprPtr Parser::parsePower() {
  ast::ExprPtr base = parsePostfix();
  if (check(TokenKind::StarStar)) {
    advance();
    // Right-associative, and binds tighter than unary minus on the left:
    // -2 ** 2 == -4
    ast::ExprPtr exponent = parseUnary();
    auto node = std::make_unique<ast::BinaryExpr>();
    node->span = base->span.join(exponent->span);
    node->op = "**";
    node->lhs = std::move(base);
    node->rhs = std::move(exponent);
    return node;
  }
  return base;
}

ast::ExprPtr Parser::parsePostfix() {
  ast::ExprPtr expression = parsePrimary();

  while (true) {
    if (check(TokenKind::LeftParen)) {
      advance();
      auto node = std::make_unique<ast::CallExpr>();
      node->span = expression->span;
      node->callee = std::move(expression);
      if (!check(TokenKind::RightParen)) {
        while (true) {
          if (check(TokenKind::Identifier) && checkAt(1, TokenKind::Equal) &&
              !isHardKeyword(peek().text)) {
            const std::string name = advance().text;
            advance();  // `=`
            ast::ExprPtr value = parseExpression();
            node->kwargs.emplace_back(name, std::move(value));
          } else {
            node->args.push_back(parseExpression());
          }
          if (!match(TokenKind::Comma)) break;
          if (check(TokenKind::RightParen)) break;
        }
      }
      node->span = node->span.join(expect(TokenKind::RightParen, "')' after the arguments").span);
      expression = std::move(node);
      continue;
    }

    if (check(TokenKind::LeftBracket)) {
      advance();
      auto node = std::make_unique<ast::IndexExpr>();
      node->span = expression->span;
      node->target = std::move(expression);
      if (!check(TokenKind::RightBracket)) {
        while (true) {
          node->indices.push_back(parseExpression());
          if (!match(TokenKind::Comma)) break;
          if (check(TokenKind::RightBracket)) break;
        }
      }
      node->span = node->span.join(expect(TokenKind::RightBracket, "']' after the index").span);
      expression = std::move(node);
      continue;
    }

    if (check(TokenKind::Dot)) {
      advance();
      auto node = std::make_unique<ast::MemberExpr>();
      node->span = expression->span;
      node->target = std::move(expression);
      node->member = expectIdentifier("a field name");
      node->span = node->span.join(previous().span);
      expression = std::move(node);
      continue;
    }

    break;
  }

  return expression;
}

ast::ExprPtr Parser::parsePrimary() {
  const Token& token = peek();

  switch (token.kind) {
    case TokenKind::Integer: {
      advance();
      auto node = std::make_unique<ast::IntLiteral>();
      node->span = token.span;
      node->value = token.int_value;
      return node;
    }
    case TokenKind::Float: {
      advance();
      auto node = std::make_unique<ast::FloatLiteral>();
      node->span = token.span;
      node->value = token.float_value;
      return node;
    }
    case TokenKind::String: {
      advance();
      auto node = std::make_unique<ast::StringLiteral>();
      node->span = token.span;
      node->value = token.text;
      return node;
    }
    case TokenKind::LeftParen: {
      advance();
      ast::ExprPtr inner = parseExpression();
      expect(TokenKind::RightParen, "')'");
      return inner;
    }
    case TokenKind::LeftBracket:
      return parseListLiteral();
    case TokenKind::LeftBrace:
      return parseDictLiteral();
    case TokenKind::Identifier: {
      // `true`, `false` and `null` are literals, not names.
      if (token.text == "true" || token.text == "false") {
        advance();
        auto node = std::make_unique<ast::BoolLiteral>();
        node->span = token.span;
        node->value = toLower(token.text) == "true";
        return node;
      }
      if (token.text == "null") {
        advance();
        auto node = std::make_unique<ast::NullLiteral>();
        node->span = token.span;
        return node;
      }
      if (isHardKeyword(token.text)) {
        error("`" + token.text + "` is a keyword, so it cannot be used as a name",
              token.span);
        synchronizeToLineEnd();
        auto node = std::make_unique<ast::NameExpr>();
        node->span = token.span;
        node->name = token.text;
        return node;
      }
      advance();
      auto node = std::make_unique<ast::NameExpr>();
      node->span = token.span;
      node->name = token.text;
      return node;
    }
    default:
      break;
  }

  error("expected a value but found " + token.describe(), token.span);
  synchronizeToLineEnd();
  auto node = std::make_unique<ast::NameExpr>();
  node->span = token.span;
  node->name = "";
  return node;
}

ast::ExprPtr Parser::parseListLiteral() {
  const Token& start = advance();  // `[`

  auto node = std::make_unique<ast::ListExpr>();
  node->span = start.span;

  if (!check(TokenKind::RightBracket)) {
    while (true) {
      node->elements.push_back(parseExpression());
      if (!match(TokenKind::Comma)) break;
      if (check(TokenKind::RightBracket)) break;
    }
  }

  node->span = node->span.join(expect(TokenKind::RightBracket, "']' after the list").span);
  return node;
}

ast::ExprPtr Parser::parseDictLiteral() {
  const Token& start = advance();  // `{`

  auto node = std::make_unique<ast::DictExpr>();
  node->span = start.span;

  if (!check(TokenKind::RightBrace)) {
    while (true) {
      ast::ExprPtr key = parseExpression();
      expect(TokenKind::Colon, "':' after the key");
      ast::ExprPtr value = parseExpression();
      node->entries.emplace_back(std::move(key), std::move(value));
      if (!match(TokenKind::Comma)) break;
      if (check(TokenKind::RightBrace)) break;
    }
  }

  node->span = node->span.join(expect(TokenKind::RightBrace, "'}' after the dictionary").span);
  return node;
}

// ---------------------------------------------------------------------------
// Sentences
// ---------------------------------------------------------------------------

std::string Parser::describeVerbShape(const core::VerbSpec& spec) const {
  return shapeFor(spec);
}

bool Parser::canParseSentence() {
  const std::size_t save = index_;
  ast::SentenceClause clause;
  const bool matched = tryParseClause(clause);
  index_ = save;
  return matched;
}

ast::StmtPtr Parser::parseSentence() {
  auto statement = std::make_unique<ast::SentenceStmt>();
  statement->span = peek().span;

  ast::Sentence& sentence = statement->sentence;
  bool expect_clause = true;

  while (!check(TokenKind::Newline) && !atEnd() && error_count_ == 0) {
    if (matchWord("and")) {
      expect_clause = true;
      continue;
    }

    if (expect_clause) {
      ast::SentenceClause clause;
      if (tryParseClause(clause)) {
        sentence.clauses.push_back(std::move(clause));
        expect_clause = false;
        continue;
      }
    }

    ast::SentenceOption option;
    if (tryParseOption(option)) {
      sentence.options.push_back(std::move(option));
      expect_clause = false;
      continue;
    }

    // Nothing matched. If the line started with a verb we know the shape
    // the user was aiming for.
    const core::VerbSpec* verb = core::findVerb(peek().text);
    if (verb != nullptr) {
      error("I could not read this sentence; expected `" + describeVerbShape(*verb) + "`",
            peek().span);
    } else {
      error("I could not read `" + peek().text + "` here", peek().span);
    }
    synchronizeToLineEnd();
    break;
  }

  // A sentence may continue in an indented block below it.
  if (match(TokenKind::Newline) && check(TokenKind::Indent)) {
    advance();  // `Indent`
    while (!check(TokenKind::Dedent) && !atEnd() && error_count_ == 0) {
      if (match(TokenKind::Newline)) continue;

      // A line of the body is either a run of options or an ordinary
      // statement.
      {
        bool parsed_option = false;
        while (true) {
          ast::SentenceOption option;
          const std::size_t save = index_;
          if (!tryParseOption(option)) {
            index_ = save;
            break;
          }
          sentence.options.push_back(std::move(option));
          parsed_option = true;
        }
        if (parsed_option) {
          if (!check(TokenKind::Newline) && !atEnd()) {
            error("unexpected `" + peek().text + "` after this option", peek().span);
            synchronizeToLineEnd();
            break;
          }
          continue;
        }
      }

      // A nested sentence becomes another clause of this sentence.
      if (check(TokenKind::Identifier) && core::findVerb(peek().text) != nullptr) {
        ast::SentenceClause clause;
        if (tryParseClause(clause)) {
          sentence.clauses.push_back(std::move(clause));
          continue;
        }
        const core::VerbSpec* verb = core::findVerb(peek().text);
        error("I could not read this sentence; expected `" + describeVerbShape(*verb) + "`",
              peek().span);
        break;
      }

      // Otherwise: an ordinary statement.
      ast::StmtPtr nested = parseStatement();
      if (!nested) break;
      statement->body.push_back(std::move(nested));
    }
    if (!match(TokenKind::Dedent)) {
      error("this block is never closed", peek().span);
    }
  }

  statement->span = statement->span.join(previous().span);
  return statement;
}

bool Parser::tryParseClause(ast::SentenceClause& out) {
  const core::VerbSpec* verb = core::findVerb(peek().text);
  if (verb == nullptr) return false;

  const std::size_t start = index_;
  advance();  // the verb itself; the template covers what follows it

  std::size_t best_end = index_;
  std::vector<std::pair<std::string, ast::ExprPtr>> best_slots;
  std::size_t best_length = 0;

  for (const std::vector<core::TemplatePart>& alternative : verb->alternatives) {
    const std::size_t save = index_;
    std::vector<std::pair<std::string, ast::ExprPtr>> slots;
    if (matchTemplate(alternative, slots)) {
      const std::size_t length = index_ - save;
      if (length > best_length) {
        best_length = length;
        best_end = index_;
        best_slots = std::move(slots);
      }
    }
    index_ = save;
  }

  if (best_length == 0) {
    index_ = start;
    return false;
  }

  index_ = best_end;
  out.verb = verb->verb;
  out.slots = std::move(best_slots);
  out.span = tokens_[start].span.join(tokens_[best_end - 1].span);
  return true;
}

bool Parser::matchTemplate(const std::vector<core::TemplatePart>& parts,
                           std::vector<std::pair<std::string, ast::ExprPtr>>& slots) {
  for (const core::TemplatePart& part : parts) {
    if (part.literal) {
      if (!isWord(part.literal_text)) return false;
      advance();
      continue;
    }

    const Token& token = peek();
    switch (part.slot) {
      case core::SlotKind::Identifier: {
        if (token.kind != TokenKind::Identifier || isHardKeyword(token.text)) return false;
        auto node = std::make_unique<ast::NameExpr>();
        node->span = token.span;
        node->name = token.text;
        slots.emplace_back(part.slot_name, std::move(node));
        advance();
        break;
      }
      case core::SlotKind::Path: {
        if (token.kind != TokenKind::String) return false;
        auto node = std::make_unique<ast::StringLiteral>();
        node->span = token.span;
        node->value = token.text;
        slots.emplace_back(part.slot_name, std::move(node));
        advance();
        break;
      }
      case core::SlotKind::Number: {
        if (token.kind != TokenKind::Integer) return false;
        auto node = std::make_unique<ast::IntLiteral>();
        node->span = token.span;
        node->value = token.int_value;
        slots.emplace_back(part.slot_name, std::move(node));
        advance();
        break;
      }
      case core::SlotKind::Value: {
        if (token.kind == TokenKind::Identifier && !isHardKeyword(token.text)) {
          auto node = std::make_unique<ast::NameExpr>();
          node->span = token.span;
          node->name = token.text;
          slots.emplace_back(part.slot_name, std::move(node));
          advance();
        } else if (token.kind == TokenKind::String) {
          auto node = std::make_unique<ast::StringLiteral>();
          node->span = token.span;
          node->value = token.text;
          slots.emplace_back(part.slot_name, std::move(node));
          advance();
        } else if (token.kind == TokenKind::Integer) {
          auto node = std::make_unique<ast::IntLiteral>();
          node->span = token.span;
          node->value = token.int_value;
          slots.emplace_back(part.slot_name, std::move(node));
          advance();
        } else {
          return false;
        }
        break;
      }
      case core::SlotKind::Kind: {
        if (token.kind != TokenKind::Identifier) return false;
        const std::string lower = toLower(token.text);
        if (std::find(part.allowed.begin(), part.allowed.end(), lower) == part.allowed.end()) {
          return false;
        }
        auto node = std::make_unique<ast::NameExpr>();
        node->span = token.span;
        node->name = lower;
        slots.emplace_back(part.slot_name, std::move(node));
        advance();
        break;
      }
    }
  }
  return true;
}

const core::OptionKey* Parser::matchOptionKey() const {
  for (const core::OptionKey& key : core::optionKeys()) {
    bool matches = true;
    for (std::size_t i = 0; i < key.words.size(); ++i) {
      if (!isWordAt(i, key.words[i])) {
        matches = false;
        break;
      }
    }
    if (matches) return &key;
  }
  return nullptr;
}

bool Parser::tryParseOption(ast::SentenceOption& out) {
  const std::size_t save = index_;

  // `for 3 epochs` is a spelling of `epochs 3`.
  if (isWord("for") && checkAt(1, TokenKind::Integer) && isWordAt(2, "epochs")) {
    const Token& number = peek(1);
    advance();
    advance();
    advance();
    auto node = std::make_unique<ast::IntLiteral>();
    node->span = number.span;
    node->value = number.int_value;
    out.key = "epochs";
    out.value = std::move(node);
    out.span = number.span;
    return true;
  }

  // An optional separator: `with rank 16`.
  for (const std::string& separator : core::optionSeparators()) {
    if (isWord(separator)) {
      advance();
      break;
    }
  }

  const core::OptionKey* key = matchOptionKey();
  if (key == nullptr) {
    index_ = save;
    return false;
  }
  // Consume the key's words.
  for (std::size_t i = 0; i < key->words.size(); ++i) {
    advance();
  }

  // An optional `is` or `=`.
  if (isWord("is")) {
    advance();
  } else if (check(TokenKind::Equal)) {
    advance();
  }

  const Token& token = peek();
  ast::ExprPtr value;
  if (token.kind == TokenKind::Integer) {
    auto node = std::make_unique<ast::IntLiteral>();
    node->span = token.span;
    node->value = token.int_value;
    value = std::move(node);
  } else if (token.kind == TokenKind::Float) {
    auto node = std::make_unique<ast::FloatLiteral>();
    node->span = token.span;
    node->value = token.float_value;
    value = std::move(node);
  } else if (token.kind == TokenKind::String) {
    auto node = std::make_unique<ast::StringLiteral>();
    node->span = token.span;
    node->value = token.text;
    value = std::move(node);
  } else if (token.kind == TokenKind::Identifier && !isHardKeyword(token.text)) {
    // A bare word in an option is a literal, not a variable: `precision is
    // bf16` means the text "bf16", not the value of a name called bf16.
    auto node = std::make_unique<ast::StringLiteral>();
    node->span = token.span;
    node->value = token.text;
    value = std::move(node);
  } else {
    index_ = save;
    return false;
  }
  advance();

  out.key = key->canonical;
  out.span = token.span;
  out.value = std::move(value);
  return true;
}

}  // namespace milo
