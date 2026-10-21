// The parser: tokens -> AST.
#ifndef MILO_CORE_PARSER_HPP
#define MILO_CORE_PARSER_HPP

#include <string>
#include <vector>

#include "core/ast.hpp"
#include "core/diagnostic.hpp"
#include "core/directives.hpp"
#include "core/token.hpp"

namespace milo {

/// A recursive-descent parser for Milo.
///
/// Two statement families share one grammar:
///
///   * explicit mode — typed variables, functions, loops, direct calls
///   * sentence mode — plain-English templates (`fine-tune base on train`)
///
/// Sentences are matched against the template table in
/// `core/directives.hpp`. A sentence is parsed into an ordinary
/// `SentenceStmt`, which the interpreter dispatches on the verb.
class Parser {
 public:
  Parser(std::vector<Token> tokens, std::string filename);

  /// Parse the whole token stream.
  ///
  /// On a syntax error the diagnostics bag is filled and parsing stops; the
  /// statements parsed so far are still returned so that a REPL can keep
  /// working.
  std::vector<ast::StmtPtr> parseProgram();

  const DiagnosticBag& diagnostics() const { return diagnostics_; }
  DiagnosticBag& diagnostics() { return diagnostics_; }

  const std::string& filename() const { return filename_; }

 private:
  // --- token helpers -----------------------------------------------------
  const Token& peek(std::size_t offset = 0) const;
  const Token& previous() const;
  bool atEnd() const;
  bool check(TokenKind kind) const;
  bool checkAt(std::size_t offset, TokenKind kind) const;
  bool match(TokenKind kind);
  const Token& advance();
  bool isWord(const std::string& word) const;
  bool isWordAt(std::size_t offset, const std::string& word) const;
  bool matchWord(const std::string& word);
  Token expect(TokenKind kind, const std::string& what);
  std::string expectIdentifier(const std::string& what);

  // --- diagnostics --------------------------------------------------------
  void error(const std::string& message, SourceSpan span);
  void errorWithSuggestion(const std::string& message, SourceSpan span,
                           const std::string& word,
                           const std::vector<std::string>& candidates);
  void synchronizeToLineEnd();

  // --- statements --------------------------------------------------------
  ast::StmtPtr parseStatement();
  ast::StmtPtr parseLet();
  ast::StmtPtr parseSet();
  ast::StmtPtr parseIf();
  ast::StmtPtr parseWhile();
  ast::StmtPtr parseFor();
  ast::StmtPtr parseFn();
  ast::StmtPtr parseReturn();
  ast::StmtPtr parseBreak();
  ast::StmtPtr parseContinue();
  ast::StmtPtr parseImport();
  ast::StmtPtr parseFromImport();
  ast::StmtPtr parseAssignmentOrExpression();
  std::vector<ast::StmtPtr> parseBlock(const std::string& owner);

  // --- types -------------------------------------------------------------
  std::shared_ptr<ast::TypeRef> parseType();

  // --- expressions -------------------------------------------------------
  ast::ExprPtr parseExpression();
  ast::ExprPtr parseOr();
  ast::ExprPtr parseAnd();
  ast::ExprPtr parseNot();
  ast::ExprPtr parseComparison();
  ast::ExprPtr parseAdditive();
  ast::ExprPtr parseMultiplicative();
  ast::ExprPtr parseUnary();
  ast::ExprPtr parsePower();
  ast::ExprPtr parsePostfix();
  ast::ExprPtr parsePrimary();
  ast::ExprPtr parseListLiteral();
  ast::ExprPtr parseDictLiteral();

  // --- sentences ---------------------------------------------------------
  ast::StmtPtr parseSentence();
  bool canParseSentence();
  bool tryParseClause(ast::SentenceClause& out);
  bool matchTemplate(const std::vector<core::TemplatePart>& parts,
                     std::vector<std::pair<std::string, ast::ExprPtr>>& slots);
  bool tryParseOption(ast::SentenceOption& out);
  const core::OptionKey* matchOptionKey() const;
  std::string describeVerbShape(const core::VerbSpec& spec) const;

  std::vector<Token> tokens_;
  std::string filename_;
  std::size_t index_ = 0;
  DiagnosticBag diagnostics_;
  /// Guards against runaway error recovery.
  int error_count_ = 0;
};

}  // namespace milo

#endif  // MILO_CORE_PARSER_HPP
