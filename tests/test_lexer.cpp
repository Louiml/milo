#include "core/lexer.hpp"

#include <string>
#include <vector>

#include "test_framework.hpp"

using namespace milo;

namespace {

std::vector<Token> lex(const std::string& source, DiagnosticBag* diagnostics = nullptr) {
  Lexer lexer(source, "test.milo");
  std::vector<Token> tokens = lexer.tokenize();
  if (diagnostics != nullptr) *diagnostics = lexer.diagnostics();
  return tokens;
}

/// The kind of the n-th token, or EndOfFile past the end.
TokenKind kindAt(const std::vector<Token>& tokens, std::size_t index) {
  if (index >= tokens.size()) return TokenKind::EndOfFile;
  return tokens[index].kind;
}

std::string textAt(const std::vector<Token>& tokens, std::size_t index) {
  if (index >= tokens.size()) return {};
  return tokens[index].text;
}

}  // namespace

TEST(lexerLexesIntegersAndFloats) {
  const auto tokens = lex("1 2.5 0x1f 1_000");
  CHECK_EQ(kindAt(tokens, 0), TokenKind::Integer);
  CHECK_EQ(tokens[0].int_value, static_cast<std::int64_t>(1));
  CHECK_EQ(kindAt(tokens, 1), TokenKind::Float);
  CHECK(tokens[1].float_value > 2.49 && tokens[1].float_value < 2.51);
  CHECK_EQ(kindAt(tokens, 2), TokenKind::Integer);
  CHECK_EQ(tokens[2].int_value, static_cast<std::int64_t>(31));
  CHECK_EQ(kindAt(tokens, 3), TokenKind::Integer);
  CHECK_EQ(tokens[3].int_value, static_cast<std::int64_t>(1000));
}

TEST(lexerDoesNotTreatTrailingDotAsAFloat) {
  // `1.` must lex as the integer 1 followed by a dot, so that a sentence
  // may end with a period.
  const auto tokens = lex("1.");
  CHECK_EQ(kindAt(tokens, 0), TokenKind::Integer);
  CHECK_EQ(kindAt(tokens, 1), TokenKind::Dot);
}

TEST(lexerDecodesStringEscapes) {
  const auto tokens = lex(R"("a\nb\"c")");
  CHECK_EQ(kindAt(tokens, 0), TokenKind::String);
  CHECK_EQ(textAt(tokens, 0), std::string("a\nb\"c"));
}

TEST(lexerHandlesSingleQuotedStrings) {
  const auto tokens = lex("'hi'");
  CHECK_EQ(kindAt(tokens, 0), TokenKind::String);
  CHECK_EQ(textAt(tokens, 0), std::string("hi"));
}

TEST(lexerEmitsIndentAndDedent) {
  const auto tokens = lex("if x:\n    a\n    b\n");
  // if, x, :, NEWLINE, INDENT, a, NEWLINE, b, NEWLINE, DEDENT
  CHECK_EQ(kindAt(tokens, 0), TokenKind::Identifier);
  CHECK_EQ(kindAt(tokens, 3), TokenKind::Newline);
  CHECK_EQ(kindAt(tokens, 4), TokenKind::Indent);
  CHECK_EQ(kindAt(tokens, 5), TokenKind::Identifier);
  CHECK_EQ(kindAt(tokens, 7), TokenKind::Identifier);
  CHECK_EQ(kindAt(tokens, 9), TokenKind::Dedent);
}

TEST(lexerIgnoresBlankAndCommentLines) {
  const auto tokens = lex("a\n\n# a comment\nb\n");
  CHECK_EQ(kindAt(tokens, 0), TokenKind::Identifier);
  CHECK_EQ(textAt(tokens, 0), std::string("a"));
  CHECK_EQ(kindAt(tokens, 1), TokenKind::Newline);
  // The comment and the blank line produce nothing.
  CHECK_EQ(kindAt(tokens, 2), TokenKind::Identifier);
  CHECK_EQ(textAt(tokens, 2), std::string("b"));
}

TEST(lexerSuppressesNewlinesInsideBrackets) {
  const auto tokens = lex("f(\n  1,\n  2\n)\n");
  bool sawNewlineInside = false;
  for (std::size_t i = 0; i + 3 < tokens.size(); ++i) {
    if (tokens[i].kind == TokenKind::Newline && tokens[i + 1].kind == TokenKind::Integer) {
      sawNewlineInside = true;
    }
  }
  CHECK(!sawNewlineInside);
}

TEST(lexerKnowsHyphenatedVerbs) {
  const auto tokens = lex("fine-tune base on train");
  CHECK_EQ(kindAt(tokens, 0), TokenKind::Identifier);
  CHECK_EQ(textAt(tokens, 0), std::string("fine-tune"));
}

TEST(lexerKeepsOrdinaryHyphensAsOperators) {
  const auto tokens = lex("a-b");
  CHECK_EQ(textAt(tokens, 0), std::string("a"));
  CHECK_EQ(kindAt(tokens, 1), TokenKind::Minus);
  CHECK_EQ(textAt(tokens, 2), std::string("b"));
}

TEST(lexerReportsUnterminatedStrings) {
  DiagnosticBag diagnostics;
  lex("\"oops", &diagnostics);
  CHECK(diagnostics.hasErrors());
}

TEST(lexerReportsMixedIndentation) {
  DiagnosticBag diagnostics;
  lex("if x:\n\t  a\n", &diagnostics);
  CHECK(diagnostics.hasErrors());
}

TEST(lexerRecognisesOperators) {
  const auto tokens = lex("== != <= >= ** // -> += ");
  CHECK_EQ(kindAt(tokens, 0), TokenKind::EqualEqual);
  CHECK_EQ(kindAt(tokens, 1), TokenKind::BangEqual);
  CHECK_EQ(kindAt(tokens, 2), TokenKind::LessEqual);
  CHECK_EQ(kindAt(tokens, 3), TokenKind::GreaterEqual);
  CHECK_EQ(kindAt(tokens, 4), TokenKind::StarStar);
  CHECK_EQ(kindAt(tokens, 5), TokenKind::SlashSlash);
  CHECK_EQ(kindAt(tokens, 6), TokenKind::Arrow);
}

TEST(lexerCountsColumns) {
  const auto tokens = lex("x");
  CHECK_EQ(tokens[0].span.line, 1);
  CHECK_EQ(tokens[0].span.column, 1);

  // The column of an indented line counts the indentation.
  const auto indented = lex("if x:\n    y\n");
  std::size_t y_index = 0;
  for (std::size_t i = 0; i < indented.size(); ++i) {
    if (indented[i].kind == TokenKind::Identifier && indented[i].text == "y") y_index = i;
  }
  CHECK_EQ(indented[y_index].span.line, 2);
  CHECK_EQ(indented[y_index].span.column, 5);
}
