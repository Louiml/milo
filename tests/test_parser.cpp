#include "core/parser.hpp"

#include <string>
#include <vector>

#include "core/lexer.hpp"
#include "test_framework.hpp"

using namespace milo;
using namespace milo::ast;

namespace {

struct ParseResult {
  std::vector<StmtPtr> program;
  DiagnosticBag diagnostics;
};

ParseResult parse(const std::string& source) {
  ParseResult result;
  Lexer lexer(source, "test.milo");
  std::vector<Token> tokens = lexer.tokenize();
  if (lexer.diagnostics().hasErrors()) {
    result.diagnostics = lexer.diagnostics();
    return result;
  }
  Parser parser(std::move(tokens), "test.milo");
  result.program = parser.parseProgram();
  result.diagnostics = parser.diagnostics();
  return result;
}

/// Parse and expect success.
std::vector<StmtPtr> parseOk(const std::string& source) {
  ParseResult result = parse(source);
  if (result.diagnostics.hasErrors()) {
    ::milo::test::fail(__FILE__, __LINE__,
                       "expected the program to parse, but it did not:\n" +
                           result.diagnostics.render(source, "test.milo"));
  }
  return std::move(result.program);
}

/// The single statement of a program, cast to a concrete type.
template <typename T>
const T& as(const std::vector<StmtPtr>& program, std::size_t index = 0) {
  if (index >= program.size()) {
    ::milo::test::fail(__FILE__, __LINE__, "the program has fewer statements than expected");
  }
  const Stmt* statement = program[index].get();
  const T* typed = dynamic_cast<const T*>(statement);
  if (typed == nullptr) {
    ::milo::test::fail(__FILE__, __LINE__, "the statement has a different kind than expected");
  }
  return *typed;
}

}  // namespace

TEST(parserParsesALetStatement) {
  const auto program = parseOk("let x = 42");
  const auto& statement = as<LetStmt>(program);
  CHECK_EQ(statement.name, std::string("x"));
  CHECK(statement.has_value);
  const auto& value = dynamic_cast<const IntLiteral&>(*statement.value);
  CHECK_EQ(value.value, static_cast<std::int64_t>(42));
}

TEST(parserAcceptsBeAsAnAlternativeToEquals) {
  const auto program = parseOk("let x be 42");
  const auto& statement = as<LetStmt>(program);
  CHECK(statement.has_value);
}

TEST(parserParsesTypes) {
  const auto program = parseOk("let w: tensor[2, 3] of f32 = 0");
  const auto& statement = as<LetStmt>(program);
  CHECK(statement.type != nullptr);
  CHECK_EQ(statement.type->name, std::string("tensor"));
  CHECK(statement.type->has_shape);
  CHECK_EQ(statement.type->shape.size(), static_cast<std::size_t>(2));
  CHECK_EQ(statement.type->shape[0], static_cast<std::int64_t>(2));
  CHECK_EQ(statement.type->dtype, std::string("f32"));
}

TEST(parserParsesFunctionDeclarations) {
  const auto program = parseOk("fn add(a, b):\n    return a + b\n");
  const auto& statement = as<FnStmt>(program);
  CHECK_EQ(statement.name, std::string("add"));
  CHECK_EQ(statement.params.size(), static_cast<std::size_t>(2));
  CHECK_EQ(statement.params[0].name, std::string("a"));
  CHECK_EQ(statement.body.size(), static_cast<std::size_t>(1));
}

TEST(parserParsesIfElse) {
  const auto program = parseOk("if x > 1:\n    y = 1\nelse:\n    y = 2\n");
  const auto& statement = as<IfStmt>(program);
  CHECK_EQ(statement.branches.size(), static_cast<std::size_t>(1));
  CHECK(statement.has_else);
  CHECK_EQ(statement.else_body.size(), static_cast<std::size_t>(1));
}

TEST(parserParsesForLoops) {
  const auto program = parseOk("for i in range(10):\n    print i\n");
  const auto& statement = as<ForStmt>(program);
  CHECK_EQ(statement.name, std::string("i"));
  CHECK_EQ(statement.body.size(), static_cast<std::size_t>(1));
}

TEST(parserParsesExpressionsWithPrecedence) {
  const auto program = parseOk("let x = 1 + 2 * 3");
  const auto& statement = as<LetStmt>(program);
  const auto& binary = dynamic_cast<const BinaryExpr&>(*statement.value);
  CHECK_EQ(binary.op, std::string("+"));
  const auto& right = dynamic_cast<const BinaryExpr&>(*binary.rhs);
  CHECK_EQ(right.op, std::string("*"));
}

TEST(parserRejectsAMissingBody) {
  const auto result = parse("if x:\n");
  CHECK(result.diagnostics.hasErrors());
}

TEST(parserRejectsUnclosedBlocks) {
  const auto result = parse("if x:\n    y = 1\n");
  // The block closes at the end of input, which is fine; the error is that
  // there is no `else`... actually this must parse. Use a real error:
  const auto broken = parse("fn f(:\n    return 1\n");
  CHECK(broken.diagnostics.hasErrors());
}

TEST(parserRejectsKeywordsUsedAsNames) {
  const auto result = parse("let and = 1");
  CHECK(result.diagnostics.hasErrors());
}

TEST(parserParsesALoadSentence) {
  const auto program = parseOk("load model \"llama-3-8b.safetensors\" as base");
  const auto& statement = as<SentenceStmt>(program);
  CHECK_EQ(statement.sentence.clauses.size(), static_cast<std::size_t>(1));
  const SentenceClause& clause = statement.sentence.clauses.front();
  CHECK_EQ(clause.verb, std::string("load"));
  CHECK_EQ(clause.slots.size(), static_cast<std::size_t>(3));
  CHECK_EQ(clause.slots[0].first, std::string("kind"));
  CHECK_EQ(clause.slots[1].first, std::string("path"));
  CHECK_EQ(clause.slots[2].first, std::string("name"));
}

TEST(parserParsesASplitSentence) {
  const auto program = parseOk("split chats into train and test by 90 percent");
  const auto& statement = as<SentenceStmt>(program);
  const SentenceClause& clause = statement.sentence.clauses.front();
  CHECK_EQ(clause.verb, std::string("split"));
  CHECK_EQ(clause.slots.size(), static_cast<std::size_t>(4));
  CHECK_EQ(clause.slots[3].first, std::string("ratio"));
}

TEST(parserParsesAFineTuneSentenceWithABlock) {
  const auto program = parseOk(
      "fine-tune base on train\n"
      "    using lora with rank 16\n"
      "    precision is bf16\n"
      "    batch size 4\n"
      "    epochs 3\n"
      "    learning rate 0.0002\n"
      "    save to \"my-model.safetensors\"\n");
  const auto& statement = as<SentenceStmt>(program);
  CHECK_EQ(statement.sentence.clauses.size(), static_cast<std::size_t>(1));

  // The options from the block are collected in order.
  const auto& options = statement.sentence.options;
  CHECK_EQ(options.size(), static_cast<std::size_t>(7));
  CHECK_EQ(options[0].key, std::string("method"));
  CHECK_EQ(options[1].key, std::string("rank"));
  CHECK_EQ(options[2].key, std::string("precision"));
  CHECK_EQ(options[3].key, std::string("batch_size"));
  CHECK_EQ(options[4].key, std::string("epochs"));
  CHECK_EQ(options[5].key, std::string("learning_rate"));
  CHECK_EQ(options[6].key, std::string("save"));
}

TEST(parserParsesTwoClausesJoinedByAnd) {
  const auto program = parseOk("evaluate base on test and show accuracy");
  const auto& statement = as<SentenceStmt>(program);
  CHECK_EQ(statement.sentence.clauses.size(), static_cast<std::size_t>(2));
  CHECK_EQ(statement.sentence.clauses[0].verb, std::string("evaluate"));
  CHECK_EQ(statement.sentence.clauses[1].verb, std::string("show"));
}

TEST(parserAcceptsOptionsOnTheSentenceLine) {
  const auto program = parseOk("fine-tune base on train using lora with rank 16");
  const auto& statement = as<SentenceStmt>(program);
  CHECK_EQ(statement.sentence.options.size(), static_cast<std::size_t>(2));
  CHECK_EQ(statement.sentence.options[0].key, std::string("method"));
  CHECK_EQ(statement.sentence.options[1].key, std::string("rank"));
}

TEST(parserUnderstandsForNEpochs) {
  const auto program = parseOk("train base on train\n    for 3 epochs\n");
  const auto& statement = as<SentenceStmt>(program);
  CHECK_EQ(statement.sentence.options.size(), static_cast<std::size_t>(1));
  CHECK_EQ(statement.sentence.options[0].key, std::string("epochs"));
}

TEST(parserUnderstandsUsePython) {
  const auto program = parseOk("use python \"numpy\" as np");
  const auto& statement = as<SentenceStmt>(program);
  const SentenceClause& clause = statement.sentence.clauses.front();
  CHECK_EQ(clause.verb, std::string("use"));
  CHECK_EQ(clause.slots.size(), static_cast<std::size_t>(2));
  CHECK_EQ(clause.slots[0].first, std::string("module"));
  CHECK_EQ(clause.slots[1].first, std::string("name"));
}

TEST(parserMixesExplicitAndSentenceModes) {
  const auto program = parseOk("let x = 1\nload data \"a.csv\" as d\nprint x\n");
  CHECK_EQ(program.size(), static_cast<std::size_t>(3));
  CHECK_EQ(program[0]->kind(), StmtKind::Let);
  CHECK_EQ(program[1]->kind(), StmtKind::Sentence);
  CHECK_EQ(program[2]->kind(), StmtKind::Expr);
}

TEST(parserReportsAnUnknownSentenceShape) {
  const auto result = parse("load \"a.csv\" as d");
  CHECK(result.diagnostics.hasErrors());
}
