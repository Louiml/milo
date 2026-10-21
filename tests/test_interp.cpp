#include "runtime/interp.hpp"

#include <sstream>
#include <string>

#include "cli/session.hpp"
#include "test_framework.hpp"

using namespace milo;
using namespace milo::runtime;

namespace {

/// The directory holding the test data files.
std::string dataDir() {
#ifdef MILO_TEST_DATA_DIR
  return MILO_TEST_DATA_DIR;
#else
  return ".";
#endif
}

/// Run a program and return the value of its last statement.
Value run(const std::string& source) {
  cli::Session session("test.milo");
  const cli::RunOutcome outcome = session.run(source);
  if (outcome != cli::RunOutcome::Ok) {
    ::milo::test::fail(__FILE__, __LINE__,
                       "the program failed:\n" +
                           session.diagnostics().render(source, "test.milo"));
  }
  return session.lastValue();
}

/// Run a program that is expected to fail, and return the error message.
std::string runExpectingFailure(const std::string& source) {
  cli::Session session("test.milo");
  const cli::RunOutcome outcome = session.run(source);
  if (outcome == cli::RunOutcome::Ok) {
    ::milo::test::fail(__FILE__, __LINE__, "expected the program to fail, but it did not");
  }
  return session.diagnostics().render(source, "test.milo");
}

}  // namespace

TEST(interpEvaluatesArithmetic) {
  CHECK_EQ(run("1 + 2").asInt(), static_cast<std::int64_t>(3));
  CHECK_EQ(run("2 * 3 + 4").asInt(), static_cast<std::int64_t>(10));
  CHECK_EQ(run("(2 + 3) * 4").asInt(), static_cast<std::int64_t>(20));
  CHECK_EQ(run("10 - 4 - 3").asInt(), static_cast<std::int64_t>(3));
  CHECK_EQ(run("2 ** 10").asInt(), static_cast<std::int64_t>(1024));
  CHECK_EQ(run("-2 ** 2").asInt(), static_cast<std::int64_t>(-4));
  CHECK_EQ(run("7 // 2").asInt(), static_cast<std::int64_t>(3));
  CHECK_EQ(run("7 % 3").asInt(), static_cast<std::int64_t>(1));
}

TEST(interpEvaluatesFloats) {
  CHECK(run("1.5 + 2.5").asFloat() > 3.99);
  CHECK(run("3.0 / 2").asFloat() > 1.49);
}

TEST(interpEvaluatesComparisons) {
  CHECK_EQ(run("1 < 2").asBool(), true);
  CHECK_EQ(run("1 == 1.0").asBool(), true);
  CHECK_EQ(run("1 != 2").asBool(), true);
  CHECK_EQ(run("\"a\" == \"a\"").asBool(), true);
}

TEST(interpEvaluatesBooleanOperators) {
  CHECK_EQ(run("true and false").asBool(), false);
  CHECK_EQ(run("true or false").asBool(), true);
  CHECK_EQ(run("not true").asBool(), false);
}

TEST(interpHandlesVariables) {
  CHECK_EQ(run("let x = 5\nx + 1").asInt(), static_cast<std::int64_t>(6));
  CHECK_EQ(run("let x = 5\nset x to 7\nx").asInt(), static_cast<std::int64_t>(7));
  CHECK_EQ(run("let x = 5\nx = 9\nx").asInt(), static_cast<std::int64_t>(9));
}

TEST(interpHandlesIfElse) {
  const std::string program =
      "let x = 5\n"
      "let y = 0\n"
      "if x > 3:\n"
      "    y = 1\n"
      "else:\n"
      "    y = 2\n"
      "y\n";
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(1));
}

TEST(interpHandlesLoops) {
  const std::string program =
      "let total = 0\n"
      "for i in range(5):\n"
      "    total = total + i\n"
      "total\n";
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(10));
}

TEST(interpHandlesWhileLoops) {
  const std::string program =
      "let i = 0\n"
      "while i < 4:\n"
      "    i = i + 1\n"
      "i\n";
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(4));
}

TEST(interpHandlesBreakAndContinue) {
  const std::string program =
      "let total = 0\n"
      "for i in range(10):\n"
      "    if i == 3:\n"
      "        continue\n"
      "    if i == 6:\n"
      "        break\n"
      "    total = total + i\n"
      "total\n";
  // 0 + 1 + 2 + 4 + 5
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(12));
}

TEST(interpHandlesFunctions) {
  const std::string program =
      "fn add(a, b):\n"
      "    return a + b\n"
      "add(2, 3)\n";
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(5));
}

TEST(interpHandlesRecursion) {
  const std::string program =
      "fn fact(n):\n"
      "    if n <= 1:\n"
      "        return 1\n"
      "    return n * fact(n - 1)\n"
      "fact(5)\n";
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(120));
}

TEST(interpHandlesClosures) {
  const std::string program =
      "fn make_adder(n):\n"
      "    fn add(x):\n"
      "        return x + n\n"
      "    return add\n"
      "let add5 = make_adder(5)\n"
      "add5(3)\n";
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(8));
}

TEST(interpHandlesListsAndIndexing) {
  CHECK_EQ(run("let xs = [1, 2, 3]\nxs[1]").asInt(), static_cast<std::int64_t>(2));
  CHECK_EQ(run("let xs = [1, 2, 3]\nxs[-1]").asInt(), static_cast<std::int64_t>(3));
  CHECK_EQ(run("len([1, 2, 3])").asInt(), static_cast<std::int64_t>(3));
}

TEST(interpHandlesDictionaries) {
  const std::string program =
      "let d = {\"a\": 1, \"b\": 2}\n"
      "d[\"b\"]\n";
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(2));
}

TEST(interpHandlesTypeTests) {
  CHECK_EQ(run("1 is int").asBool(), true);
  CHECK_EQ(run("\"a\" is int").asBool(), false);
  CHECK_EQ(run("1.5 is float").asBool(), true);
}

TEST(interpHandlesTheSentenceForms) {
  // The plain-English sentences parse and run; they report that they do
  // not do any real work yet.
  const std::string program =
      "load model \"" + dataDir() + "/fixture.safetensors\" as base\n"
      "load data \"support_chats.csv\" as chats\n"
      "split chats into train and test by 90 percent\n"
      "fine-tune base on train\n"
      "    using lora with rank 16\n"
      "    precision is bf16\n"
      "    batch size 4\n"
      "    epochs 3\n"
      "    learning rate 0.0002\n"
      "    save to \"my-model.safetensors\"\n"
      "evaluate base on test and show accuracy\n"
      "use python \"math\" as math\n"
      "let scores be math.sqrt(0.25)\n"
      "scores\n";
  // Python ran for real: the square root of a quarter is a half.
  CHECK_NEAR(run(program).asFloat(), 0.5, 1e-6);
}

TEST(interpLoadsARealSafetensorsFile) {
  const std::string program =
      "load model \"" + dataDir() + "/fixture.safetensors\" as base\n"
      "base.parameters\n";
  CHECK_EQ(run(program).asInt(), static_cast<std::int64_t>(10));
}

TEST(interpReportsUnknownNames) {
  const std::string message = runExpectingFailure("chats");
  CHECK_CONTAINS(message, "cannot find");
  CHECK_CONTAINS(message, "chats");
}

TEST(interpSuggestsSimilarNames) {
  const std::string message = runExpectingFailure("let chats = 1\nchat");
  CHECK_CONTAINS(message, "did you mean");
  CHECK_CONTAINS(message, "chats");
}

TEST(interpReportsDivisionByZero) {
  const std::string message = runExpectingFailure("1 / 0");
  CHECK_CONTAINS(message, "divide by zero");
}

TEST(interpReportsTypeErrors) {
  const std::string message = runExpectingFailure("\"a\" + 1");
  CHECK_CONTAINS(message, "cannot apply");
}
