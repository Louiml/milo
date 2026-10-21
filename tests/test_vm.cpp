#include <sstream>
#include <string>
#include <vector>

#include "cli/session.hpp"
#include "test_framework.hpp"

using namespace milo;
using namespace milo::cli;

namespace {

/// What one engine did with a program.
struct Outcome {
  bool ran = false;
  std::string output;
  std::string error;
  std::string value;
};

/// Run a program with one engine, capturing what it printed.
Outcome runWith(Engine engine, const std::string& source) {
  Outcome outcome;

  Session session("test.milo");
  session.setEngine(engine);

  std::ostringstream out;
  std::ostringstream notes;
  session.interpreter().setOutput(&out);
  session.interpreter().setNotes(&notes);

  const RunOutcome result = session.run(source);
  outcome.output = out.str();
  outcome.error = notes.str();
  outcome.ran = result == RunOutcome::Ok;
  if (outcome.ran) {
    outcome.value = session.lastValue().toReprString();
  } else {
    outcome.value = session.diagnostics().render(source, "test.milo");
  }
  return outcome;
}

/// Run a program with both engines and check that they agree.
void checkBothEnginesAgree(const std::string& name, const std::string& source) {
  const Outcome evaluated = runWith(Engine::Evaluator, source);
  const Outcome compiled = runWith(Engine::Machine, source);

  if (!evaluated.ran) {
    // The program does not run at all; the machine must fail the same way.
    if (compiled.ran) {
      ::milo::test::fail(__FILE__, __LINE__,
                         name + ": the evaluator failed but the machine did not");
    }
    return;
  }

  if (!compiled.ran) {
    ::milo::test::fail(__FILE__, __LINE__,
                       name + ": the evaluator ran but the machine failed:\n" + compiled.value);
    return;
  }

  if (evaluated.output != compiled.output) {
    ::milo::test::fail(__FILE__, __LINE__,
                       name + ": the two engines printed different things.\n  the evaluator: " +
                           evaluated.output + "\n  the machine:    " + compiled.output);
    return;
  }

  if (evaluated.value != compiled.value) {
    ::milo::test::fail(__FILE__, __LINE__,
                       name + ": the two engines ended with different values.\n  the evaluator: " +
                           evaluated.value + "\n  the machine:    " + compiled.value);
  }
}

}  // namespace

TEST(theMachineAgreesWithTheEvaluatorOnArithmetic) {
  checkBothEnginesAgree("arithmetic", "print 1 + 2 * 3\nprint (1 + 2) * 3\nprint 10 / 4\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnStrings) {
  checkBothEnginesAgree("strings", "let a = \"hello\"\nprint a + \" \" + \"world\"\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnVariables) {
  checkBothEnginesAgree("variables", "let x = 5\nx = x + 1\nprint x\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnConditionals) {
  checkBothEnginesAgree("conditionals",
                        "let x = 7\n"
                        "if x > 5:\n"
                        "    print \"big\"\n"
                        "elif x > 2:\n"
                        "    print \"medium\"\n"
                        "else:\n"
                        "    print \"small\"\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnLoops) {
  checkBothEnginesAgree("loops",
                        "let total = 0\n"
                        "for i in range(5):\n"
                        "    total = total + i\n"
                        "print total\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnBreakAndContinue) {
  checkBothEnginesAgree("break and continue",
                        "let total = 0\n"
                        "for i in range(10):\n"
                        "    if i == 3:\n"
                        "        continue\n"
                        "    if i == 6:\n"
                        "        break\n"
                        "    total = total + i\n"
                        "print total\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnWhileLoops) {
  checkBothEnginesAgree("while",
                        "let i = 0\n"
                        "while i < 4:\n"
                        "    i = i + 1\n"
                        "print i\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnFunctions) {
  checkBothEnginesAgree("functions",
                        "fn add(a, b):\n"
                        "    return a + b\n"
                        "print add(2, 3)\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnRecursion) {
  checkBothEnginesAgree("recursion",
                        "fn fact(n):\n"
                        "    if n <= 1:\n"
                        "        return 1\n"
                        "    return n * fact(n - 1)\n"
                        "print fact(6)\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnClosures) {
  checkBothEnginesAgree("closures",
                        "fn make_adder(n):\n"
                        "    fn add(x):\n"
                        "        return x + n\n"
                        "    return add\n"
                        "let add5 = make_adder(5)\n"
                        "print add5(3)\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnListsAndDictionaries) {
  checkBothEnginesAgree("lists and dictionaries",
                        "let xs = [1, 2, 3]\n"
                        "print xs[1]\n"
                        "let d = {\"a\": 1, \"b\": 2}\n"
                        "print d[\"b\"]\n"
                        "print len(xs)\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnBooleans) {
  checkBothEnginesAgree("booleans",
                        "print true and false\n"
                        "print true or false\n"
                        "print not true\n"
                        "print 1 < 2\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnShortCircuit) {
  // `and` and `or` return the value that decided the answer.
  checkBothEnginesAgree("short circuit", "print 0 or \"fallback\"\nprint 1 and \"second\"\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnTypeTests) {
  checkBothEnginesAgree("type tests", "print 1 is int\nprint \"a\" is int\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnSentences) {
  checkBothEnginesAgree("sentences",
                        "load data \"chats.csv\" as chats\n"
                        "split chats into train and test by 90 percent\n"
                        "use python \"math\" as math\n"
                        "print \"done\"\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnTensors) {
  checkBothEnginesAgree(
      "tensors",
      "let a = tensor([1.0, 2.0, 3.0])\n"
      "let b = tensor([4.0, 5.0, 6.0])\n"
      "print add(a, b)\n"
      "print mul(a, b)\n"
      "print sum(a)\n"
      "print shape(a)\n");
}

TEST(theMachineAgreesWithTheEvaluatorOnTraining) {
  checkBothEnginesAgree(
      "training",
      "seed(3)\n"
      "let model = linear(2, 1)\n"
      "let optimizer = sgd(model.parameters, 0.05)\n"
      "let inputs = tensor([[1.0, 1.0]])\n"
      "let targets = tensor([[2.0]])\n"
      "for step in range(5):\n"
      "    let prediction = model(inputs)\n"
      "    let loss = mse_loss(prediction, targets)\n"
      "    optimizer.zero_grad()\n"
      "    loss.backward()\n"
      "    optimizer.step()\n"
      "print round(loss * 1000.0)\n");
}

TEST(theMachineFailsWhereTheEvaluatorFails) {
  const Outcome evaluated = runWith(Engine::Evaluator, "print undefined_name\n");
  const Outcome compiled = runWith(Engine::Machine, "print undefined_name\n");
  CHECK(!evaluated.ran);
  CHECK(!compiled.ran);
}
