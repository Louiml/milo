#include "test_framework.hpp"

#include <cstring>
#include <iostream>

namespace milo::test {

std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

Registrar::Registrar(const std::string& name, std::function<void()> body) {
  registry().push_back(TestCase{name, std::move(body)});
}

void fail(const char* file, int line, const std::string& message) {
  std::ostringstream out;
  out << file << ":" << line << ": " << message;
  throw TestFailure(out.str());
}

int runAll() {
  int failures = 0;
  for (const TestCase& test : registry()) {
    try {
      test.body();
      std::cout << "  ok    " << test.name << '\n';
    } catch (const TestFailure& failure) {
      std::cout << "  FAIL  " << test.name << "\n        " << failure.what() << '\n';
      ++failures;
    } catch (const std::exception& error) {
      std::cout << "  FAIL  " << test.name << "\n        unexpected exception: " << error.what()
                << '\n';
      ++failures;
    } catch (...) {
      std::cout << "  FAIL  " << test.name << "\n        unexpected exception\n";
      ++failures;
    }
  }

  std::cout << "\n"
            << registry().size() << " test(s), " << failures << " failure(s)\n";
  return failures;
}

}  // namespace milo::test

int main() {
  std::cout << "running " << milo::test::registry().size() << " test(s)\n";
  return milo::test::runAll() == 0 ? 0 : 1;
}
