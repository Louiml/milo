// A tiny test framework.
//
// The project has no third-party dependencies, so tests use this instead of
// GoogleTest. It is deliberately small: register a test, assert, and let
// the runner report.
#ifndef MILO_TESTS_TEST_FRAMEWORK_HPP
#define MILO_TESTS_TEST_FRAMEWORK_HPP

#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace milo::test {

/// Thrown by a failed check to abandon the current test.
class TestFailure : public std::exception {
 public:
  explicit TestFailure(std::string message) : message_(std::move(message)) {}
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

struct TestCase {
  std::string name;
  std::function<void()> body;
};

std::vector<TestCase>& registry();

struct Registrar {
  Registrar(const std::string& name, std::function<void()> body);
};

/// Run every registered test. Returns the number of failures.
int runAll();

/// Report a failed check.
[[noreturn]] void fail(const char* file, int line, const std::string& message);

/// Is `T` printable with `operator<<`?
template <typename T, typename = void>
struct Streamable : std::false_type {};

template <typename T>
struct Streamable<T, std::void_t<decltype(std::declval<std::ostream&>()
                                           << std::declval<const T&>())>> : std::true_type {};

/// Render a value for a failure message, when it can be rendered at all.
template <typename T>
std::string describeValue(const T& value) {
  if constexpr (Streamable<T>::value) {
    std::ostringstream out;
    out << value;
    return out.str();
  } else {
    return "<value>";
  }
}

template <typename A, typename B>
void checkEqual(const char* file, int line, const char* expressionA, const char* expressionB,
                const A& a, const B& b) {
  if (!(a == b)) {
    std::ostringstream out;
    out << expressionA << " == " << expressionB << "\n"
        << "    left:  " << describeValue(a) << "\n"
        << "    right: " << describeValue(b);
    fail(file, line, out.str());
  }
}

template <typename A, typename B>
void checkNotEqual(const char* file, int line, const char* expressionA, const char* expressionB,
                   const A& a, const B& b) {
  if (a == b) {
    std::ostringstream out;
    out << expressionA << " != " << expressionB << "\n"
        << "    both sides were: " << describeValue(a);
    fail(file, line, out.str());
  }
}

/// Assert that two numbers are within `tolerance` of each other.
template <typename A, typename B, typename T>
void checkNear(const char* file, int line, const char* expressionA, const char* expressionB,
               const A& a, const B& b, const T& tolerance) {
  const double difference = static_cast<double>(a) - static_cast<double>(b);
  if (!(difference <= static_cast<double>(tolerance) &&
        difference >= -static_cast<double>(tolerance))) {
    std::ostringstream out;
    out << expressionA << " ~= " << expressionB << "\n"
        << "    left:  " << describeValue(a) << "\n"
        << "    right: " << describeValue(b) << "\n"
        << "    tolerance: " << describeValue(tolerance);
    fail(file, line, out.str());
  }
}

}  // namespace milo::test

#define MILO_TEST_CONCAT_INNER(a, b) a##b
#define MILO_TEST_CONCAT(a, b) MILO_TEST_CONCAT_INNER(a, b)

/// Define a test case.
#define TEST(name)                                                                   \
  static void name();                                                                \
  static ::milo::test::Registrar MILO_TEST_CONCAT(milo_registrar_, name)(#name, name); \
  static void name()

/// Assert that a condition holds.
#define CHECK(condition)                                                             \
  do {                                                                               \
    if (!(condition)) {                                                              \
      ::milo::test::fail(__FILE__, __LINE__, "expected " #condition);               \
    }                                                                                \
  } while (false)

/// Assert that two values are equal.
#define CHECK_EQ(a, b) ::milo::test::checkEqual(__FILE__, __LINE__, #a, #b, (a), (b))

/// Assert that two values are different.
#define CHECK_NE(a, b) ::milo::test::checkNotEqual(__FILE__, __LINE__, #a, #b, (a), (b))

/// Assert that two numbers are close to each other.
#define CHECK_NEAR(a, b, tolerance) \
  ::milo::test::checkNear(__FILE__, __LINE__, #a, #b, (a), (b), (tolerance))

/// Assert that a string contains a substring.
#define CHECK_CONTAINS(haystack, needle)                                                  \
  do {                                                                                    \
    const std::string milo_haystack = (haystack);                                         \
    const std::string milo_needle = (needle);                                             \
    if (milo_haystack.find(milo_needle) == std::string::npos) {                           \
      ::milo::test::fail(__FILE__, __LINE__,                                               \
                         "expected `" + milo_haystack + "` to contain `" + milo_needle + "`"); \
    }                                                                                     \
  } while (false)

#endif  // MILO_TESTS_TEST_FRAMEWORK_HPP
