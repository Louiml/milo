#include "tensor/random.hpp"

#include <random>

namespace milo::tensor {
namespace {

std::mt19937& engine() {
  static std::mt19937 generator(20240101u);
  return generator;
}

}  // namespace

void setRandomSeed(std::uint32_t seed) { engine().seed(seed); }

double randomNormal() {
  std::normal_distribution<double> distribution(0.0, 1.0);
  return distribution(engine());
}

double randomUniform() {
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(engine());
}

}  // namespace milo::tensor
