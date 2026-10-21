// Random numbers.
//
// The generator is seeded with a fixed value, so a program that uses
// random numbers is reproducible unless it says otherwise.
#ifndef MILO_TENSOR_RANDOM_HPP
#define MILO_TENSOR_RANDOM_HPP

#include <cstdint>

namespace milo::tensor {

/// Reseed the generator.
void setRandomSeed(std::uint32_t seed);

/// A number from the standard normal distribution.
double randomNormal();

/// A number from the uniform distribution on [0, 1).
double randomUniform();

}  // namespace milo::tensor

#endif  // MILO_TENSOR_RANDOM_HPP
