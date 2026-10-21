// A plain C++ file in the CUDA backend, so that the build system knows how
// to link the library. Everything that touches CUDA is in the .cu files.
#include "tensor/cuda/cuda_backend.hpp"

namespace milo::tensor::cuda {

/// The version of the CUDA backend.
///
/// It is here so that the library has a C++ translation unit of its own.
const char* backendVersion() { return "0.1.0"; }

}  // namespace milo::tensor::cuda
