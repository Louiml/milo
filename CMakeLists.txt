cmake_minimum_required(VERSION 3.24)

project(Milo
        VERSION 0.1.0
        DESCRIPTION "Milo ג€” a programming language for AI and data work"
        LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(MILO_BUILD_TESTS      "Build the unit test suite"        ON)
option(MILO_BUILD_CLI        "Build the `milo` executable"       ON)
option(MILO_ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" ON)
option(MILO_ENABLE_LTO       "Enable link-time optimization in Release" ON)
option(MILO_WERROR          "Treat warnings as errors"          OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The Python extension is a shared object that links the static libraries,
# so everything has to be built position-independent. Some distributions
# compile with -fPIE by default, which the linker refuses to put in a
# shared object.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
include(CheckLanguage)

# ---------------------------------------------------------------------------
# The Python bridge.
#
# It is optional: without Python the build still works, and the runtime
# says so when a program asks for it.
#
# CPython on Windows only ships the release C runtime, and a program that
# mixes two of them can free a string in the wrong heap. So when Python is
# here, the whole build uses the release runtime.
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Embed QUIET)

set(MILO_HAVE_PYTHON OFF)
if(Python3_Interpreter_FOUND AND Python3_Development.Embed_FOUND)
  set(MILO_HAVE_PYTHON ON)
  message(STATUS "Milo: building the Python bridge (${Python3_VERSION})")

  if(MSVC)
    # The dynamic release runtime, so that the program and CPython share
    # one heap.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "" FORCE)
    set(MILO_CUDA_CRT "/MD")
    set(MILO_CUDA_DEFINES -D_ITERATOR_DEBUG_LEVEL=0 -DNDEBUG)
  endif()
else()
  set(MILO_CUDA_CRT "$<IF:$<CONFIG:Debug>,/MDd,/MD>")
  set(MILO_CUDA_DEFINES
    "$<IF:$<CONFIG:Debug>,-D_ITERATOR_DEBUG_LEVEL=2,-D_ITERATOR_DEBUG_LEVEL=0>"
    "$<IF:$<CONFIG:Debug>,-D_DEBUG,-DNDEBUG>")
endif()

# ---------------------------------------------------------------------------
# Warning flags, per compiler
# ---------------------------------------------------------------------------
add_library(milo_warnings INTERFACE)

if(MSVC)
  target_compile_options(milo_warnings INTERFACE
    /W4 /permissive- /utf-8 /Zc:__cplusplus /MP)
  if(MILO_WERROR)
    target_compile_options(milo_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(milo_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual)
  if(MILO_WERROR)
    target_compile_options(milo_warnings INTERFACE -Werror)
  endif()
endif()

# ---------------------------------------------------------------------------
# Sanitizers (Debug only)
# ---------------------------------------------------------------------------
add_library(milo_sanitizers INTERFACE)
if(MILO_ENABLE_SANITIZERS
   AND NOT CMAKE_BUILD_TYPE STREQUAL "Release"
   AND NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  if(MSVC)
    # MSVC injects the AddressSanitizer runtime through the compiler
    # driver; the linker needs no extra flag.
    target_compile_options(milo_sanitizers INTERFACE
      /fsanitize=address
      # MSVC's AddressSanitizer changes the standard library's container
      # annotations, and the two settings cannot be mixed in one program.
      # The CUDA backend cannot use /fsanitize=address, so the same defines
      # are given to both.
      -D_DISABLE_STRING_ANNOTATION
      -D_DISABLE_VECTOR_ANNOTATION)
  else()
    target_compile_options(milo_sanitizers INTERFACE
      -fsanitize=address,undefined -fno-omit-frame-pointer
      -fno-sanitize-recover=undefined)
    target_link_options(milo_sanitizers INTERFACE
      -fsanitize=address,undefined)
  endif()
endif()


# On Windows, nvcc needs to be told where the host compiler is.
if(MSVC AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  find_program(MILO_CL_EXE NAMES cl.exe)
  if(MILO_CL_EXE)
    set(CMAKE_CUDA_HOST_COMPILER "${MILO_CL_EXE}")
  endif()
endif()

# The CUDA toolkit is not always on the path; look where it usually is.
if(NOT DEFINED MILO_NVCC AND DEFINED ENV{CUDA_PATH})
  find_program(MILO_NVCC NAMES nvcc nvcc.exe HINTS "$ENV{CUDA_PATH}/bin")
endif()

set(MILO_HAVE_CUDA OFF)
if(MILO_NVCC)
  # The CUDA code is compiled by nvcc rather than by CMake's CUDA language
  # support: that support runs the Visual Studio environment, which is not
  # always usable. nvcc needs a recent Visual Studio, so the version check
  # is turned off; the kernels use no standard library at run time.
  # AddressSanitizer changes the standard library's container annotations,
  # and the two settings cannot be mixed in one program. nvcc cannot use
  # /fsanitize=address, so the defines are given by hand.
  set(MILO_CUDA_ASAN_DEFINES "")
  if(MILO_ENABLE_SANITIZERS AND MSVC)
    set(MILO_CUDA_ASAN_DEFINES -D_DISABLE_STRING_ANNOTATION -D_DISABLE_VECTOR_ANNOTATION)
  endif()

  find_library(MILO_CUDART
    NAMES cudart cudart.lib
    HINTS "$ENV{CUDA_PATH}/lib/x64" "$ENV{CUDA_PATH}/lib64" "$ENV{CUDA_PATH}/lib/x64")

  # The name of the CUDA runtime library, so that the backend can look for
  # it before it calls into it. The library is loaded on demand: a machine
  # without CUDA can still run Milo, it simply has no GPU to talk to.
  set(MILO_CUDART_DLL "cudart64_12.dll")
  if(MILO_NVCC)
    execute_process(COMMAND "${MILO_NVCC}" --version
                    OUTPUT_VARIABLE MILO_NVCC_VERSION_TEXT
                    ERROR_QUIET)
    string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" MILO_CUDA_VERSION_MATCH
           "${MILO_NVCC_VERSION_TEXT}")
    if(MILO_CUDA_VERSION_MATCH)
      set(MILO_CUDA_VERSION_MAJOR "${CMAKE_MATCH_1}")
      if(MILO_CUDA_VERSION_MAJOR GREATER_EQUAL 12)
        set(MILO_CUDART_DLL "cudart64_${MILO_CUDA_VERSION_MAJOR}.dll")
      else()
        set(MILO_CUDART_DLL "cudart64_${MILO_CUDA_VERSION_MAJOR}0.dll")
      endif()
    endif()
  endif()

  set(MILO_CUDA_OBJECTS "")
  foreach(cuda_source cuda_backend)
    set(cuda_object "${CMAKE_CURRENT_BINARY_DIR}/milo_${cuda_source}.obj")
    add_custom_command(
      OUTPUT "${cuda_object}"
      COMMAND "${MILO_NVCC}" -std=c++17 -O3 -Xcompiler /utf-8
              -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
              # The kernel object goes into a static library that the
              # Python extension links, so it has to be position-independent.
              -Xcompiler -fPIC
              # nvcc uses the static C runtime by default; the rest of the
              # project uses the dynamic one, and the two cannot be mixed.
              -Xcompiler ${MILO_CUDA_CRT}
              # The standard library's iterator debugging has to agree with
              # the rest of the build, or the linker complains.
              ${MILO_CUDA_DEFINES}
              ${MILO_CUDA_ASAN_DEFINES}
              -DMILO_CUDART_DLL="${MILO_CUDART_DLL}"
              -allow-unsupported-compiler
              -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
              -c "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/cuda/${cuda_source}.cu"
              -o "${cuda_object}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/cuda/${cuda_source}.cu"
      COMMENT "Building ${cuda_source}.cu"
      VERBATIM)
    list(APPEND MILO_CUDA_OBJECTS "${cuda_object}")
  endforeach()

  add_library(milo_cuda STATIC ${MILO_CUDA_OBJECTS} src/tensor/cuda/cuda_version.cpp)
  target_include_directories(milo_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(milo_cuda PUBLIC MILO_HAVE_CUDA)
  target_link_libraries(milo_cuda PUBLIC milo_headers milo_warnings PRIVATE milo_sanitizers)
  if(MILO_CUDART)
    target_link_libraries(milo_cuda PUBLIC ${MILO_CUDART})
    target_compile_definitions(milo_cuda PUBLIC
      "MILO_CUDART_DLL=\"${MILO_CUDART_DLL}\"")
    if(WIN32)
      target_link_options(milo_cuda PUBLIC "/DELAYLOAD:${MILO_CUDART_DLL}")
      target_link_libraries(milo_cuda PUBLIC delayimp)
    endif()
  endif()
  set_target_properties(milo_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

  set(MILO_HAVE_CUDA ON)
  message(STATUS "Milo: building the CUDA backend (${MILO_NVCC})")
else()
  message(STATUS "Milo: no CUDA toolkit; building without the GPU backend")
endif()

# ---------------------------------------------------------------------------
# Public headers. `include/milo/milo.h` is the only surface that foreign
# code may use.
# ---------------------------------------------------------------------------
add_library(milo_headers INTERFACE)
target_include_directories(milo_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ---------------------------------------------------------------------------
# The Python bridge.
#
# The detection happened at the top of this file, because it decides which
# C runtime the whole build uses.
# ---------------------------------------------------------------------------
if(MILO_HAVE_PYTHON)
  add_library(milo_python STATIC
    src/python/python.cpp
    src/python/tensor_object.cpp
  )
  target_include_directories(milo_python PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(milo_python PUBLIC
    Python3::Python
    milo_headers
    milo_warnings
    PRIVATE milo_sanitizers)
  if(MSVC)
    # `getenv` is only read from, but MSVC's secure-CRT check does not
    # know that.
    target_compile_definitions(milo_python PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()

  # The extension module, so that Python can call Milo.
  add_library(milo_extension MODULE src/python/extension.cpp)
  target_include_directories(milo_extension PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(milo_extension PRIVATE
    milo_python
    milo_abi
    milo_sanitizers
    milo_warnings)
  # Python looks for `milo.cp312-win_amd64.pyd` on Windows and for
  # `milo.cpython-312-x86_64-linux-gnu.so` elsewhere; the ABI tag tells
  # us which is which.
  if(Python3_SOABI MATCHES "^\\.")
    set(MILO_PY_MODULE_TAG "${Python3_SOABI}")
  else()
    set(MILO_PY_MODULE_TAG ".${Python3_SOABI}")
  endif()
  if(WIN32)
    set(MILO_PY_MODULE_SUFFIX "${MILO_PY_MODULE_TAG}.pyd")
  else()
    set(MILO_PY_MODULE_SUFFIX "${MILO_PY_MODULE_TAG}.so")
  endif()
  set_target_properties(milo_extension PROPERTIES
    PREFIX ""
    OUTPUT_NAME "milo"
    # The import library of the extension is not used by anybody — Python
    # loads the module directly — and its name must not clash with the
    # import library of the shared runtime.
    ARCHIVE_OUTPUT_NAME "milo_extension"
    SUFFIX "${MILO_PY_MODULE_SUFFIX}")
endif()

# ---------------------------------------------------------------------------
# The tensor engine.
#
# The vectorised kernels live in their own translation units, compiled with
# the instruction set they need, and are chosen at run time. A binary built
# on a new machine still runs on an old one.
# ---------------------------------------------------------------------------
set(MILO_CORE_SOURCES
  src/core/diagnostic.cpp
  src/core/vm/compiler.cpp
  src/core/vm/machine.cpp
  src/core/token.cpp
  src/core/lexer.cpp
  src/core/ast.cpp
  src/core/directives.cpp
  src/core/parser.cpp
  src/core/json.cpp
)

set(MILO_RUNTIME_SOURCES
  src/runtime/value.cpp
  src/runtime/environment.cpp
  src/runtime/builtins.cpp
  src/runtime/builtins_tensor.cpp
  src/runtime/builtins_training.cpp
  src/runtime/modules.cpp
  src/runtime/sentences.cpp
  src/runtime/interp.cpp
)

set(MILO_AUTOGRAD_SOURCES
  src/autograd/tape.cpp
  src/autograd/ops.cpp
  src/autograd/modules.cpp
)

set(MILO_TENSOR_SOURCES
  src/tensor/types.cpp
  src/tensor/convert.cpp
  src/tensor/tensor.cpp
  src/tensor/ops.cpp
  src/tensor/backend.cpp
  src/tensor/kernels.cpp
  src/tensor/kernels_scalar.cpp
  src/tensor/random.cpp
  src/tensor/quantization.cpp
)

set(MILO_SAFETENSORS_SOURCES
  src/safetensors/format.cpp
  src/safetensors/mapping.cpp
  src/safetensors/reader.cpp
  src/safetensors/writer.cpp
)

set(MILO_PLANNER_SOURCES
  src/planner/planner.cpp
)

# The vectorised kernels, compiled for one instruction set each.
set(MILO_ISA_OBJECTS "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x86)$")
  set(MILO_X86 ON)
  add_library(milo_kernels_avx2 OBJECT src/tensor/kernels_avx2.cpp)
  target_compile_definitions(milo_kernels_avx2 PRIVATE MILO_X86)
  target_include_directories(milo_kernels_avx2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(MSVC)
    target_compile_options(milo_kernels_avx2 PRIVATE /arch:AVX2)
  else()
    target_compile_options(milo_kernels_avx2 PRIVATE -mavx2)
  endif()
  list(APPEND MILO_ISA_OBJECTS $<TARGET_OBJECTS:milo_kernels_avx2>)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(MILO_ARM ON)
  add_library(milo_kernels_neon OBJECT src/tensor/kernels_neon.cpp)
  target_compile_definitions(milo_kernels_neon PRIVATE MILO_ARM)
  target_include_directories(milo_kernels_neon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  list(APPEND MILO_ISA_OBJECTS $<TARGET_OBJECTS:milo_kernels_neon>)
endif()

add_library(milo_tensor STATIC ${MILO_TENSOR_SOURCES} ${MILO_ISA_OBJECTS})
target_include_directories(milo_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(milo_tensor PUBLIC milo_headers milo_warnings PRIVATE milo_sanitizers)
if(MILO_HAVE_CUDA)
  target_link_libraries(milo_tensor PUBLIC milo_cuda)
endif()
set_target_properties(milo_tensor PROPERTIES CXX_VISIBILITY_PRESET hidden
                                           VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# Core: lexer, parser, AST ג€” the front end, with no dependency on the
# runtime.
# ---------------------------------------------------------------------------
add_library(milo_core STATIC ${MILO_CORE_SOURCES})
target_include_directories(milo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(milo_core PUBLIC milo_headers milo_warnings PRIVATE milo_sanitizers)
# Only the C ABI may leave the shared library.
set_target_properties(milo_core PROPERTIES CXX_VISIBILITY_PRESET hidden
                                         VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# Autograd: the gradient tape, the differentiable operations, and the layers
# and optimisers.
# ---------------------------------------------------------------------------
add_library(milo_autograd STATIC ${MILO_AUTOGRAD_SOURCES})
target_include_directories(milo_autograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(milo_autograd PUBLIC milo_tensor milo_headers milo_warnings
                                    PRIVATE milo_sanitizers)

# ---------------------------------------------------------------------------
# The memory planner: what a training run would need, and what could be
# changed.
# ---------------------------------------------------------------------------
add_library(milo_planner STATIC ${MILO_PLANNER_SOURCES})
target_include_directories(milo_planner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(milo_planner PUBLIC milo_headers milo_warnings PRIVATE milo_sanitizers)

# ---------------------------------------------------------------------------
# Safetensors: reading and writing .safetensors files.
# ---------------------------------------------------------------------------
add_library(milo_safetensors STATIC ${MILO_SAFETENSORS_SOURCES})
target_include_directories(milo_safetensors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(milo_safetensors PUBLIC milo_tensor milo_headers milo_warnings
                                    PRIVATE milo_sanitizers)

# ---------------------------------------------------------------------------
# Runtime: values, environments, builtins, and the evaluator.
# ---------------------------------------------------------------------------
add_library(milo_runtime STATIC ${MILO_RUNTIME_SOURCES})
target_include_directories(milo_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(milo_runtime PUBLIC milo_core milo_tensor milo_autograd milo_safetensors
                                  milo_planner milo_warnings PRIVATE milo_sanitizers)
if(MILO_HAVE_PYTHON)
  target_link_libraries(milo_runtime PUBLIC milo_python)
endif()
set_target_properties(milo_runtime PROPERTIES CXX_VISIBILITY_PRESET hidden
                                               VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# The stable C ABI ג€” the only surface mpm / Python / extensions may use.
# ---------------------------------------------------------------------------
add_library(milo_abi STATIC
  src/abi/milo_abi.cpp
)
target_link_libraries(milo_abi
  PUBLIC  milo_headers milo_warnings
  PRIVATE milo_runtime milo_sanitizers
)

# ---------------------------------------------------------------------------
# The shared runtime: one artefact that foreign code links against.
#
# Only the C ABI is exported. On ELF platforms the C++ symbols are hidden
# with -fvisibility=hidden; on Windows an explicit .def file lists the
# exports.
#
# The shared library is built from its own copy of the sources, without
# sanitizers: MSVC's AddressSanitizer changes the standard library's
# container annotations, and two translation units with different
# annotation settings cannot be linked together. It also means the Rust
# package manager does not have to carry a sanitizer runtime.
# ---------------------------------------------------------------------------
set(MILO_SHARED_SOURCES "")
if(MILO_HAVE_PYTHON)
  list(APPEND MILO_SHARED_SOURCES src/python/python.cpp src/python/tensor_object.cpp)
endif()

add_library(milo_shared SHARED
  src/abi/milo_abi.cpp
  ${MILO_SHARED_SOURCES}
  ${MILO_CORE_SOURCES}
  ${MILO_RUNTIME_SOURCES}
  ${MILO_TENSOR_SOURCES}
  ${MILO_AUTOGRAD_SOURCES}
  ${MILO_SAFETENSORS_SOURCES}
  ${MILO_PLANNER_SOURCES}
  ${MILO_ISA_OBJECTS}
)
target_link_libraries(milo_shared PRIVATE milo_headers milo_warnings)
if(MILO_HAVE_PYTHON)
  target_link_libraries(milo_shared PRIVATE Python3::Python)
  if(MSVC)
    target_compile_definitions(milo_shared PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
endif()
target_include_directories(milo_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(milo_shared PROPERTIES
  OUTPUT_NAME milo
  # The import library gets its own name so that it does not collide with
  # the `milo` executable's own import library.
  # The import library keeps the natural name, so that `-l milo` finds it.
  ARCHIVE_OUTPUT_NAME milo
  # The debug symbols get their own name, so that they do not collide with
  # the `milo` executable's.
  PDB_NAME milo_shared
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)
if(WIN32)
  target_sources(milo_shared PRIVATE include/milo/milo.def)
  # A shared library gains nothing from incremental linking, and the
  # .ilk file would collide with the `milo` executable's.
  set_property(TARGET milo_shared APPEND_STRING PROPERTY LINK_FLAGS " /INCREMENTAL:NO")
endif()

# ---------------------------------------------------------------------------
# The `milo` executable
# ---------------------------------------------------------------------------
add_library(milo_cli STATIC
  src/cli/session.cpp
  src/cli/repl.cpp
  src/cli/inspect.cpp
  src/cli/plan.cpp
)
target_link_libraries(milo_cli PUBLIC milo_runtime milo_planner milo_warnings
                               PRIVATE milo_sanitizers)

if(MILO_BUILD_CLI)
  add_executable(milo src/cli/main.cpp)
  target_link_libraries(milo PRIVATE milo_cli milo_abi milo_warnings milo_sanitizers)
  # The executable's own import library gets a different name, so that
  # `lib/milo.lib` can belong to the shared runtime.
  set_target_properties(milo PROPERTIES ARCHIVE_OUTPUT_NAME milo_exe)
  # The two link steps both want `bin/milo.ilk`, so they must not run at
  # the same time.
  add_dependencies(milo milo_shared)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(MILO_BUILD_TESTS)
  enable_testing()

  add_library(milo_testing INTERFACE)
  target_include_directories(milo_testing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(milo_testing INTERFACE milo_cli milo_runtime milo_tensor
                                          milo_autograd milo_safetensors milo_planner milo_warnings)

  set(MILO_TESTS
    test_lexer
    test_parser
    test_interp
    test_directives
    test_tensor
    test_safetensors
    test_autograd
    test_planner
    test_quantization
    test_vm
  )
  if(MILO_HAVE_CUDA)
    list(APPEND MILO_TESTS test_cuda)
  endif()
  foreach(t IN LISTS MILO_TESTS)
    add_executable(${t} tests/${t}.cpp tests/test_main.cpp)
    target_link_libraries(${t} PRIVATE milo_testing milo_sanitizers)
    # The tests need to find the data files.
    target_compile_definitions(${t} PRIVATE
      MILO_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts")
    add_test(NAME ${t} COMMAND ${t})
  endforeach()

  # A program that embeds Python needs Python's own directory on the
  # path, so that python3xx.dll can be found. The same is true of the
  # script tests, which run programs that embed Python.

  # Script-level golden tests: run .milo programs and compare stdout.
  add_test(NAME scripts
           COMMAND ${CMAKE_COMMAND}
                   -DMILO_BIN=$<TARGET_FILE:milo>
                   -DMILO_SCRIPT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_scripts.cmake)

  # The public header must compile as plain C. This test proves it.
  add_executable(test_abi_c tests/test_abi_c.c)
  target_link_libraries(test_abi_c PRIVATE milo_abi milo_sanitizers)
  set_target_properties(test_abi_c PROPERTIES C_STANDARD 99)
  add_test(NAME test_abi_c COMMAND test_abi_c)

  if(MILO_HAVE_PYTHON)
    get_filename_component(MILO_PYTHON_DIR "${Python3_EXECUTABLE}" DIRECTORY)
    # The ENVIRONMENT property is a list, so every semicolon in the path
    # has to be escaped.
    string(REPLACE ";" "\\;" MILO_PATH_ESCAPED "$ENV{PATH}")
    set(MILO_PYTHON_TESTS ${MILO_TESTS} scripts test_abi_c)
    set_tests_properties(${MILO_PYTHON_TESTS} PROPERTIES
      ENVIRONMENT "PATH=${MILO_PYTHON_DIR}\;${MILO_PATH_ESCAPED}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS milo_abi
        EXPORT MiloTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(MILO_BUILD_CLI)
  install(TARGETS milo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
