---
title: Architecture overview
description: How Milo is put together, from the lexer to the C ABI.
section: Reference
order: 10
---
# Milo — Architecture Overview

## 1. Layering

```
        ┌──────────────────────────────────────────────────────────┐
        │  User programs (.milo)  |  Python callers  |  mpm        │
        └───────────────┬──────────────────────┬───────────┬──────┘
                        │ C ABI (milo.h)       │ C ABI     │ C ABI
        ┌───────────────▼──────────────────────▼───────────▼──────┐
        │  ABI layer  (src/abi)  — opaque handles, error transfer  │
        └───────────────┬──────────────────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────────────────────────┐
        │  Language runtime                                          │
        │   core/      lexer · parser · AST · compiler · VM          │
        │   runtime/   values · environments · builtins · scheduler   │
        │   autograd/  tape · adjoint rules                          │
        │   python/    embedded CPython bridge                      │
        └───────────────┬──────────────────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────────────────────────┐
        │  Tensor engine (src/tensor)                               │
        │   Tensor · Storage · Strides · Shape · DType               │
        │   Backend interface: CPU (SIMD) | CUDA | Metal | Vulkan   │
        └───────────────┬──────────────────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────────────────────────┐
        │  I/O (src/safetensors)  mmap · header parse · writer      │
        └──────────────────────────────────────────────────────────┘
```

**The C ABI boundary is a hard rule.** `include/milo/milo.h` is the only
surface that the Rust package manager, the Python extension module, or any
third-party C/C++ extension may use. Everything above that line is C++20 and
may change freely; everything below it is frozen for the 0.x series and
versioned with `MILO_ABI_VERSION`.

The Rust `mpm` crate links `milo.h` directly. It does not link any C++ symbol.

## 2. Component responsibilities

### 2.1 `src/core` — front end

| Unit | Responsibility |
|------|----------------|
| `lexer` | Bytes ? tokens. Owns indentation state (INDENT/DEDENT), bracket nesting (NEWLINE suppression), contextual keyword recognition, hyphenated verbs. |
| `token` | Token kind, span (byte offset, line, column). |
| `ast` | Owning AST with `std::variant` node bodies and `unique_ptr` children. Nodes carry spans. |
| `parser` | Recursive descent for explicit mode; template matcher for sentence mode. Emits `Diagnostic` on failure rather than throwing. |
| `directives` | The verb/option template tables (data, not code). |
| `json` | A small, strict JSON parser, used by the safetensors reader. |
| `diagnostic` | Diagnostics and the renderer that draws them. |

### 2.2 `src/core/vm` — the bytecode machine

| Unit | Responsibility |
|------|----------------|
| `bytecode` | The opcodes, the chunks, and the compiled program. |
| `compiler` | The AST becomes bytecode. It never throws: a problem becomes a diagnostic. |
| `machine` | The stack machine. It uses the runtime's operations, so the two engines cannot disagree. |

Milo has **two execution engines**. The tree-walking evaluator is the
reference: it is simple, and it is what the tests are written against. The
bytecode machine is the fast one, and it is checked against the evaluator
program by program.

The two engines share the *operations* — the code that decides what `+`
means on two integers, two strings, or two tensors. That is the important
part: it is where a second engine would otherwise drift.

### 2.3 `src/tensor` — tensor engine

| Unit | Responsibility |
|------|----------------|
| `types` | `DType`, `Shape`, `Strides`, `Device`, and broadcasting rules. |
| `convert` | Element conversion, including exact F16 and BF16 rounding. |
| `tensor` | The handle: storage, dtype, shape, strides, and the view operations. |
| `ops` | Elementwise maths, broadcasting, reductions, `matmul`, `softmax`. |
| `quantization` | int8 and NF4 quantisation, and the fused matrix product. |
| `backend` | The `Backend` interface, the registry, and the device operations. |
| `cuda/` | The CUDA backend: the device's memory, the copies, and the kernels. |
| `kernels*` | The scalar, AVX2, and NEON kernels, and the run-time dispatch between them. |

See [`tensor-engine.md`](./tensor-engine.md).

The tensor engine does not know how a tensor's memory is stored. It asks
a **backend**, and the backend answers. The CPU is one backend and CUDA is
another; a Metal or Vulkan backend would be a third. The interface is
deliberately narrow, so that a new backend is a day's work.

The CUDA backend is optional. When the toolkit is not there, the build
simply has no GPU backend, and the runtime says so rather than failing
oddly.

### 2.4 `src/autograd` — the gradient tape
| Unit | Responsibility |
|------|----------------|
| `tape` | The graph of the current forward pass, and the backward pass. |
| `ops` | The differentiable operations and their adjoint rules. |
| `modules` | Layers (`linear`, `layer_norm`, `lora`, `qlora`) and optimisers (SGD, Adam, AdamW). |
| `state` | The moments and the step count, shared between copies of an optimiser. |

The autograd layer sits **above** the tensor engine and knows nothing about
it. A tensor that does not ask for a gradient never touches the tape, so
the fast path stays free of tape allocation.

### 2.5 `src/safetensors` — the file format
| Unit | Responsibility |
|------|----------------|
| `format` | The header structures, the strict validation, and the header writer. |
| `mapping` | A read-only memory mapping, on Windows and POSIX. |
| `reader` | The file: header validation and zero-copy tensor views. |
| `writer` | The writer: lays the data out and streams it out. |

### 2.6 `src/planner` — what a run would need

The memory planner estimates the memory a training run needs, and reports
what could be changed. It never changes anything itself: applying the
suggestions is the user's decision.

### 2.7 `src/runtime` — execution

`Value` is a tagged union over: null, bool, int, float, string, bytes, list,
dict, tensor, model, dataset, layer, optimiser, iterator, function, builtin.

Ownership rule: `Value` is copyable but may hold a `shared_ptr` to a
reference-counted payload (tensor storage, function closure). Copying a `Value`
copies the *handle*, not the payload. This is how Milo gets Python-like
convenience without Python-like accidental tensor copies — the tensor handle
carries an explicit "owned" vs "view" distinction at the storage layer.

`Environment` is a scope chain with `shared_ptr<Scope>` children; lookup
walks outward. Environments are captured by reference in closures.

### 2.8 `src/python`

Embedded CPython behind a C++ RAII wrapper. One `PyMiloContext` owns the
interpreter state, converts exceptions to `MiloError`, and exposes
`PyObject*`?`Value` conversion. A separate CPython extension module (`milo`)
exposes the same C ABI upward to Python callers.

### 2.9 `src/abi`

Implements `include/milo/milo.h`. Owns the global runtime instance, error
transfer (thread-local last-error buffer), and the stable entry points:

```
milo_abi_version()      -> uint32
milo_version_string()   -> const char*
milo_status_to_string() -> const char*
milo_run_source(src, len, out_result, out_error) -> milo_status
milo_last_error(buf, cap) -> size_t
milo_check_source(...) -> milo_status
```

The ABI is **exception-free**: no C++ exception may cross the boundary.
Internal errors become `milo_status` codes plus a thread-local error buffer.

## 3. Build graph

```
milo_core     (static lib)   lexer · parser · AST · diagnostics · json
  └── milo_tensor (static lib)  tensor engine + SIMD kernels
        ├── milo_autograd (static lib)  the gradient tape + layers + optimisers
        │     └── milo_safetensors (static lib)  the .safetensors format
        │           └── milo_runtime (static lib)  values · scopes · builtins
        │                 ├── milo_abi (static lib)  the C ABI, in the CLI
        │                 ├── milo_cli (static lib)  session · repl · inspect
        │                 │     └── milo (executable)
        │                 └── milo_shared (shared library, only the C ABI)
        │                       └── mpm (Rust, links the shared library)
        └── milo_safetensors
```

`mpm` (Rust) links the shared library and reaches the runtime only through
the C ABI in `include/milo/milo.h`.

Two artefacts exist for the ABI on purpose:

* `milo_abi` (static) is linked into the `milo` executable, so the CLI is
  one self-contained file with no DLL to ship.
* `milo_shared` (shared) is what foreign code links against. Only the
  `milo_*` symbols are exported; every C++ symbol stays hidden, so the
  implementation can change without breaking anyone. On ELF platforms the
  C++ symbols are hidden with `-fvisibility=hidden`; on Windows an explicit
  `.def` file lists the exports.

CMake builds the C++ side; Cargo builds the Rust side. `build.py` drives both
and knows how to pass the right include and library paths to Cargo.

### 3.1 Why the ABI sits on top

The C ABI is implemented in C++ but *exposes* only C. It sits above the
runtime rather than below it, so that the runtime can stay a plain C++
library and the ABI can be a thin, exception-free shim. The important
property is unchanged: `include/milo/milo.h` is the only header that other
code may include, and it is frozen for the 0.x series.

## 4. Threading model (v0.1)

* The language runtime is **single-threaded**. No threads are spawned by the
  language surface in v0.1.
* The tensor engine may use an internal thread pool for large GEMMs and
  reductions, but only from a single calling thread.
* `mpm` may parallelise downloads. The runtime ABI calls are thread-safe
  except for the thread-local error buffer, which is per-thread by
  construction.

## 5. Directory layout

```
Milo/
├── CMakeLists.txt            top-level C++ build
├── CMakePresets.json         named configurations
├── build.py                  the one build script (drives cmake + cargo)
├── Cargo.toml                workspace manifest for the Rust side
├── VERSION                   single source of truth for the version
├── include/milo/
│   ├── milo.h                the stable C ABI  (frozen)
│   ├── milo.def              the Windows export list for the shared library
│   └── milo/…                internal C++ headers
├── src/
│   ├── core/                 lexer, parser, ast, diagnostics, json
│   ├── tensor/               the tensor engine and its SIMD kernels
│   ├── safetensors/          the file format: reader, writer, mapping
│   ├── runtime/              values, scopes, builtins, the evaluator
│   ├── abi/                  implementation of the C ABI
│   └── cli/                  the `milo` executable, the REPL, inspect
├── tools/mpm/                Rust package manager
├── tools/make_fixture.py     builds the safetensors test fixture
├── tests/                    C++ unit tests (CTest) + golden scripts
├── examples/                 runnable .milo programs
└── docs/                     spec, architecture, guides
```

## 6. Build configuration matrix

| Config | Compiler | Flags | Use |
|--------|----------|-------|-----|
| `Debug` | any | `-O0 -g3`, ASan+UBSan on MSVC/GCC/Clang | development, fuzzing |
| `Release` | any | `-O3 -DNDEBUG`, LTO | distribution, benchmarks |
| `RelWithDebInfo` | any | `-O2 -g` | profiling |

Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer are enabled by
default in `Debug` on GCC/Clang. MSVC's `/fsanitize=address` is supported
where the toolset provides it. The `fuzz` preset adds `-fsanitize=fuzzer` on
Clang only.

## 7. Testing strategy

| Layer | Mechanism |
|-------|-----------|
| Lexer / parser | Table-driven unit tests in `tests/`; every grammar production has at least one positive and one negative case. |
| Evaluator | Script-level tests: a `tests/scripts/*.milo` corpus, each with an expected-output file. |
| Evaluator vs VM | Differential test: same script, both engines, outputs must match exactly. |
| Tensor engine | Property tests: shapes, dtypes, contiguity invariants; comparisons against a slow reference implementation written in the same repo. |
| Safetensors | Round-trip vs. the upstream Python library; a fuzz target that feeds random bytes to the parser. |
| Autograd | `gradcheck` against finite differences for every differentiable op. |
| ABI | A C test program (compiled as C, not C++) proves the header is C-clean. |