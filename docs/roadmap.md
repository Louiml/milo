---
title: Roadmap
description: The phases, the milestones, and an honest account of what is likely to be hard.
section: Reference
order: 20
---
# Milo — Roadmap

Last updated: 2026-10-21. Companion to [`spec/language-spec-v0-1.md`](./spec/language-spec-v0-1.md).

## How to read this

Each phase ends with something you can **run**, not something you can read.
Every phase lists: what was built, how to run it, how to test it, what's next.

Status legend: `DONE` / `ACTIVE` / `NEXT` / `PLANNED` / `AT RISK`

---

## Phase 0 — Foundation

**Status: DONE**

- Repository layout, license, version file.
- Language spec v0.1 (semantics) and normative EBNF grammar.
- Architecture overview, VM design, tensor engine design, memory planner design.
- CMake build for C/C++, Cargo workspace for Rust, one top-level build script.
- Stable C ABI header `include/milo/milo.h` (this is the contract `mpm` uses).

**Exit criteria:** a clean checkout configures and builds on Linux, macOS, Windows
with either MSVC or GCC/Clang; `ctest` runs green.

---

## Phase 1 — Language front end

**Status: DONE** (2026-10-21)

- Lexer with indentation handling, contextual keywords, hyphenated verbs.
- AST.
- Recursive-descent parser for explicit mode.
- Template matcher for sentence mode (verbs + options), inline and block forms.
- Tree-walking evaluator (kept afterwards as a differential-testing oracle).
- REPL with multi-line block support.
- CLI: `milo run`, `milo check`, `milo repl`, `milo version`, `milo -e`.
- Plain-English diagnostics with "did you mean" suggestions.
- The stable C ABI, as a static library for the CLI and a shared library
  (exporting only the C ABI) for foreign code.
- The `mpm` crate, bound to that ABI.

**Exit criteria:** met. `milo run examples/hello.milo` works; 52 C++ tests
and 13 Rust tests pass; every example in the spec parses.

---

## Phase 2 — Tensors and the CPU tensor engine

**Status: DONE (engine) / NEXT (bytecode VM)**

- Tensor value type: shape, dtype, strides, and a shared storage. A tensor
  handle is cheap to copy; the data is not. Views (`reshape`, `transpose`,
  `slice`) share the storage; `contiguous()` is the explicit copy.
- CPU tensor engine: contiguous storage, broadcasting, elementwise maths,
  reductions, `matmul`, and a numerically stable `softmax`.
- SIMD: AVX2 and NEON kernels in their own translation units, compiled with
  the instruction set they need and chosen at run time, with a scalar
  fallback. A binary built on a new machine still runs on an old one.
- Element types: F32, F64, F16, BF16, I64, I32, I8, U8, BOOL, with exact
  conversions to and from F16 and BF16.
- The tensor type is wired into the language: `zeros`, `ones`, `full`,
  `randn`, `arange`, `eye`, `tensor`, `matmul`, `add`, `mul`, `relu`, `gelu`,
  `softmax`, `sum`, `mean`, `reshape`, `transpose`, and the rest.

**Exit criteria:** met for the engine. 26 tensor tests, including a
differential test of `matmul` against a naive triple loop.

### 2.1 The bytecode VM — moved, not dropped

The VM was originally part of this phase. It is now scheduled after
Phase 3, for one reason: **the safetensors reader is on the critical path
to the thing the project is for, and the VM is not.** A user who wants to
fine-tune a model needs `milo load "model.safetensors"` to work; nobody
needs a bytecode VM to get there.

The VM is also the riskiest part of the front end, because it has to agree
with the tree-walking evaluator to the last bit. Doing it while the tensor
engine was still settling would have meant debugging two new systems at
once.

The plan is unchanged, and the evaluator stays as the differential-testing
oracle: the VM lands as Phase 4a, before autograd.

---

## Phase 3 — Safetensors

**Status: DONE**

- Reader: 8-byte LE u64 header length, JSON header, data buffer.
  Strict validation: header length bounds, offset monotonicity, shape and
  dtype product against the byte length, overflow-safe arithmetic, and no
  overlapping ranges.
- mmap-backed zero-copy tensor views, on both Windows and POSIX. A tensor
  from a file shares the file's memory; it is read-only, so a bug cannot
  corrupt the checkpoint.
- Writer: streaming writer, no full-file buffering.
- Dtypes: the format's full set is recognised; the ones Milo's tensor
  engine holds are F64, F32, F16, BF16, I64, I32, I8, U8, and BOOL. The rest
  are reported by `milo inspect` and refused with a clear message.
- CLI: `milo inspect model.safetensors` — sizes, a table of tensors, a
  dtype breakdown, a shape breakdown, and the metadata.
- The language can load and save models: `load model "x.safetensors" as m`
  and `save m to "y.safetensors"`, plus a `save` builtin for a dictionary
  of tensors.

**Exit criteria:** met. 12 tests, including a fuzz test that corrupts a
valid file 400 ways and checks that every attempt either reads correctly
or fails cleanly. The reader agrees byte for byte with an independent
Python implementation of the format (`tools/make_fixture.py`).

---

## Phase 4 — Autograd and training

**Status: DONE (the VM, autograd, optimisers, layers, losses, the MLP loop) /
NEXT (the tiny transformer)**

- 4a. The **bytecode VM**: a compiler (AST → bytecode) and a stack machine
  with a chain of scopes. It uses the runtime's operations, so it and the
  tree-walking evaluator cannot disagree about what `+` means. **Done.**
- Reverse-mode autograd as a runtime facility (a tape of ops + adjoint
  rules). The tensor engine knows nothing about it.
- Gradient checking: every adjoint rule is checked against finite
  differences. 21 tests.
- Optimisers: SGD, Adam, AdamW. The moments and the step count are shared,
  so a method call on a copy of the optimiser still sees the same state.
- Losses: mean squared error, cross entropy.
- Layers: `linear`, `layer_norm`. A layer is a callable value with named
  parameters.
- A training loop on an MLP, written in the language:

  ```
  let model = linear(3, 2)
  let optimizer = adamw(model.parameters, 0.05)
  for step in range(60):
      let prediction = model(inputs)
      let loss = mse_loss(prediction, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  ```

- Tensor arithmetic in the language itself: `a + b`, `a * b`, `a < b` work
  on tensors, and go through the gradient tape.

**Exit criteria:** met. The gradient checks pass; a linear model converges
to a loss of 1e-12 in forty steps; the training loop is a golden test.

**How the two engines are kept honest.** The tree-walking evaluator is the
reference. The bytecode machine is checked against it, program by program:
every golden script is run by both engines and must produce the same
output, and eighteen dedicated tests compare them on arithmetic, strings,
variables, conditionals, loops, `break` and `continue`, functions,
recursion, closures, lists, dictionaries, booleans, short-circuit
evaluation, type tests, sentences, tensors, and a training loop.

**Not done:** the tiny transformer, gradient checkpointing, and gradient
accumulation in the runtime. They need the attention layer and the memory
planner, which come later.

---
## Phase 5 — The "no supercomputer" stack

**Status: DONE (quantisation, LoRA, QLoRA, the memory planner, the CUDA
backend) / NEXT (Metal, Vulkan, gradient checkpointing in the runtime)**

- **int8** quantisation, symmetric, one scale per output channel.
- **NF4** quantisation, one scale and one absolute maximum per block of 64
  values, with the standard sixteen-value codebook.
- A **fused matrix product** for a quantised weight: the weight is read
  one row at a time, dequantised into a small buffer, multiplied, and
  thrown away. The whole dequantised matrix is never in memory.
- **LoRA**: a frozen base with two small trained matrices, `a` of
  `[inputs, rank]` and `b` of `[rank, outputs]`. The update is
  `scaling * (a @ b)`, where the scaling is `alpha / rank`.
- **QLoRA**: the same, over a base weight stored in NF4.
- **Merging**: the adapters can be folded into the base weight, so what you
  save is an ordinary model.
- The **memory planner**: it estimates what a run needs, and it reports
  what could be changed, in the order that costs the least quality. **It
  never changes anything itself** — applying the suggestions is the user's
  decision, and the planner is built so that it cannot take that decision
  away.
- The CLI: `milo plan <model> --batch-size 4 --memory 12`.
- The language: `quantize`, `dequantize`, `lora`, `qlora`, `merge`,
  `memory_plan`.
- **The CUDA backend.** A `Backend` interface with a CPU implementation and
  a CUDA one: the device's memory, the copies, and the kernels — the
  elementwise maths and a tiled matrix product. The backend is optional:
  the build works without it, and the runtime says so. The GPU and the CPU
  are checked against each other, and they agree.
- The language: `to_device(tensor, "cuda")`, `device(tensor)`,
  `gpu_info()`.

**Exit criteria:** met. Twelve tests cover the quantisation round trips,
the fused matrix product, the LoRA gradients, and the merge. Seven more
cover the GPU, when there is one. The planner's suggestions are a golden
test.

**Not done:** the Metal and Vulkan backends, and gradient checkpointing
inside the runtime — the planner can already ask for it, but the runtime
does not yet do it.

---

## Phase 6 — Python interop

**Status: DONE** (2026-10-22)

- Embedded CPython (C API). The main interpreter is used by default; an
  isolated sub-interpreter can be asked for with `MILO_PYTHON_ISOLATED=1`
  (NumPy's C extensions refuse to load in one, so the default is the one
  that works).
- `use python "numpy" as np` binds a module namespace.
- Zero-copy tensor exchange via the buffer protocol (NumPy) and DLPack
  (PyTorch).
- A Python extension module `milo` so Python can call Milo
  (`milo.run(source)` → result, `milo.load(...)` → a `MiloTensor` with
  `__dlpack__`).
- Guard rails: Python exceptions become Milo errors with the Python traceback
  attached; the GIL is released around the program and taken again by the
  bridge when the program calls into Python.
- The C ABI gained the tensor accessors a foreign caller needs
  (`milo_value_tensor_data` and friends).

**Exit criteria:** `np` and `torch` interop both work with no copies in the
hot path; a Python `pytest` suite exercises the bridge. — **met**: 16
tests in `python/tests/test_interop.py`.

---

## Phase 7 — `mpm` package manager

**Status: DONE** (2026-10-22)

- Rust crate `mpm`, `milo.toml` manifest, `milo.lock` lockfile.
- Semantic versioning, dependency resolution with strict version agreement.
- Checksums (SHA-256) of every download, verified before anything is
  unpacked.
- Local cache, `--offline` mode, `mpm search`, `mpm publish`.
- Registry protocol v1: a static index, served over HTTP or kept in a
  directory. A JSON index, a JSON lockfile, and a dull `.mpkg` archive
  format — all written out by hand, so that `mpm` has no dependencies and
  can be built and audited without a network.
- Binds the runtime through `include/milo/milo.h` only — never through C++.

**Exit criteria:** `mpm add` a package with a transitive dependency tree,
`mpm build` produces a loadable artifact, and `mpm install --offline`
succeeds with the network disabled. — **met**: the end-to-end test in
`tools/mpm/tests/cli.rs` does exactly that.

---

## Phase 8 — Polish, docs, benchmarks

**Status: PLANNED**

- Full plain-English sentence layer (every `milo.nn` / `milo.optim` call has a
  sentence form).
- Documentation site: tutorial, reference, "how to fine-tune on your laptop".
- Benchmarks against PyTorch: MLP training, transformer forward/backward,
  matmul, tokenizer throughput.
- Editor support: LSP server, syntax highlighting for VS Code / Neovim.

---

## Honest risk assessment

The following are the parts most likely to blow up the schedule, in rough order
of danger.

1. **GPU backend (#5, #6).** Writing a correct, fast CUDA backend plus
   DLPack interop is a multi-month project on its own. *Mitigation:* ship the
   CPU path first and make the backend interface real before any CUDA code
   exists. A CPU-only Milo is still useful. The GPU work is separable.

2. **Autograd + views + in-place ops (#4).** Gradient correctness through
   views, transposes, and in-place updates is where every DL framework
   accumulates bugs. *Mitigation:* implement autograd over a small op set
   first (`add`, `mul`, `matmul`, `softmax`, `sum`), and forbid in-place
   mutation of tensors that require grad until the rules are solid.

3. **NF4 + dequant kernels (#5).** The numerics are easy; the *performance* of
   dequant-on-the-fly inside a GEMM is not. Expect a 3-5x slowdown vs. fp16 if
   done naively. *Mitigation:* plan a fused dequant+GEMM kernel from the start
   rather than dequantizing into a temporary buffer.

4. **Sentence-mode ergonomics (#8).** The hard part is not the grammar, it is
   deciding which sentences to support. Every template is a permanent API
   commitment. *Mitigation:* keep the template count small and make the explicit
   mode a complete escape hatch so no user is *forced* onto a bad sentence.

5. **Python interop (#6).** CPython embedding is well-trodden but every
   version bump and every free-threading change matters. *Mitigation:* pin to
   a supported CPython range, keep the bridge behind one C ABI, and never
   expose `PyObject*` in the public C header.

6. **Numerical reproducibility.** Bit-identical results across the interpreter,
   the VM, and the GPU backend are *not* simultaneously achievable in general.
   We will define reproducibility as "within 1e-6 relative for f32 GEMM on the
   same backend" and stop there.

### Non-goals

* **Pretraining a large model from scratch.** Milo's tooling makes *fine-tuning*
  feasible on modest hardware. It does not make pretraining cheap, and the
  docs will say so plainly. A from-scratch 1B-parameter pretrain still needs
  a cluster.
* **A general-purpose systems language.** Milo has no unsafe, no raw pointers,
  no manual memory management. That is a deliberate constraint, and it is the
  right one for the target audience.
* **Replacing NumPy/PyTorch.** Milo interoperates with them. It does not
  attempt to match their full API surface.
