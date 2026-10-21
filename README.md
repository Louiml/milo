# Milo

**A programming language for AI and data work.**

Milo aims to let you write an experiment the way you would describe it —
close to plain English — while remaining a fast, statically analysable,
compiled language with tensors, datasets, models, and training loops as
first-class features.

```milo
load model "llama-3-8b.safetensors" as base
load data "support_chats.csv" as chats

split chats into train and test by 90 percent

fine-tune base on train
    using lora with rank 16
    precision is bf16
    batch size 4
    epochs 3
    learning rate 0.0002
    save to "my-model.safetensors"

evaluate base on test and show accuracy
```

The same language, in explicit mode, when a sentence is not precise enough:

```milo
fn dot(a: list, b: list) -> float:
    let total = 0.0
    for i in range(len(a)):
        total = total + a[i] * b[i]
    return total
```

> **Status: Phase 7.** The language front end (lexer, parser, evaluator),
> the bytecode machine, the REPL, the CLI, the stable C ABI, the tensor
> engine with SIMD kernels, the safetensors reader and writer, automatic
> differentiation with optimisers and layers, quantisation, LoRA and QLoRA,
> the memory planner, the CUDA backend, the Python bridge, and the `mpm`
> package manager are done and tested. See
> [`docs/roadmap.md`](docs/roadmap.md) for the honest picture, including
> what is likely to be hard.

---

## Quick start

You need a C++20 compiler, CMake 3.24+, and (for the Rust package
manager) a Rust toolchain.

```bash
git clone <this repository>
cd Milo
python build.py            # debug build with sanitizers, then run the tests
python build.py --release  # optimised build
```

Then:

```bash
./build/debug/bin/milo run examples/hello.milo
./build/debug/bin/milo repl
./build/debug/bin/milo check examples/fine_tune.milo
./build/debug/bin/milo inspect tests/scripts/fixture.safetensors
```

On Windows the binary lands in `build\debug\bin\milo.exe`.

### Two engines

Milo has a tree-walking evaluator and a bytecode machine. The evaluator is
the reference: simple, and the thing the tests are written against. The
machine is the fast one.

```console
$ milo run program.milo          # the evaluator
$ milo run --vm program.milo     # the bytecode machine
```

They share the *operations* — the code that decides what `+` means — so
they cannot drift apart. And they are checked against each other: every
golden script is run by both, and eighteen dedicated tests compare them on
everything from arithmetic to a training loop.

### The GPU

The tensor engine has a **backend** interface: the CPU is one backend and
CUDA is another. When the machine has the CUDA toolkit, the backend is
built; when it does not, the build still works and the runtime says so.

`milo
let a = to_device(tensor([[1.0, 2.0], [3.0, 4.0]]), "cuda")
let b = to_device(tensor([[5.0, 6.0], [7.0, 8.0]]), "cuda")

print to_device(matmul(a, b), "cpu")
`

`
tensor<f32>[2, 2]
[19, 22]
[43, 50]
`

The GPU and the CPU are checked against each other, and they agree.

### Tensors

Tensors are first-class values, with a dtype and a shape:

```milo
let weights = tensor([1, 2, 3])
let biases = tensor([0.5, -0.5, 0.25])

let layer = add(mul(weights, 2.0), biases)
print layer
print shape(layer)
print dtype(layer)
```

The engine has AVX2 and NEON kernels chosen at run time, so one binary runs
on old and new hardware alike.

### Models

Milo reads and writes `.safetensors` files natively. The weights are
memory-mapped, so opening a 70 GB checkpoint costs nothing and only the
tensors you use become resident:

```milo
load model "llama-3-8b.safetensors" as base
print base.parameters
save base to "fine-tuned.safetensors"
```

And you can look at a file without running anything:

```console
$ milo inspect model.safetensors
model.safetensors
  format       safetensors
  file size    13.0 GiB (13,958,643,456 bytes)
  tensor data  12.9 GiB (13,850,000,000 bytes)
  tensors      291
  parameters   7,024,352,512
  ...
```

### Fitting on a small card

Quantised weights, low-rank updates, and a planner that tells you the
truth before you start:

```milo
let plan = memory_plan(model, batch_size = 4, sequence = 2048,
                       device_memory = 12, lora = true, lora_rank = 16)

print plan["fits"]
for suggestion in plan["suggestions"]:
    print suggestion["knob"] + ": " + suggestion["from"] + " -> " + suggestion["to"]
```

**The planner never changes anything.** It reports what would help and in
what order; applying it is your decision. The same answer is available
from the command line:

```console
$ milo plan model.safetensors --batch-size 4 --memory 12 --lora
  weights             13.0 GiB
  gradients          112.0 MiB
  optimizer state    224.0 MiB
  activations         22.0 GiB
  attention scores    32.0 GiB
  total               67.5 GiB

  it does not fit: it needs 12429607302 bytes of 12884901888

  What would make it fit, in the order to try it:
    1. batch_size: 4 -> 2
         smaller batches use less memory; raise the gradient accumulation to
         keep the effective batch the same
    2. gradient_checkpointing: off -> on
         ...
  Nothing has been changed. Apply these yourself if you want them.
```

Milo also has int8 and NF4 quantisation, LoRA and QLoRA layers, and a
fused matrix product that never holds a dequantised weight in memory.

### Training

Automatic differentiation is built in. A training loop is a loop:

```milo
let model = linear(3, 2)
let optimizer = adamw(model.parameters, 0.05)

for step in range(60):
    let prediction = model(inputs)
    let loss = mse_loss(prediction, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Every adjoint rule is checked against finite differences, so the gradients
are known to be right rather than merely plausible.

### Python

Milo and Python are two halves of one tool. A Milo program can use a
Python library, a Python program can use the Milo runtime, and tensors
move between them without being copied.

```milo
use python "numpy" as np

let scores = np.mean(np.array([0.25, 0.5, 0.75]))
print scores
```

```
0.5
```

```python
import milo
import numpy as np

tensor = milo.run("let a = tensor([1, 2, 3])\na")
array = np.asarray(tensor)
array[0] = 99.0
print(np.asarray(tensor))   # [99. 2. 3.] — one memory, two languages
```

NumPy sees Milo's tensors through the buffer protocol, and PyTorch
through DLPack. A Python exception inside a Milo program comes back as a
Milo error with Python's traceback attached.

### The build script

`build.py` is the one entry point. It drives both toolchains:

| | |
|---|---|
| `python build.py` | Debug build (ASan + UBSan) and run the tests |
| `python build.py --release` | Optimised build, LTO on |
| `python build.py --no-sanitizers` | Debug build without sanitizers (some MinGW distributions do not ship them) |
| `python build.py --no-rust` | C/C++ only |
| `python build.py --install` | Install into `build/<preset>/install` |
| `python build.py --clean` | Wipe `build/` and start over |

Requirements: CMake 3.24+, Ninja (recommended) or Make, a C++20 compiler
(MSVC 19.3x+, GCC 11+, Clang 14+), and Cargo 1.74+ if you want `mpm`.

### Running the tests

```bash
ctest --test-dir build/debug --output-on-failure
```

The suite has five parts:

| Test | What it covers |
|------|----------------|
| `test_lexer` | Tokens, indentation, strings, numbers, lexical errors |
| `test_parser` | The grammar, both modes, the sentence templates |
| `test_interp` | Evaluation: arithmetic, control flow, functions, errors |
| `test_directives` | The verb and option tables |
| `test_tensor` | Shapes, strides, views, broadcasting, `matmul`, conversions |
| `test_safetensors` | Round-trips, malformed headers, and a fuzz test |
| `test_autograd` | Every adjoint rule, against finite differences |
| `test_quantization` | int8 and NF4 round trips, the fused product, LoRA |
| `test_planner` | The memory estimates and the order of the suggestions |
| `test_vm` | The bytecode machine against the evaluator, program by program |
| `scripts` | Golden tests: `.milo` files and their expected output |
| `test_abi_c` | The public header compiles and works as plain C |

The Rust side has its own suite:

```bash
cargo test
```

---

## The two modes

Milo has one grammar with two statement families.

**Sentence mode** is a closed set of templates. A sentence is not free-form
English; it is a phrase drawn from a fixed vocabulary, which is what keeps
it unambiguous and analysable. The templates are data, in
[`src/core/directives.cpp`](src/core/directives.cpp):

```
load    <kind> "<path>" as <name>
split   <name> into <name> and <name> [by <n> percent]
fine-tune <model> on <dataset> [options...]
train   <model> on <dataset> [options...]
evaluate <model> on <dataset> [options...]
predict <model> on <dataset> [options...]
save    <model> to "<path>"
show    <metric>
use     python "<module>" as <name>
quantize <model> to <scheme>
```

Options may be written inline or on indented lines below the sentence:

```
fine-tune base on train
    using lora with rank 16
    precision is bf16
    batch size 4
```

**Explicit mode** is the escape hatch: typed variables, functions, loops,
and direct operations. A sentence desugars into an ordinary call, so both
modes compile to the same thing and the runtime never needs to know which
you wrote.

### Diagnostics

Errors are written in plain English, and say what to do about it when there
is one obvious fix:

```
error: I cannot find anything called 'chats'
 --> train.milo:4:1
  | split chats into train and test by 90 percent
  | ^^^^^ did you mean 'chat'?
```

---

## Repository layout

```
Milo/
├── build.py                the one build script (CMake + Cargo)
├── CMakeLists.txt          the C/C++ build
├── Cargo.toml              the Rust workspace
├── include/milo/milo.h     the stable C ABI — the only public surface
├── src/
│   ├── core/               lexer, parser, AST, diagnostics, JSON
│   │   └── vm/             the compiler and the bytecode machine
│   ├── tensor/             the tensor engine, its SIMD kernels, quantisation
│   ├── autograd/           the gradient tape, layers, and optimisers
│   ├── safetensors/        the .safetensors format: reader, writer, mmap
│   ├── planner/            what a training run would need
│   ├── runtime/            values, scopes, builtins, the evaluator
│   ├── abi/                the C ABI implementation
│   └── cli/                the `milo` executable, the REPL, inspect, plan
├── tools/mpm/              the Rust package manager
├── tests/                  unit tests, golden scripts, the C ABI test
├── examples/               runnable `.milo` programs
└── docs/
    ├── spec/               the language specification and the EBNF grammar
    ├── architecture/       VM, tensor engine, memory planner
    └── roadmap.md          milestones, and an honest risk assessment
```

## The C ABI

`include/milo/milo.h` is the only surface other code may use. It is plain
C, exception-free, and versioned.

The runtime is shipped twice on purpose: as a static library linked into the
`milo` executable (so the CLI is one self-contained file), and as a shared
library that exports **only** the C ABI. Foreign code links the shared
library, so the C++ implementation can change without breaking anyone.

```c
milo_runtime* rt = milo_runtime_create();
milo_value*   out = NULL;

milo_status status = milo_runtime_run(rt, "let x = 21\nx * 2\n", 0, "demo.milo", &out);
if (status != MILO_OK) {
  char buffer[512];
  milo_last_error(buffer, sizeof(buffer));
  fprintf(stderr, "%s", buffer);
}
milo_value_release(out);
milo_runtime_destroy(rt);
```

## The package manager

`mpm` is written in Rust and reaches the runtime only through that ABI.

```bash
mpm init my-project
mpm add "milo-numpy@^0.2"
mpm install
mpm runtime          # reports the runtime it is linked against
```

It has no Cargo dependencies, so it builds offline and is easy to audit.
The resolver, the lockfile, the cache, the SHA-256 checks, `--offline`,
`mpm build`, `mpm publish`, and `mpm search` all work; the registry
protocol is a static index, served over HTTP or kept in a directory.

```bash
mpm publish --registry .milo/registry   # publish to a local registry
mpm install --offline                   # install from the cache alone
```

---

## Design principles

1. **Unambiguous.** Every program has exactly one parse tree.
2. **Statically analysable.** Types, shapes, dtypes, and effects are
   knowable before running.
3. **Fast.** No mandatory heap allocation on the hot path; a path to
   native code stays open.
4. **Boring semantics.** No hidden metaprogramming, no implicit copies.

The full reasoning, including the trade-offs we rejected, is in
[`docs/spec/language-spec-v0.1.md`](docs/spec/language-spec-v0.1.md).

## Honest limitations

* Milo's tooling makes **fine-tuning** feasible on modest hardware. It
  does not make pretraining a large model from scratch cheap. A from-scratch
  1B-parameter pretrain still needs a cluster.
* The tensor engine is **single-threaded** on the CPU, and the CUDA
  backend is a reference implementation rather than a tuned one. There is
  no internal thread pool for large GEMMs.
* `matmul` is a straightforward blocked implementation, not a
  BLAS-quality GEMM. It is correct and reasonably fast, but it will not
  beat cuBLAS or MKL.
* The bytecode machine and the tree-walking evaluator agree, but only the
  evaluator has been profiled. The machine is the fast path; it is not yet
  the measured one.
* The Python bridge uses Python's main interpreter. An isolated
  sub-interpreter can be asked for, but NumPy's C extensions refuse to
  load in one, so it is not the default.
* The memory planner's estimate is a model, not a measurement. It will
  publish its accuracy against measured peak allocations.

## License

MIT. See [LICENSE](LICENSE).
