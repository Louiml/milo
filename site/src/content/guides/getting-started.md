---
title: Getting started
description: Install the toolchain, build Milo, and run your first program.
section: Guides
order: 1
---

# Getting started

Milo is a programming language for AI and data work. This page takes you
from nothing to a working program in about five minutes.

## What you need

| | |
|---|---|
| **A C++20 compiler** | MSVC 19.3x, GCC 11+, or Clang 14+ |
| **CMake** | 3.24 or newer |
| **Rust** | 1.74 or newer — only for the `mpm` package manager |
| **Ninja** | Recommended, and much faster than Make |

On Linux:

```bash
sudo apt-get install build-essential cmake ninja-build
```

On macOS:

```bash
brew install cmake ninja
```

On Windows, install Visual Studio 2022 with the "Desktop development with
C++" workload, or the Build Tools.

## Build it

There is one build script, and it drives both toolchains:

```bash
git clone <this repository>
cd Milo
python build.py
```

That gives you a debug build with AddressSanitizer and UndefinedBehaviorSanitizer
turned on, and it runs the test suite. It takes a minute or two.

```bash
python build.py --release     # an optimised build
python build.py --no-rust     # C++ only
python build.py --no-sanitizers  # some toolchains do not ship them
```

The script works out the MSVC environment for you, even when
`vcvars64.bat` is unhappy.

## Run something

```bash
./build/debug/bin/milo run examples/hello.milo
```

```
hello, world
```

On Windows the binary is at `build\debug\bin\milo.exe`.

## The interactive prompt

```bash
./build/debug/bin/milo repl
```

```
Milo 0.1.0 (language 0.1)
Type an expression to see its value, or `help` for what you can do.
Press Ctrl-D (or Ctrl-Z on Windows) to leave.

milo> let x = 6
milo> let y = 7
milo> x * y
42
```

## Your first program

```milo
fn greet(who):
    return "hello, " + who

print greet("world")
```

```bash
./build/debug/bin/milo run hello.milo
```

## Where to go next

- [The language](./the-language.md) — the whole language in one page.
- [Tensors and models](./tensors-and-models.md) — reading and writing models.
- [Training a model](./training.md) — the training loop.
- [The command line](./the-command-line.md) — everything `milo` can do.
- [The specification](../../../../docs/spec/language-spec-v0-1.md) — the exact semantics.
