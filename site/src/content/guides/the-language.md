---
title: The language
description: Milo's whole language in one page — two modes, one runtime.
section: Guides
order: 2
---

# The language

Milo has one grammar with two statement families. They are the same
language, and they compile to the same thing.

## Sentence mode

A sentence is a fixed template. This is the point: Milo's English is a
**closed vocabulary of templates**, not free-form prose. That is what keeps
it unambiguous and analysable.

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

The options can go on the same line or on indented lines below. Both mean
the same thing.

The verbs are:

| Verb | Shape |
|------|-------|
| `load` | `load <kind> "<path>" as <name>` |
| `split` | `split <name> into <name> and <name> [by <n> percent]` |
| `fine-tune` | `fine-tune <model> on <dataset>` |
| `train` | `train <model> on <dataset>` |
| `evaluate` | `evaluate <model> on <dataset>` |
| `predict` | `predict <model> on <dataset>` |
| `save` | `save <model> to "<path>"` |
| `show` | `show <metric>` |
| `use` | `use python "<module>" as <name>` |
| `quantize` | `quantize <model> to <scheme>` |

## Explicit mode

Everything a sentence does, you can write out. When a sentence is not
precise enough, this is the way out.

```milo
let x = 42
let y: float = 3.5

fn add(a, b):
    return a + b

if x > 10:
    print "big"
elif x > 5:
    print "medium"
else:
    print "small"

for i in range(5):
    print i

while x > 0:
    x = x - 1
```

## Values

`int`, `float`, `bool`, `str`, `list`, `dict`, `tensor`, `model`, `layer`,
`optimizer`.

```milo
let xs = [1, 2, 3]
let ages = {"ada": 36, "alan": 41}
let ok = true
let nothing = null
```

## Functions

```milo
fn fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fn make_adder(n):
    fn add(x):
        return x + n
    return add
```

Functions are values. A closure captures the scope it was made in.

## Errors

Milo's errors are written in plain English, and they say what to do:

```
error: I cannot find anything called 'chats'
 --> train.milo:4:1
  | split chats into train and test by 90 percent
  | ^^^^^ did you mean 'chat'?
```

## Two engines

Milo has a tree-walking evaluator and a bytecode machine. The evaluator is
the reference; the machine is the fast one. They share the operations, so
they cannot disagree.

```bash
milo run program.milo          # the evaluator
milo run --vm program.milo     # the bytecode machine
```

Every golden test runs under both.

## The whole language

The [specification](../../../../docs/spec/language-spec-v0-1.md) has the exact
semantics of every construct, and the [grammar](../../../../docs/spec/grammar.md) has the
normative syntax.
