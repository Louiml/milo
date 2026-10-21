---
title: The bytecode machine
description: The instruction set, the calling convention, and where the native code will go.
section: Reference
order: 11
---
# Milo — Bytecode VM design

Status: designed in Phase 0, implemented in Phase 2.

## 1. Why a VM and not LLVM-first

Milo needs three things from its execution strategy: fast numeric code, a
stable place to hang autograd, and a path to real native code. A bytecode VM
gives the first two quickly and the third later (JIT to LLVM, or a
pattern-matching JIT that fuses tensor ops).

Compiling straight to LLVM in Phase 1 would have put a code generator in front
of an unstable grammar. The grammar is the thing most likely to change; the
backend is the thing most expensive to change. The order is therefore
deliberate: **grammar → VM → native codegen.**

## 2. Machine model

A **stack machine** with one accumulator-free design: values are pushed and
popped from an operand stack, locals live in a frame's slot array.

```
frame:
  slots    : Value[slot_count]      // locals, temporaries, parameters
  ip       : instruction pointer
  closure  : shared_ptr<Closure>
  parent   : Environment*           // lexical scope for name lookups
```

* **Slots** hold locals. Slot indices are resolved at compile time, so
  variable access is an array index, not a hash lookup.
* **Upvalues** (closed-over variables) are handled by an open/closed
  upvalue scheme, like Lua: a scope keeps a list of upvalue descriptors; when
  a scope is popped, any upvalue still referenced is closed over the heap.
* **Globals** live in a single global scope with a name→slot index map.
* **Constants** (integers, floats, strings) are interned in a per-module
  constant pool.

This is a design choice worth defending: a register machine would be faster,
but a stack machine is dramatically simpler to get *correct*, and correctness
in the first VM matters more than a 20% throughput difference that a later
JIT can erase.

## 3. Instruction set (draft)

```
OP_CONST        k          push constant pool[k]
OP_NULL                     push null
OP_TRUE / OP_FALSE
OP_POP                     discard top
OP_DUP                      duplicate top
OP_LOAD_LOCAL   slot        push slots[slot]
OP_STORE_LOCAL  slot        slots[slot] = pop
OP_LOAD_GLOBAL  idx         push globals[idx]
OP_STORE_GLOBAL idx
OP_LOAD_UPVAL   idx
OP_STORE_UPVAL  idx
OP_LOAD_ATTR    k
OP_STORE_ATTR   k
OP_CALL         argc        call top-of-stack callee with argc args
OP_RETURN                   return top
OP_JUMP         off
OP_JUMP_IF_FALSE off
OP_JUMP_IF_TRUE  off
OP_GET_ITER
OP_FOR_ITER     off         iterator protocol
OP_ADD / OP_SUB / OP_MUL / OP_DIV / OP_MOD / OP_POW
OP_NEG / OP_NOT
OP_EQ / OP_NE / OP_LT / OP_LE / OP_GT / OP_GE
OP_INDEX        argc        x[i]
OP_SET_INDEX    argc
OP_CONCAT
OP_CLOSURE      const_idx, n_upvals
OP_TRY / OP_END_TRY          (v0.2, for error handling)
```

Numeric ops on `int` and `float` are specialised at compile time when the
static type is known (`OP_ADD_INT` vs `OP_ADD`), falling back to a generic
boxed op when the type is dynamic.

## 4. Tensor ops are not opcodes

Tensor operations are **not** VM opcodes. A tensor op compiles to a call to a
builtin function whose implementation lives in the tensor engine. This keeps
the VM small and lets the autograd tape intercept at exactly one place (the
builtin dispatcher) instead of at every opcode.

This is the same split CPython uses between the eval loop and `numpy`, and it
is the right one: tensor ops want to be fused, batched, and dispatched to
SIMD/GPU backends, none of which are VM concerns.

## 5. Compilation pipeline

```
source ──lex──▶ tokens ──parse──▶ AST ──lower──▶ HIR ──codegen──▶ bytecode
                                     │              │
                                     └──────────────┴──▶ diagnostics
```

* **AST** — surface syntax, owns spans.
* **HIR** — a small, span-preserving, name-resolved form: no `let`/`set`
  distinction (both become `Define`/`Assign`), explicit slot indices, resolved
  identifiers, desugared sentences. Sentences desugar here, so the VM never
  sees a "sentence".
* **bytecode** — instructions + a constant pool + function prototypes.

## 6. Sentences in the pipeline

A sentence desugars in the parser to an explicit-mode AST:

```
fine-tune base on train using lora with rank 16
```
becomes
```
milo.finetune(base, train, method="lora", rank=16)
```

The `milo` namespace is auto-imported. If the standard library defines a
sentence verb, the parser emits a call to it. A user program can therefore
always fall back to the explicit form, and a future "user-defined sentences"
feature only needs to extend the template table plus a namespace import.

## 7. Error handling and the stack

Milo has no `try/catch` in v0.1. Runtime errors unwind to the top level and
are reported with the Milo source location where the error occurred, plus the
call stack rendered as `fnName (file:line)`.

Exceptions are implemented with C++ exceptions inside the runtime, caught at
the VM's frame boundary, and converted into a `MiloError` value at the ABI
edge. **No C++ exception crosses the C ABI.**

## 8. Performance targets

| Benchmark | Target |
|-----------|--------|
| Scalar loop, 1e8 iterations | ≥ 1e9 simple ops/s (release, native CPU) |
| `for x in list` over 1e6 ints | ≤ 20 ms |
| Function call overhead | ≤ 15 ns |
| MLP step, batch 32, hidden 512, f32 | within 1.5x of PyTorch CPU |
| Transformer forward, 1M tokens/s parse rate | within 2x of PyTorch CPU |

The last two are the ones that matter; the first three just need to not be
embarrassing.
