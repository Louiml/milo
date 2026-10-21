---
title: Python and Milo together
description: Calling Python from Milo, calling Milo from Python, and moving tensors without copying.
section: Guides
order: 6
---

# Python and Milo together

Milo and Python are two halves of one tool. A Milo program can use a
Python library; a Python program can use the Milo runtime. Tensors move
between them without being copied.

## Milo calls Python

The `use python` sentence binds a Python module to a name.

```milo
use python "numpy" as np

let scores = np.mean(np.array([0.25, 0.5, 0.75]))
print scores
```

```
0.5
```

From then on, `np` is a module: read an attribute with `np.array`, call it
like any other function.

A NumPy array that comes back into Milo becomes a Milo tensor, with the
same memory underneath. Nothing is copied.

```milo
use python "numpy" as np

let a = np.array([1.0, 2.0, 3.0])
print shape(a)
print dtype(a)
```

```
[3]
f64
```

### When Python goes wrong

A Python exception inside a Milo program becomes a Milo error, with
Python's own traceback attached, so you can see where it went wrong.

```milo
use python "numpy" as np
np.mean(np.array([1, 2]), axis="not an axis")
```

```
error: python: TypeError: only integer scalar arrays can be converted to a
scalar index
  Traceback (most recent call last):
    File ".../numpy/_core/fromnumeric.py", line 3862, in mean
    ...
```

## Python calls Milo

The `milo` extension module lets Python run Milo programs and read
Milo's tensors.

```python
import milo

value = milo.run("1 + 2")
assert value == 3
```

`milo.run` gives back the value of the last expression. A tensor comes
back as a `MiloTensor`.

```python
import milo
import numpy as np

tensor = milo.run("let a = tensor([1, 2, 3])\na")
array = np.asarray(tensor)
print(array)          # [1. 2. 3.]
array[0] = 99.0
print(np.asarray(tensor))   # [99. 2. 3.] — the same memory
```

The array and the tensor share one buffer. Writing in NumPy is writing in
Milo.

### PyTorch

`MiloTensor` speaks DLPack, so PyTorch can take it directly.

```python
import milo
import torch

tensor = milo.run("let a = tensor([1, 2, 3])\na")
shared = torch.from_dlpack(tensor)
print(shared)     # tensor([1., 2., 3.])
```

No copy here either.

### Models on disk

`milo.load` reads one tensor out of a safetensors file.

```python
tensor = milo.load("model.safetensors", "layer.weight")
print(tensor.shape)   # (2, 3)
```

## The lock

While a Milo program runs inside Python, the lock is given up, so other
Python threads keep running. The bridge takes the lock again when the
program calls into Python.

```python
import threading
import milo

def work():
    ...

thread = threading.Thread(target=work)
thread.start()
milo.run("let total = 0\nfor i in range(200000):\n    total = total + i\ntotal\n")
```

The other thread is not blocked.

## Which interpreter?

Milo uses Python's main interpreter by default. An isolated
sub-interpreter — with its own global state — can be asked for with the
`MILO_PYTHON_ISOLATED` environment variable.

```console
$ MILO_PYTHON_ISOLATED=1 milo run program.milo
```

Not every Python library works that way. NumPy's C extensions refuse to
load in a sub-interpreter, so the main interpreter is the default.

## Building and testing

The extension module is built with everything else.

```console
$ python build.py
```

The tests live in `python/tests` and need `pytest`, `numpy`, and — for one
test — `torch`.

```console
$ python -m pytest python/tests -q
16 passed
```
