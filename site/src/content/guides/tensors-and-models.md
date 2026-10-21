---
title: Tensors and models
description: Milo's tensor type, and reading and writing safetensors files.
section: Guides
order: 3
---

# Tensors and models

## Tensors

A tensor has a **dtype** and a **shape**. Copying a tensor copies the
handle, not the data; a view shares it.

```milo
let a = tensor([1.0, 2.0, 3.0])
let b = zeros(2, 3)
let c = ones(2, 3)
let d = full(2, 3, 7.0)
let e = randn(2, 3)
let f = arange(0, 10, 2)
let g = eye(4)

print shape(a)      # [3]
print dtype(a)      # f32
```

The elementwise operations broadcast, in the way NumPy does:

```milo
print add(a, a)          # [2, 4, 6]
print mul(a, 2.0)        # [2, 4, 6]
print matmul(m, m)        # a matrix product
print softmax(scores)     # over the last axis
```

`reshape`, `transpose` and slicing make **views**: they do not copy the
data.

```milo
let view = reshape(a, [1, 3])
let flipped = transpose(view)
```

## Reading a model

Milo reads and writes `.safetensors` files natively. The weights are
memory-mapped, so opening a seventy-gigabyte checkpoint costs nothing and
only the tensors you actually use become resident.

```milo
load model "llama-3-8b.safetensors" as base

print base.parameters      # how many parameters there are
print base.tensors        # a dictionary of the tensors
print base.metadata       # the file's metadata
```

## Writing a model

```milo
save base to "fine-tuned.safetensors"
```

Or, from a dictionary of tensors:

```milo
save({"w": weights, "b": biases}, "weights.safetensors")
```

## Looking at a file

You do not need to run anything to see what is in a checkpoint:

```bash
milo inspect model.safetensors
```

```
model.safetensors
  format       safetensors
  file size    13.0 GiB (13,958,643,456 bytes)
  tensor data  12.9 GiB (13,850,000,000 bytes)
  tensors      291
  parameters   7,024,352,512
  metadata     2 entries

  name                          dtype  shape                     size
  model.layers.0.self_attn.q_proj.weight  F32  [4096, 4096]   64.0 MiB
  ...

  dtypes
    F32        289 tensors  12.9 GiB
    I64          2 tensors  16.0 KiB
```

## Quantisation

Two schemes are built in. **int8** is symmetric, with one scale per
output channel. **NF4** has one scale and one absolute maximum per block
of sixty-four values, and takes half a byte per weight.

```milo
let q = quantize(weights, "nf4")
let back = dequantize(q)

print q.scheme       # nf4
print q.bytes        # how much memory it takes
```

The matrix product with a quantised weight never holds the dequantised
matrix in memory: it reads the weight a row at a time.

## Low-rank updates

A LoRA layer has a frozen base and two small trained matrices, so a big
model can be fine-tuned on a small card.

```milo
let layer = lora(4096, 4096, 16)

print layer.parameters["lora_a"]   # [4096, 16]
print layer.parameters["lora_b"]   # [16, 4096]
```

`qlora` is the same thing over a base stored in NF4, which is even
smaller. `merge` folds the adapters back into the base, so what you save
is an ordinary model.
