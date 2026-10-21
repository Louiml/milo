---
title: The tensor engine
description: Shapes, strides, storage, the SIMD kernels, and the backend interface.
section: Reference
order: 12
---
# Milo — Tensor engine design

Status: designed in Phase 0, implemented in Phase 2–5.

## 1. Data model

```
Tensor
  ├── handle    : shared_ptr<TensorImpl>
  │                 ├── dtype      : DType  (f32, f16, bf16, f64, i8, i32, i64, u8, bool)
  │                 ├── shape      : Shape  (vector<int64>)
  │                 ├── strides    : Strides (vector<int64>, element counts)
  │                 ├── storage    : shared_ptr<Storage>
  │                 │                 ├── device   : Device
  │                 │                 ├── bytes    : span<uint8>
  │                 │                 └── owner    : true (owns) | false (view)
  │                 ├── requires_grad : bool
  │                 └── grad       : weak/shared_ptr<Tensor>  (autograd)
  └── version    : uint32  (mutation counter; used to invalidate cached plans)
```

Two rules:

1. **A tensor handle is cheap to copy; a storage is not.** Copying a `Tensor`
   copies the handle. To duplicate the data you call `copy(x)`.
2. **A view shares storage.** `reshape`, `transpose`, `slice` produce views
   with adjusted strides. Mutations through a view are visible through the
   base. This is required for performance (attention, KV cache) and is also a
   correctness hazard, so views are marked and the type checker records
   aliasing when it can.

## 2. Layout

Default: **row-major contiguous**. A tensor is contiguous iff
`strides[i] == prod(shape[i+1:])` for all i, with the last stride 1.

Non-contiguous tensors are legal and supported. Ops that require contiguity
call `contiguous()`, which materialises a copy. Ops that can preserve layout
(reshape on a contiguous tensor, transpose, slice) avoid the copy.

A **general strided kernel** is the fallback: for elementwise ops, we iterate
with N-d index arithmetic. Broadcast shapes are supported by zeroing the
stride of broadcast dimensions.

## 3. Backend interface

```cpp
class Backend {
public:
  virtual ~Backend();
  virtual const char* name() const = 0;
  virtual bool supports(DType) const = 0;

  virtual void* allocate(size_t bytes, Device) = 0;
  virtual void  deallocate(void*) = 0;

  virtual void copy(Device dst, const void* src, Device src_, size_t bytes,
                    Stream) = 0;

  // Elementwise: out = f(a, b) with broadcasting.
  virtual void elementwise_unary(const Tensor&, Tensor& out, UnaryOp) = 0;
  virtual void elementwise_binary(const Tensor&, const Tensor&, Tensor& out,
                                  BinaryOp) = 0;

  // Linear algebra.
  virtual void matmul(const Tensor& a, const Tensor& b, Tensor& out,
                      bool trans_a, bool trans_b) = 0;

  // Reductions.
  virtual void reduce(const Tensor&, Tensor& out, ReduceOp, axes) = 0;

  // Quantized.
  virtual void dequantize(const Tensor& q, Tensor& out) = 0;
  virtual void matmul_nf4(const Tensor& qa, const Tensor& qb, Tensor& out) = 0;
};
```

Implementations: `CPUBackend` (always present), `CUDABackend` (Phase 5),
`MetalBackend` / `VulkanBackend` (later, community).

**Dispatch is on `(op, dtype, device, layout, contiguity)`, resolved once per
call site and cached on the tensor's shape signature.** The cache is the
single biggest source of tensor-engine performance; getting it wrong is how
you end up 5x slower than PyTorch for no visible reason.

## 4. CPU kernels and SIMD

* **x86-64**: AVX2 (baseline), with an AVX-512 path later. Compile the
  kernels with per-ISA object files and select at load time via
  `__builtin_cpu_supports` / `__cpuid`. Never use `-march=native` in shipped
  binaries.
* **aarch64**: NEON.
* **Fallback**: a scalar path with the same semantics, used for odd shapes
  and for testing.
* **F16/BF16**: on CPUs without native support, convert to f32 in blocks and
  convert back. On CPUs with `AVX512-BF16` / `AMX-BF16`, use the native path.

GEMM uses a blocked algorithm with register tiling (e.g. 4x4 or 6x16
micro-kernels) and a pack routine. Target: ≥ 80% of single-threaded OpenBLAS
`sgemm` on square matrices. A full BLAS-level GEMM is the correct goal but is
a known time sink — we start with a blocked micro-kernel and iterate.

## 5. Broadcasting

Rules match NumPy: dimensions are right-aligned; sizes must be equal or one of
them must be 1.

```
(2, 3) + (3,)   -> (2, 3)
(2, 3) + (1, 3) -> (2, 3)
(2, 3) + (2,)   -> error
```

## 6. Autograd integration

Autograd lives *above* the backend, in `src/autograd`. The tape records
(op, input tensors, output tensor) and backward rules call back into the
backend to compute the adjoint. This means:

* The backend never knows about gradients (keeps it simple, keeps the fast
  path clean).
* A custom op only needs a backward rule registered once.
* Gradient checkpointing = re-running forward under `no_grad` during backward.

## 7. Quantization

| Format | Granularity | Scales | Use |
|--------|-----------|--------|-----|
| int8 symmetric | per-tensor or per-channel | 1 f32 per channel | general |
| NF4 (QLoRA) | per-block of 64 | 1 f32 per block + absmax | LLM weights |
| bf16/f16 | n/a | n/a | compute |

NF4 block layout: 64 values packed as 4-bit nibbles (32 bytes) + one f32
scale + one f32 absmax. A f32 block is dequantized as
`v = (nibble_value - 8) * absmax / 8` where `nibble_value` is the codebook
index for the NF4 normal-float value.

The dequantize-on-the-fly kernel fuses dequantization with the GEMM inner
loop. Dequantizing to a temporary f32 buffer first is explicitly rejected:
it costs 2x the memory traffic and 2-3x the time.

## 8. Memory layout for inference vs training

* **Inference**: weights are mmap'd and read on demand. Resident memory grows
  only with the layers actually touched. The OS page cache acts as a free
  LRU cache for cold layers.
* **Training**: gradient-requiring weights are materialised in device memory
  (they need random access and in-place updates). Frozen weights (LoRA's
  base model) stay mmap'd and are read directly in the forward pass.
  This is exactly why LoRA makes fine-tuning fit on small GPUs.
