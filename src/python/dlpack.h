// A minimal DLPack header.
//
// DLPack is the exchange format that PyTorch, JAX, CuPy and NumPy all
// speak. The struct layout is stable and part of the ABI, so it can be
// written out here rather than vendored: only the parts Milo needs are
// declared.
//
// The layout follows the DLPack specification exactly — the order of the
// fields matters, because a consumer reads the struct without knowing
// anything about the producer.
#ifndef MILO_PYTHON_DLPACK_H
#define MILO_PYTHON_DLPACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Where the data lives.
typedef enum {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLROCM = 10,
} DLDeviceType;

typedef struct {
  int32_t device_type;
  int32_t device_id;
} DLDevice;

/// The type of one element.
typedef enum {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
  kDLOpaqueHandle = 3,
  kDLBfloat = 4,
  kDLComplex = 5,
  kDLBool = 6,
} DLDataTypeCode;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

/// A tensor's shape and strides, without any ownership.
typedef struct {
  /// Where the numbers live.
  void* data;
  /// The device the numbers live on.
  DLDevice device;
  /// How many dimensions the tensor has.
  int32_t ndim;
  /// The type of one element.
  DLDataType dtype;
  /// The length of each dimension.
  int64_t* shape;
  /// The distance between elements, in elements rather than bytes.
  int64_t* strides;
  /// The offset of the first element, in bytes.
  uint64_t byte_offset;
} DLTensor;

/// A tensor, and who is responsible for freeing it.
typedef struct DLManagedTensor {
  /// The tensor itself.
  DLTensor dl_tensor;
  /// Whatever the producer needs to remember about this tensor.
  void* manager_ctx;
  /// Called when the consumer is done with the tensor.
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

/// The name of the capsule that carries a tensor.
#define MILO_DLPACK_CAPSULE_NAME "dltensor"

#ifdef __cplusplus
}
#endif

#endif  // MILO_PYTHON_DLPACK_H
