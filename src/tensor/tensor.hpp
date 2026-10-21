// The tensor handle: shape, dtype, strides, and a shared storage.
#ifndef MILO_TENSOR_TENSOR_HPP
#define MILO_TENSOR_TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensor/backend.hpp"
#include "tensor/types.hpp"

namespace milo::tensor {

/// A block of memory that one or more tensors share.
///
/// Storage is reference counted, which is what makes views cheap: a view
/// holds a reference to the same storage as the tensor it was made from.
class Storage {
 public:
  /// Allocate `bytes` of zeroed memory.
  ///
  /// On a device, the memory belongs to that device and the storage is
  /// freed when the last reference goes away.
  static std::shared_ptr<Storage> allocate(std::size_t bytes, Device device = Device::Cpu);

  /// A storage over memory that somebody else owns.
  ///
  /// `owner` keeps that memory alive for as long as the storage exists.
  /// A read-only storage refuses writes, so a tensor that views a
  /// memory-mapped file cannot corrupt it.
  static std::shared_ptr<Storage> wrap(const std::byte* data, std::size_t bytes, Device device,
                                       std::shared_ptr<void> owner);

  /// A storage over memory that somebody else owns, and that may be
  /// written to.
  static std::shared_ptr<Storage> wrapMutable(void* data, std::size_t bytes, Device device,
                                               std::shared_ptr<void> owner);

  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  void* data() { return external_ != nullptr ? const_cast<std::byte*>(external_) : owned_.get(); }
  const void* data() const { return external_ != nullptr ? external_ : owned_.get(); }

  std::size_t size() const { return size_; }
  Device device() const { return device_; }

  /// Is this storage read-only?
  bool readOnly() const { return read_only_; }

  /// A counter that changes whenever the contents change. Cached kernels
  /// use it to know when their plan is stale.
  std::uint64_t version() const { return version_; }
  void bumpVersion() { ++version_; }

 private:
  Storage(std::size_t bytes, Device device);

  /// Memory this storage owns.
  std::unique_ptr<std::byte[]> owned_;
  /// Memory this storage views but does not own.
  const std::byte* external_ = nullptr;
  std::size_t size_ = 0;
  Device device_ = Device::Cpu;
  std::uint64_t version_ = 1;
  /// What keeps `external_` alive.
  std::shared_ptr<void> owner_ = nullptr;
  /// The backend the external memory belongs to.
  Backend* backend_ = nullptr;
  bool read_only_ = false;
};

/// A tensor: a strided view over a block of memory.
///
/// Copying a `Tensor` copies the handle, not the data. To duplicate the
/// data, call `contiguous()` on a non-contiguous view, or copy explicitly.
class Tensor {
 public:
  Tensor() = default;

  /// A new tensor with uninitialised contents.
  static Tensor create(DType dtype, Shape shape, Device device = Device::Cpu);
  static Tensor zeros(DType dtype, Shape shape, Device device = Device::Cpu);
  static Tensor ones(DType dtype, Shape shape, Device device = Device::Cpu);
  static Tensor full(DType dtype, Shape shape, double value, Device device = Device::Cpu);

  /// A tensor filled from a list of numbers.
  static Tensor fromDoubles(Shape shape, const std::vector<double>& values,
                            DType dtype = DType::F32);

  /// A tensor over existing bytes. The bytes are copied.
  static Tensor fromBytes(Shape shape, const std::vector<std::byte>& bytes,
                           DType dtype, Device device = Device::Cpu);

  /// A view over an existing storage.
  static Tensor view(std::shared_ptr<Storage> storage, DType dtype, Shape shape, Strides strides,
                     std::int64_t elementOffset);

  bool valid() const { return storage_ != nullptr; }
  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  const Strides& strides() const { return strides_; }
  Device device() const { return storage_ != nullptr ? storage_->device() : Device::Cpu; }

  std::int64_t numel() const { return shape_.numel(); }
  std::size_t rank() const { return shape_.rank(); }

  /// Does the tensor store its elements contiguously, in row-major order?
  bool isContiguous() const;

  /// Does this tensor own its storage outright (as opposed to viewing
  /// another tensor's data)?
  bool ownsStorage() const { return owns_storage_; }

  const std::shared_ptr<Storage>& storage() const { return storage_; }

  /// The byte offset of element zero within the storage.
  std::int64_t byteOffset() const { return byte_offset_; }

  // --- element access ----------------------------------------------------

  /// Read one element. `indices` must have one entry per axis.
  double get(const std::vector<std::int64_t>& indices) const;

  /// Read one element by its position in row-major order.
  ///
  /// For a contiguous tensor this is a direct index. For a strided tensor
  /// the position is the row-major position of the element in the
  /// tensor's own shape, which is what `set` and `get` agree on.
  double getFlat(std::int64_t flat) const;

  /// Write one element.
  void set(const std::vector<std::int64_t>& indices, double value);

  /// Write one element by its position in row-major order.
  void setFlat(std::int64_t flat, double value);

  /// The element offset of a row-major position.
  std::int64_t elementOffsetForFlat(std::int64_t flat) const;

  /// The flat index of a multi-index, for a contiguous tensor.
  std::int64_t flatIndex(const std::vector<std::int64_t>& indices) const;

  // --- views -------------------------------------------------------------

  /// A view with a different shape. The element count must match.
  Tensor reshape(Shape shape) const;

  /// A view with the axes reversed.
  Tensor transpose() const;

  /// A view with the axes permuted.
  Tensor transpose(const std::vector<std::size_t>& axes) const;

  /// A strided view of one axis.
  Tensor slice(std::size_t axis, std::int64_t start, std::int64_t stop,
               std::int64_t step = 1) const;

  /// A tensor with the same values, stored contiguously. This copies when
  /// the tensor is not already contiguous.
  Tensor contiguous() const;

  /// A tensor with the same values in a different element type.
  Tensor to(DType dtype) const;

  /// A copy of this tensor on the host.
  Tensor hostCopy() const;

  /// A copy of this tensor on a device.
  Tensor toDevice(Device device) const;

  /// Copy another tensor's values into this one.
  void copyFrom(const Tensor& other);

  /// Replace this tensor's contents with another's.
  ///
  /// Unlike assignment, this keeps the gradient bookkeeping: a parameter
  /// that is updated in place keeps its gradient, which is what the
  /// optimiser needs.
  void assignValues(const Tensor& other);

  // --- gradients ---------------------------------------------------------

  /// Does this tensor take part in automatic differentiation?
  ///
  /// Only leaf tensors — the ones the user created — are ever marked. The
  /// tensors the engine produces from them are marked as it goes.
  bool requiresGrad() const { return requires_grad_; }
  void setRequiresGrad(bool value) { requires_grad_ = value; }

  /// The accumulated gradient of this tensor, if there is one.
  const std::shared_ptr<Tensor>& grad() const { return grad_; }

  /// Set the gradient outright.
  void setGrad(std::shared_ptr<Tensor> gradient) { grad_ = std::move(gradient); }

  /// Add to the gradient, creating it if this is the first contribution.
  ///
  /// The shapes must match; a gradient that has been broadcast back to
  /// this tensor's shape is added as it is.
  void accumulateGrad(const std::shared_ptr<Tensor>& gradient);

  /// Set the gradient to zero, keeping it allocated.
  void zeroGrad();

  // --- printing ----------------------------------------------------------

  /// "tensor<f32>[2, 3]"
  std::string toString() const;

  /// A readable dump of the values.
  std::string toDisplayString() const;

 private:
  std::shared_ptr<Storage> storage_;
  DType dtype_ = DType::F32;
  Shape shape_;
  Strides strides_;
  std::int64_t byte_offset_ = 0;
  bool owns_storage_ = false;
  /// Whether this tensor takes part in automatic differentiation.
  bool requires_grad_ = false;
  /// The accumulated gradient.
  std::shared_ptr<Tensor> grad_;
};

}  // namespace milo::tensor

#endif  // MILO_TENSOR_TENSOR_HPP
