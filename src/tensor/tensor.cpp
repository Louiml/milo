#include "tensor/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "tensor/convert.hpp"
#include "tensor/ops.hpp"

namespace milo::tensor {

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

Storage::Storage(std::size_t bytes, Device device)
    : owned_(bytes == 0 ? nullptr : std::make_unique<std::byte[]>(bytes)),
      size_(bytes),
      device_(device) {
  if (bytes != 0) {
    std::memset(owned_.get(), 0, bytes);
  }
}

Storage::~Storage() {
  if (external_ != nullptr) {
    // The memory belongs to a device: give it back.
    if (backend_ != nullptr) backend_->deallocate(const_cast<std::byte*>(external_));
    external_ = nullptr;
  }
}

std::shared_ptr<Storage> Storage::allocate(std::size_t bytes, Device device) {
  if (device == Device::Cpu) {
    // The constructor is private, so make_shared cannot reach it.
    return std::shared_ptr<Storage>(new Storage(bytes, device));
  }

  Backend* backend = backendFor(device);
  if (backend == nullptr) {
    throw std::runtime_error("there is no backend for this device");
  }
  void* data = backend->allocate(bytes);
  if (data == nullptr) {
    throw std::runtime_error("the device could not give us the memory we asked for");
  }

  auto storage = std::shared_ptr<Storage>(new Storage(0, device));
  storage->external_ = static_cast<const std::byte*>(data);
  storage->size_ = bytes;
  storage->backend_ = backend;
  return storage;
}

std::shared_ptr<Storage> Storage::wrap(const std::byte* data, std::size_t bytes, Device device,
                                       std::shared_ptr<void> owner) {
  // The constructor is private, so make_shared cannot reach it.
  auto storage = std::shared_ptr<Storage>(new Storage(0, device));
  storage->size_ = bytes;
  storage->owner_ = std::move(owner);
  storage->read_only_ = true;
  storage->external_ = data;
  return storage;
}

std::shared_ptr<Storage> Storage::wrapMutable(void* data, std::size_t bytes, Device device,
                                               std::shared_ptr<void> owner) {
  auto storage = std::shared_ptr<Storage>(new Storage(0, device));
  storage->size_ = bytes;
  storage->owner_ = std::move(owner);
  storage->read_only_ = false;
  storage->external_ = static_cast<const std::byte*>(data);
  return storage;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Tensor Tensor::create(DType dtype, Shape shape, Device device) {
  const std::int64_t elements = shape.numel();
  if (elements < 0) {
    throw std::invalid_argument("a tensor's shape cannot have a negative size");
  }
  const auto bytes = static_cast<std::size_t>(elements) * dtypeSize(dtype);

  Tensor tensor;
  tensor.storage_ = Storage::allocate(bytes, device);
  tensor.dtype_ = dtype;
  tensor.shape_ = std::move(shape);
  tensor.strides_ = Strides::contiguous(tensor.shape_);
  tensor.byte_offset_ = 0;
  tensor.owns_storage_ = true;
  return tensor;
}

Tensor Tensor::zeros(DType dtype, Shape shape, Device device) {
  return create(dtype, std::move(shape), device);
}

Tensor Tensor::ones(DType dtype, Shape shape, Device device) {
  Tensor tensor = create(dtype, std::move(shape), device);
  for (std::int64_t i = 0; i < tensor.numel(); ++i) {
    tensor.setFlat(i, 1.0);
  }
  return tensor;
}

Tensor Tensor::full(DType dtype, Shape shape, double value, Device device) {
  if (device != Device::Cpu) {
    // Fill on the host, then move it to the device.
    Tensor host = create(dtype, std::move(shape));
    for (std::int64_t i = 0; i < host.numel(); ++i) {
      host.setFlat(i, value);
    }
    return host.toDevice(device);
  }

  Tensor tensor = create(dtype, std::move(shape));
  for (std::int64_t i = 0; i < tensor.numel(); ++i) {
    tensor.setFlat(i, value);
  }
  return tensor;
}

Tensor Tensor::hostCopy() const {
  if (storage_->device() == Device::Cpu) {
    Tensor copy = create(dtype_, shape_);
    copy.copyFrom(*this);
    return copy;
  }

  // Bring the data back from the device.
  Backend* backend = backendFor(storage_->device());
  Tensor copy = create(dtype_, shape_);
  if (backend != nullptr) {
    backend->copyToHost(copy.storage_->data(),
                        static_cast<const std::byte*>(storage_->data()) + byte_offset_,
                        static_cast<std::size_t>(numel()) * dtypeSize(dtype_));
  }
  return copy;
}

Tensor Tensor::toDevice(Device device) const {
  if (storage_->device() == device) {
    return *this;
  }
  if (device == Device::Cpu) {
    return hostCopy();
  }

  Backend* backend = backendFor(device);
  if (backend == nullptr) {
    throw std::runtime_error("there is no backend for this device");
  }

  const Tensor source = hostCopy();
  Tensor copy = create(dtype_, shape_, device);
  backend->copyToDevice(copy.storage_->data(),
                        static_cast<const std::byte*>(source.storage_->data()) + source.byteOffset(),
                        static_cast<std::size_t>(numel()) * dtypeSize(dtype_));
  return copy;
}

void Tensor::copyFrom(const Tensor& other) {
  if (other.numel() != numel()) {
    throw std::invalid_argument("these tensors are not the same size");
  }
  if (storage_->device() != other.storage()->device()) {
    throw std::invalid_argument("these tensors are not on the same device");
  }

  const std::size_t bytes = static_cast<std::size_t>(numel()) * dtypeSize(dtype_);
  if (storage_->device() == Device::Cpu) {
    std::memcpy(static_cast<std::byte*>(storage_->data()) + byte_offset_,
                static_cast<const std::byte*>(other.storage()->data()) + other.byteOffset(), bytes);
  } else {
    Backend* backend = backendFor(storage_->device());
    if (backend == nullptr) {
      throw std::runtime_error("there is no backend for this device");
    }
    backend->copyBetween(static_cast<std::byte*>(storage_->data()) + byteOffset(),
                         static_cast<const std::byte*>(other.storage()->data()) + other.byteOffset(),
                         bytes);
  }
  storage_->bumpVersion();
}
Tensor Tensor::fromDoubles(Shape shape, const std::vector<double>& values, DType dtype) {
  Tensor tensor = create(dtype, std::move(shape));
  const std::int64_t count = std::min<std::int64_t>(tensor.numel(),
                                                   static_cast<std::int64_t>(values.size()));
  for (std::int64_t i = 0; i < count; ++i) {
    tensor.setFlat(i, values[static_cast<std::size_t>(i)]);
  }
  return tensor;
}

Tensor Tensor::fromBytes(Shape shape, const std::vector<std::byte>& bytes, DType dtype,
                         Device device) {
  Tensor tensor = create(dtype, std::move(shape), device);
  const std::size_t wanted = static_cast<std::size_t>(tensor.numel()) * dtypeSize(dtype);
  const std::size_t available = std::min(wanted, bytes.size());
  if (available != 0) {
    std::memcpy(tensor.storage_->data(), bytes.data(), available);
  }
  return tensor;
}

Tensor Tensor::view(std::shared_ptr<Storage> storage, DType dtype, Shape shape, Strides strides,
                    std::int64_t elementOffset) {
  Tensor tensor;
  tensor.storage_ = std::move(storage);
  tensor.dtype_ = dtype;
  tensor.shape_ = std::move(shape);
  tensor.strides_ = std::move(strides);
  tensor.byte_offset_ = elementOffset * static_cast<std::int64_t>(dtypeSize(dtype));
  tensor.owns_storage_ = false;
  return tensor;
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

bool Tensor::isContiguous() const {
  if (shape_.isScalar()) return true;
  std::int64_t expected = 1;
  for (std::size_t axis = shape_.rank(); axis-- > 0;) {
    if (strides_[axis] != expected) return false;
    expected *= shape_.dim(axis);
  }
  return true;
}

std::int64_t Tensor::flatIndex(const std::vector<std::int64_t>& indices) const {
  if (indices.size() != shape_.rank()) {
    throw std::invalid_argument("the index has the wrong number of dimensions");
  }
  std::int64_t flat = 0;
  for (std::size_t axis = 0; axis < shape_.rank(); ++axis) {
    flat = flat * shape_.dim(axis) + indices[axis];
  }
  return flat;
}

std::int64_t Tensor::elementOffsetForFlat(std::int64_t flat) const {
  if (isContiguous()) return flat;

  // The position is in row-major order over the tensor's own shape, so
  // turn it into a multi-index and then into an element offset.
  std::int64_t remaining = flat;
  std::int64_t offset = 0;
  for (std::size_t axis = shape_.rank(); axis-- > 0;) {
    const std::int64_t size = shape_.dim(axis);
    const std::int64_t index = remaining % size;
    remaining /= size;
    offset += index * strides_[axis];
  }
  return offset;
}

double Tensor::getFlat(std::int64_t flat) const {
  if (storage_->device() != Device::Cpu) {
    throw std::runtime_error("this tensor is on a device; copy it to the host to read it");
  }
  return loadElement(static_cast<const std::byte*>(storage_->data()) + byte_offset_, dtype_,
                     elementOffsetForFlat(flat));
}

void Tensor::setFlat(std::int64_t flat, double value) {
  if (storage_->readOnly()) {
    throw std::runtime_error("this tensor is read-only");
  }
  if (storage_->device() != Device::Cpu) {
    throw std::runtime_error("this tensor is on a device; copy it to the host to change it");
  }
  storeElement(static_cast<std::byte*>(storage_->data()) + byte_offset_, dtype_,
               elementOffsetForFlat(flat), value);
  storage_->bumpVersion();
}

double Tensor::get(const std::vector<std::int64_t>& indices) const {
  if (storage_->device() != Device::Cpu) {
    throw std::runtime_error("this tensor is on a device; copy it to the host to read it");
  }
  if (indices.size() != shape_.rank()) {
    throw std::invalid_argument("the index has the wrong number of dimensions");
  }
  std::int64_t offset = 0;
  for (std::size_t axis = 0; axis < shape_.rank(); ++axis) {
    offset += indices[axis] * strides_[axis];
  }
  return loadElement(static_cast<const std::byte*>(storage_->data()) + byte_offset_, dtype_,
                     offset);
}

void Tensor::set(const std::vector<std::int64_t>& indices, double value) {
  if (storage_->readOnly()) {
    throw std::runtime_error("this tensor is read-only");
  }
  if (storage_->device() != Device::Cpu) {
    throw std::runtime_error("this tensor is on a device; copy it to the host to change it");
  }
  if (indices.size() != shape_.rank()) {
    throw std::invalid_argument("the index has the wrong number of dimensions");
  }
  std::int64_t offset = 0;
  for (std::size_t axis = 0; axis < shape_.rank(); ++axis) {
    offset += indices[axis] * strides_[axis];
  }
  storeElement(static_cast<std::byte*>(storage_->data()) + byte_offset_, dtype_, offset, value);
  storage_->bumpVersion();
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

Tensor Tensor::reshape(Shape shape) const {
  if (shape.numel() != numel()) {
    throw std::invalid_argument("a reshape must keep the same number of elements");
  }
  Tensor result = *this;
  result.shape_ = std::move(shape);
  if (isContiguous()) {
    result.strides_ = Strides::contiguous(result.shape_);
  }
  // A reshaped tensor views the same data; it does not own it.
  result.owns_storage_ = false;
  // A raw view is not part of the gradient graph; the autograd layer
  // records reshape and transpose as their own operations.
  result.requires_grad_ = false;
  result.grad_.reset();
  return result;
}

Tensor Tensor::transpose() const {
  std::vector<std::size_t> axes(shape_.rank());
  for (std::size_t i = 0; i < axes.size(); ++i) {
    axes[i] = shape_.rank() - 1 - i;
  }
  return transpose(axes);
}

Tensor Tensor::transpose(const std::vector<std::size_t>& axes) const {
  if (axes.size() != shape_.rank()) {
    throw std::invalid_argument("a permutation must name every axis");
  }
  std::vector<std::size_t> seen(shape_.rank(), 0);
  for (const std::size_t axis : axes) {
    if (axis >= shape_.rank()) {
      throw std::invalid_argument("a permutation names an axis that does not exist");
    }
    if (seen[axis] != 0) {
      throw std::invalid_argument("a permutation may not repeat an axis");
    }
    seen[axis] = 1;
  }

  Shape newShape = shape_.permuted(axes);
  std::vector<std::int64_t> newStrides(shape_.rank());
  for (std::size_t i = 0; i < axes.size(); ++i) {
    newStrides[i] = strides_[axes[i]];
  }

  Tensor result = *this;
  result.shape_ = Shape(std::move(newShape));
  result.strides_ = Strides(std::move(newStrides));
  result.owns_storage_ = false;
  // A raw view is not part of the gradient graph; the autograd layer
  // records reshape and transpose as their own operations.
  result.requires_grad_ = false;
  result.grad_.reset();
  return result;
}

Tensor Tensor::slice(std::size_t axis, std::int64_t start, std::int64_t stop,
                     std::int64_t step) const {
  if (axis >= shape_.rank()) {
    throw std::invalid_argument("this axis does not exist");
  }
  if (step <= 0) {
    throw std::invalid_argument("a slice step must be positive");
  }
  const std::int64_t size = shape_.dim(axis);
  if (start < 0) start += size;
  if (stop < 0) stop += size;
  start = std::clamp<std::int64_t>(start, 0, size);
  stop = std::clamp<std::int64_t>(stop, start, size);

  const std::int64_t count = (stop - start + step - 1) / step;

  std::vector<std::int64_t> newShape = shape_.dims();
  newShape[axis] = count;
  std::vector<std::int64_t> newStrides = strides_.values();
  newStrides[axis] = strides_[axis] * step;

  Tensor result = *this;
  result.shape_ = Shape(std::move(newShape));
  result.strides_ = Strides(std::move(newStrides));
  result.byte_offset_ += start * strides_[axis] * static_cast<std::int64_t>(dtypeSize(dtype_));
  result.owns_storage_ = false;
  // A raw view is not part of the gradient graph; the autograd layer
  // records reshape and transpose as their own operations.
  result.requires_grad_ = false;
  result.grad_.reset();
  return result;
}

Tensor Tensor::contiguous() const {
  if (isContiguous()) return *this;

  Tensor result = create(dtype_, shape_);
  // Walk the source in row-major order and write it out contiguously.
  std::vector<std::int64_t> indices(shape_.rank(), 0);
  const std::int64_t total = numel();
  for (std::int64_t i = 0; i < total; ++i) {
    result.setFlat(i, get(indices));
    // Advance the index.
    for (std::size_t axis = shape_.rank(); axis-- > 0;) {
      if (++indices[axis] < shape_.dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return result;
}

Tensor Tensor::to(DType dtype) const {
  if (dtype == dtype_) return *this;

  Tensor result = create(dtype, shape_);
  std::vector<std::int64_t> indices(shape_.rank(), 0);
  const std::int64_t total = numel();
  for (std::int64_t i = 0; i < total; ++i) {
    result.setFlat(i, get(indices));
    for (std::size_t axis = shape_.rank(); axis-- > 0;) {
      if (++indices[axis] < shape_.dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return result;
}

void Tensor::assignValues(const Tensor& other) {
  dtype_ = other.dtype_;
  shape_ = other.shape_;
  strides_ = other.strides_;
  storage_ = other.storage_;
  byte_offset_ = other.byte_offset_;
  owns_storage_ = other.owns_storage_;
  // `requires_grad_` and `grad_` are deliberately left alone: this is an
  // update of the values, not a new tensor.
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

void Tensor::accumulateGrad(const std::shared_ptr<Tensor>& gradient) {
  if (!gradient) return;
  if (gradient->shape() != shape_) {
    throw std::invalid_argument("a gradient must have the same shape as the tensor");
  }
  if (!grad_) {
    grad_ = std::make_shared<Tensor>(Tensor::zeros(dtype_, shape_));
  }
  // Gradients accumulate, so this is an add.
  *grad_ = binaryOp(BinaryOp::Add, *grad_, *gradient);
}

void Tensor::zeroGrad() {
  if (!grad_) return;
  *grad_ = Tensor::zeros(dtype_, shape_);
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

std::string Tensor::toString() const {
  std::ostringstream out;
  out << "tensor<" << dtypeName(dtype_) << ">" << shape_.toString();
  return out.str();
}

std::string Tensor::toDisplayString() const {
  std::ostringstream out;
  if (shape_.isScalar()) {
    // A scalar tensor prints as its value, so that `sum(x)` reads like a
    // number.
    if (numel() == 0) return "0";
    out << getFlat(0);
    return out.str();
  }

  out << toString();
  if (numel() == 0) return out.str();

  out << "\n";
  const std::size_t limit = numel() <= 64 ? static_cast<std::size_t>(numel()) : 64;

  // A one-dimensional tensor prints on one line; anything else prints one
  // row per slice of the first axis.
  if (shape_.rank() <= 1) {
    out << "[";
    for (std::size_t i = 0; i < limit; ++i) {
      if (i != 0) out << ", ";
      out << getFlat(static_cast<std::int64_t>(i));
    }
    if (limit < static_cast<std::size_t>(numel())) out << ", ...";
    out << "]";
    return out.str();
  }

  const std::int64_t rows = shape_.dim(0);
  const std::int64_t perRow = numel() / std::max<std::int64_t>(rows, 1);
  std::vector<std::int64_t> indices(shape_.rank(), 0);
  for (std::int64_t row = 0; row < rows; ++row) {
    out << "[";
    for (std::int64_t column = 0; column < perRow; ++column) {
      if (column != 0) out << ", ";
      out << get(indices);
      // Move along the row. The carry into the next row is handled below.
      if (column + 1 < perRow) {
        ++indices[shape_.rank() - 1];
      }
    }
    out << "]";
    if (row + 1 < rows) out << "\n";
    // Move to the next row, carrying across the axes.
    for (std::size_t axis = shape_.rank(); axis-- > 0;) {
      if (++indices[axis] < shape_.dim(axis)) break;
      indices[axis] = 0;
    }
  }
  return out.str();
}

}  // namespace milo::tensor
