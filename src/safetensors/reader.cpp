#include "safetensors/reader.hpp"

#include <cstring>
#include <utility>

namespace milo::safetensors {

File File::open(const std::string& path) {
  File file;
  file.mapping_ = std::make_shared<MappedFile>(MappedFile::open(path));

  const std::byte* base = file.mapping_->data();
  const std::size_t size = file.mapping_->size();

  if (size < 8) {
    throw Error("'" + path + "' is too small to be a safetensors file");
  }

  // The first eight bytes are the length of the header.
  std::uint64_t headerSize = 0;
  for (int i = 7; i >= 0; --i) {
    headerSize = (headerSize << 8) | static_cast<std::uint64_t>(base[i]);
  }

  if (headerSize > kMaxHeaderSize) {
    throw Error("'" + path + "' has a header that is too large");
  }
  if (headerSize > size - 8) {
    throw Error("'" + path + "' is shorter than its header says it is");
  }

  const std::string headerJson(reinterpret_cast<const char*>(base + 8),
                               static_cast<std::size_t>(headerSize));

  file.data_size_ = size - 8 - static_cast<std::size_t>(headerSize);
  file.header_ = parseHeader(headerJson, file.data_size_);

  for (const TensorEntry& entry : file.header_.tensors) {
    file.names_.push_back(entry.name);
  }

  // One storage for the whole data buffer; every tensor is a view into it.
  // The mapping is kept alive by the storage, so a tensor may outlive the
  // `File` object it came from.
  file.storage_ = tensor::Storage::wrap(base + 8 + headerSize, file.data_size_,
                                        tensor::Device::Cpu, file.mapping_);

  return file;
}

std::shared_ptr<tensor::Tensor> File::tensor(const std::string& name) const {
  const TensorEntry* entry = header_.find(name);
  if (entry == nullptr) {
    throw Error("this file has no tensor called '" + name + "'");
  }
  if (!entry->supported) {
    throw Error("the tensor '" + name + "' has the dtype " + entry->dtype_name +
                ", which Milo does not support yet");
  }

  const std::size_t elementSize = tensor::dtypeSize(entry->dtype);
  if (entry->byteSize() % elementSize != 0) {
    throw Error("the tensor '" + name + "' does not hold a whole number of elements");
  }

  // The data is laid out contiguously, so the tensor is a plain view.
  return std::make_shared<tensor::Tensor>(tensor::Tensor::view(
      storage_, entry->dtype, entry->toShape(),
      tensor::Strides::contiguous(entry->toShape()),
      static_cast<std::int64_t>(entry->start / elementSize)));
}

}  // namespace milo::safetensors
