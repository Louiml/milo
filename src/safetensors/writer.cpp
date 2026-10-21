#include "safetensors/writer.hpp"

#include <cstring>
#include <fstream>
#include <utility>

namespace milo::safetensors {
namespace {

/// How much is written at a time.
constexpr std::size_t kChunkBytes = 1u << 20;  // 1 MiB

}  // namespace

Writer::Writer(std::string path) : path_(std::move(path)) {}

Writer::~Writer() = default;

void Writer::add(const std::string& name, const tensor::Tensor& tensor) {
  if (written_) {
    throw Error("this file has already been written");
  }
  if (!tensor.valid()) {
    throw Error("the tensor '" + name + "' is not valid");
  }

  TensorEntry entry;
  entry.name = name;
  entry.dtype_name = dtypeFileName(tensor.dtype());
  entry.dtype = tensor.dtype();
  entry.supported = true;
  entry.shape = tensor.shape().dims();

  entries_.push_back(std::move(entry));
  tensors_.push_back(std::make_shared<tensor::Tensor>(tensor));
}

void Writer::setMetadata(const std::string& key, const std::string& value) {
  if (written_) {
    throw Error("this file has already been written");
  }
  metadata_.emplace_back(key, value);
}

void Writer::write() {
  if (written_) {
    throw Error("this file has already been written");
  }
  written_ = true;

  // Lay the data out one tensor after another.
  std::uint64_t offset = 0;
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    const tensor::Tensor& tensor = *tensors_[i];
    const std::uint64_t bytes = static_cast<std::uint64_t>(tensor.numel()) *
                                 tensor::dtypeSize(tensor.dtype());
    entries_[i].start = offset;
    entries_[i].end = offset + bytes;
    offset += bytes;
  }

  const std::string header = buildHeader(entries_, metadata_);

  std::ofstream file(path_, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw Error("I could not open '" + path_ + "' for writing");
  }

  // The header length, as a little-endian u64.
  const auto headerSize = static_cast<std::uint64_t>(header.size());
  std::byte lengthBytes[8];
  for (int i = 0; i < 8; ++i) {
    lengthBytes[i] = static_cast<std::byte>((headerSize >> (8 * i)) & 0xFFu);
  }
  file.write(reinterpret_cast<const char*>(lengthBytes), 8);
  file.write(header.data(), static_cast<std::streamsize>(header.size()));

  // The data, one tensor at a time.
  std::vector<char> buffer(kChunkBytes);
  for (const auto& tensor : tensors_) {
    // A tensor that is not contiguous has to be laid out first.
    const tensor::Tensor source = tensor->isContiguous() ? *tensor : tensor->contiguous();
    const auto* bytes = static_cast<const std::byte*>(source.storage()->data()) +
                        source.byteOffset();
    const std::size_t total = static_cast<std::size_t>(source.numel()) *
                              tensor::dtypeSize(source.dtype());

    std::size_t written = 0;
    while (written < total) {
      const std::size_t count = std::min(kChunkBytes, total - written);
      std::memcpy(buffer.data(), bytes + written, count);
      file.write(buffer.data(), static_cast<std::streamsize>(count));
      written += count;
    }
  }

  file.flush();
  if (!file) {
    throw Error("I could not write to '" + path_ + "'");
  }
}

}  // namespace milo::safetensors
