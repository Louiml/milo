// Reading safetensors files.
#ifndef MILO_SAFETENSORS_READER_HPP
#define MILO_SAFETENSORS_READER_HPP

#include <memory>
#include <string>
#include <vector>

#include "safetensors/format.hpp"
#include "safetensors/mapping.hpp"
#include "tensor/tensor.hpp"

namespace milo::safetensors {

/// A safetensors file, mapped into memory.
///
/// The file is memory mapped, so opening a 70 GB checkpoint costs nothing.
/// Tensors are returned as zero-copy views over the mapped bytes: nothing
/// is read until the data is actually used.
class File {
 public:
  /// Open and validate a file. Throws `Error` when it is not a valid
  /// safetensors file.
  static File open(const std::string& path);

  const Header& header() const { return header_; }

  /// The names of the tensors, in the order they appear in the file.
  const std::vector<std::string>& names() const { return names_; }

  /// Is there a tensor with this name?
  bool has(const std::string& name) const { return header_.find(name) != nullptr; }

  /// A zero-copy view of one tensor.
  ///
  /// The returned tensor shares the file's memory: reading it does not
  /// copy anything. Writing to it is not allowed.
  std::shared_ptr<tensor::Tensor> tensor(const std::string& name) const;

  /// The size of the file on disk, in bytes.
  std::size_t fileSize() const { return mapping_->size(); }

  /// The size of the data buffer, in bytes.
  std::size_t dataSize() const { return data_size_; }

 private:
  std::shared_ptr<MappedFile> mapping_;
  Header header_;
  std::shared_ptr<tensor::Storage> storage_;
  std::vector<std::string> names_;
  std::size_t data_size_ = 0;
};

}  // namespace milo::safetensors

#endif  // MILO_SAFETENSORS_READER_HPP
