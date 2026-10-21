// Writing safetensors files.
#ifndef MILO_SAFETENSORS_WRITER_HPP
#define MILO_SAFETENSORS_WRITER_HPP

#include <memory>
#include <string>
#include <vector>

#include "safetensors/format.hpp"
#include "tensor/tensor.hpp"

namespace milo::safetensors {

/// Writes a safetensors file.
///
/// The tensors are added first and written in one go, so that the header
/// can describe the whole file. The data is streamed out in pieces rather
/// than buffered, so writing a large checkpoint does not need a large
/// amount of memory.
class Writer {
 public:
  explicit Writer(std::string path);
  ~Writer();

  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;

  /// Add a tensor. The data is not copied until `write()` is called.
  void add(const std::string& name, const tensor::Tensor& tensor);

  /// Add a metadata entry. Metadata is a map of strings, and the format
  /// allows any number of them.
  void setMetadata(const std::string& key, const std::string& value);

  /// Write the file.
  void write();

  /// How many tensors have been added.
  std::size_t tensorCount() const { return tensors_.size(); }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
  std::vector<TensorEntry> entries_;
  std::vector<std::shared_ptr<tensor::Tensor>> tensors_;
  std::vector<std::pair<std::string, std::string>> metadata_;
  bool written_ = false;
};

}  // namespace milo::safetensors

#endif  // MILO_SAFETENSORS_WRITER_HPP
