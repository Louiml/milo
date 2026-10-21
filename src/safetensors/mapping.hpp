// A memory-mapped file.
//
// Mapping the file rather than reading it means a model's weights are
// never copied into memory: the operating system's page cache does the
// work, and only the layers that are actually touched are resident.
#ifndef MILO_SAFETENSORS_MAPPING_HPP
#define MILO_SAFETENSORS_MAPPING_HPP

#include <cstddef>
#include <string>

namespace milo::safetensors {

/// A read-only view of a file.
class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  /// Map a file. Throws `Error` when it cannot be opened.
  static MappedFile open(const std::string& path);

  const std::byte* data() const { return data_; }
  std::size_t size() const { return size_; }

  /// Is this a valid mapping?
  explicit operator bool() const { return data_ != nullptr; }

 private:
  void close();

  const std::byte* data_ = nullptr;
  std::size_t size_ = 0;

#if defined(_WIN32)
  void* file_ = nullptr;
  void* mapping_ = nullptr;
#else
  int descriptor_ = -1;
  void* address_ = nullptr;
  std::size_t length_ = 0;
#endif
};

}  // namespace milo::safetensors

#endif  // MILO_SAFETENSORS_MAPPING_HPP
