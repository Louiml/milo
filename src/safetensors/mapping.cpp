#include "safetensors/mapping.hpp"

#include <cstring>
#include <utility>

#include "safetensors/format.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace milo::safetensors {

#if defined(_WIN32)

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(other.data_), size_(other.size_), file_(other.file_), mapping_(other.mapping_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.file_ = nullptr;
  other.mapping_ = nullptr;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    close();
    data_ = other.data_;
    size_ = other.size_;
    file_ = other.file_;
    mapping_ = other.mapping_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.file_ = nullptr;
    other.mapping_ = nullptr;
  }
  return *this;
}

void MappedFile::close() {
  if (data_ != nullptr) {
    ::UnmapViewOfFile(data_);
    data_ = nullptr;
  }
  if (mapping_ != nullptr) {
    ::CloseHandle(mapping_);
    mapping_ = nullptr;
  }
  if (file_ != nullptr) {
    ::CloseHandle(file_);
    file_ = nullptr;
  }
  size_ = 0;
}

MappedFile MappedFile::open(const std::string& path) {
  MappedFile mapped;

  HANDLE file = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    throw Error("I could not open '" + path + "'");
  }

  LARGE_INTEGER size = {};
  if (!::GetFileSizeEx(file, &size)) {
    ::CloseHandle(file);
    throw Error("I could not find out how big '" + path + "' is");
  }
  if (size.QuadPart == 0) {
    ::CloseHandle(file);
    throw Error("'" + path + "' is empty");
  }

  HANDLE mapping = ::CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    ::CloseHandle(file);
    throw Error("I could not map '" + path + "' into memory");
  }

  void* address = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (address == nullptr) {
    ::CloseHandle(mapping);
    ::CloseHandle(file);
    throw Error("I could not map '" + path + "' into memory");
  }

  mapped.data_ = static_cast<const std::byte*>(address);
  mapped.size_ = static_cast<std::size_t>(size.QuadPart);
  mapped.file_ = file;
  mapped.mapping_ = mapping;
  return mapped;
}

#else

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
      descriptor_(other.descriptor_),
      address_(other.address_),
      length_(other.length_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.descriptor_ = -1;
  other.address_ = nullptr;
  other.length_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    close();
    data_ = other.data_;
    size_ = other.size_;
    descriptor_ = other.descriptor_;
    address_ = other.address_;
    length_ = other.length_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.descriptor_ = -1;
    other.address_ = nullptr;
    other.length_ = 0;
  }
  return *this;
}

void MappedFile::close() {
  if (address_ != nullptr) {
    ::munmap(address_, length_);
    address_ = nullptr;
  }
  if (descriptor_ >= 0) {
    ::close(descriptor_);
    descriptor_ = -1;
  }
  data_ = nullptr;
  size_ = 0;
  length_ = 0;
}

MappedFile MappedFile::open(const std::string& path) {
  MappedFile mapped;

  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    throw Error("I could not open '" + path + "'");
  }

  struct stat info = {};
  if (::fstat(descriptor, &info) != 0) {
    ::close(descriptor);
    throw Error("I could not find out how big '" + path + "' is");
  }
  if (info.st_size == 0) {
    ::close(descriptor);
    throw Error("'" + path + "' is empty");
  }

  const auto length = static_cast<std::size_t>(info.st_size);
  void* address = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, descriptor, 0);
  if (address == MAP_FAILED) {
    ::close(descriptor);
    throw Error("I could not map '" + path + "' into memory");
  }

  mapped.data_ = static_cast<const std::byte*>(address);
  mapped.size_ = length;
  mapped.descriptor_ = descriptor;
  mapped.address_ = address;
  mapped.length_ = length;
  return mapped;
}

#endif

}  // namespace milo::safetensors
