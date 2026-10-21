#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cli/inspect.hpp"
#include "safetensors/format.hpp"
#include "safetensors/reader.hpp"
#include "safetensors/writer.hpp"
#include "tensor/tensor.hpp"
#include "test_framework.hpp"

using namespace milo;
using namespace milo::safetensors;

namespace {

/// A scratch file that removes itself.
class TempFile {
 public:
  explicit TempFile(const std::string& name) {
    path_ = (std::filesystem::temp_directory_path() / ("milo-test-" + name)).string();
  }
  ~TempFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  const std::string& path() const { return path_; }

  void write(const std::vector<std::byte>& bytes) const {
    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }

  std::vector<std::byte> read() const {
    std::ifstream file(path_, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes;
    bytes.reserve(raw.size());
    for (const char c : raw) {
      bytes.push_back(static_cast<std::byte>(c));
    }
    return bytes;
  }

 private:
  std::string path_;
};

/// Build a safetensors file in memory.
std::vector<std::byte> buildFile(const std::string& headerJson,
                                 const std::vector<std::byte>& data) {
  std::vector<std::byte> bytes;
  const auto length = static_cast<std::uint64_t>(headerJson.size());
  for (int i = 0; i < 8; ++i) {
    bytes.push_back(static_cast<std::byte>((length >> (8 * i)) & 0xFFu));
  }
  for (const char c : headerJson) {
    bytes.push_back(static_cast<std::byte>(c));
  }
  bytes.insert(bytes.end(), data.begin(), data.end());
  return bytes;
}

std::vector<std::byte> floatBytes(const std::vector<double>& values) {
  std::vector<std::byte> bytes;
  for (const double value : values) {
    const float narrowed = static_cast<float>(value);
    std::byte raw[4];
    std::memcpy(raw, &narrowed, 4);
    for (const std::byte b : raw) bytes.push_back(b);
  }
  return bytes;
}

}  // namespace

TEST(safetensorsParsesAValidHeader) {
  const std::string header = R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})";
  const Header parsed = parseHeader(header, 16);
  CHECK_EQ(parsed.tensors.size(), static_cast<std::size_t>(1));
  CHECK_EQ(parsed.tensors[0].name, std::string("a"));
  CHECK_EQ(parsed.tensors[0].dtype_name, std::string("F32"));
  CHECK_EQ(parsed.tensors[0].shape.size(), static_cast<std::size_t>(2));
  CHECK_EQ(parsed.tensors[0].shape[0], static_cast<std::int64_t>(2));
  CHECK_EQ(parsed.tensors[0].byteSize(), static_cast<std::uint64_t>(16));
  CHECK(parsed.tensors[0].supported);
}

TEST(safetensorsReadsMetadata) {
  const std::string header =
      R"({"__metadata__":{"format":"pt","note":"hello"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
  const Header parsed = parseHeader(header, 4);
  CHECK_EQ(parsed.metadata.size(), static_cast<std::size_t>(2));
  CHECK_EQ(parsed.metadata[0].first, std::string("format"));
  CHECK_EQ(parsed.metadata[0].second, std::string("pt"));
  CHECK_EQ(parsed.metadata[1].second, std::string("hello"));
}

TEST(safetensorsRejectsBadHeaders) {
  struct Case {
    const char* header;
    const char* because;
    std::uint64_t dataSize;
  };

  const Case cases[] = {
      {R"({"a":{"dtype":"NOPE","shape":[1],"data_offsets":[0,4]}})", "unknown dtype", 4},
      {R"({"a":{"shape":[1],"data_offsets":[0,4]}})", "no dtype", 4},
      {R"({"a":{"dtype":"F32","data_offsets":[0,4]}})", "no shape", 4},
      {R"({"a":{"dtype":"F32","shape":[1]}})", "no offsets", 4},
      {R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", "shape does not match", 4},
      {R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})", "offsets run backwards", 4},
      {R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}})", "offsets past the end", 4},
      {R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})", "negative shape", 4},
      {R"(not json)", "not json at all", 4},
      {R"({"a": 5})", "the entry is not an object", 4},
  };

  for (const Case& testCase : cases) {
    bool threw = false;
    try {
      parseHeader(testCase.header, testCase.dataSize);
    } catch (const Error&) {
      threw = true;
    }
    if (!threw) {
      ::milo::test::fail(__FILE__, __LINE__,
                         std::string("the parser accepted a header that is ") + testCase.because);
    }
  }
}

TEST(safetensorsRejectsOverlappingTensors) {
  const std::string header =
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
      R"("b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})";
  bool threw = false;
  try {
    parseHeader(header, 12);
  } catch (const Error&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(safetensorsRoundTripsThroughAFile) {
  TempFile file("round-trip.safetensors");

  tensor::Tensor first = tensor::Tensor::fromDoubles(
      tensor::Shape{std::vector<std::int64_t>{2, 3}}, {1, 2, 3, 4, 5, 6});
  tensor::Tensor second =
      tensor::Tensor::fromDoubles(tensor::Shape{std::vector<std::int64_t>{2}}, {7.5, -8.25});

  {
    Writer writer(file.path());
    writer.add("first", first);
    writer.add("second", second);
    writer.setMetadata("format", "milo");
    writer.write();
  }

  File opened = File::open(file.path());
  CHECK_EQ(opened.names().size(), static_cast<std::size_t>(2));
  CHECK(opened.has("first"));
  CHECK(!opened.has("missing"));
  CHECK_EQ(opened.header().metadata.size(), static_cast<std::size_t>(1));
  CHECK_EQ(opened.header().metadata[0].second, std::string("milo"));

  auto readFirst = opened.tensor("first");
  CHECK_EQ(readFirst->shape().toString(), std::string("[2, 3]"));
  CHECK_EQ(readFirst->dtype(), tensor::DType::F32);
  for (std::int64_t row = 0; row < 2; ++row) {
    for (std::int64_t column = 0; column < 3; ++column) {
      CHECK_EQ(readFirst->get({row, column}), first.get({row, column}));
    }
  }

  auto readSecond = opened.tensor("second");
  CHECK_EQ(readSecond->getFlat(0), 7.5);
  CHECK_EQ(readSecond->getFlat(1), -8.25);
}

TEST(safetensorsReadsWithoutCopying) {
  TempFile file("zero-copy.safetensors");

  tensor::Tensor tensor =
      tensor::Tensor::fromDoubles(tensor::Shape{std::vector<std::int64_t>{4}}, {1, 2, 3, 4});
  {
    Writer writer(file.path());
    writer.add("t", tensor);
    writer.write();
  }

  File opened = File::open(file.path());
  auto read = opened.tensor("t");
  // The storage views memory Milo does not own, so it is read-only.
  CHECK(read->storage()->readOnly());
  CHECK(!read->ownsStorage());

  bool threw = false;
  try {
    read->setFlat(0, 99.0);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(safetensorsRejectsFilesThatAreTooSmall) {
  TempFile file("tiny.safetensors");
  file.write({std::byte{0}, std::byte{1}});

  bool threw = false;
  try {
    File::open(file.path());
  } catch (const Error&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(safetensorsRejectsAHeaderLongerThanTheFile) {
  TempFile file("lying.safetensors");
  std::vector<std::byte> bytes(16, std::byte{0});
  // Claim a header of 1000 bytes in a 16-byte file.
  bytes[0] = static_cast<std::byte>(0xE8);
  bytes[1] = static_cast<std::byte>(0x03);
  file.write(bytes);

  bool threw = false;
  try {
    File::open(file.path());
  } catch (const Error&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(safetensorsSurvivesCorruptedFiles) {
  // Take a valid file, damage it in every way we can think of, and check
  // that reading it either works or fails cleanly. It must never read out
  // of bounds and it must never crash.
  TempFile file("fuzz.safetensors");
  tensor::Tensor tensor =
      tensor::Tensor::fromDoubles(tensor::Shape{std::vector<std::int64_t>{2, 2}}, {1, 2, 3, 4});
  {
    Writer writer(file.path());
    writer.add("a", tensor);
    writer.add("b", tensor);
    writer.write();
  }

  const std::vector<std::byte> original = file.read();
  CHECK(original.size() > 8);

  std::mt19937 engine(12345);
  int survived = 0;
  int rejected = 0;

  for (int attempt = 0; attempt < 400; ++attempt) {
    std::vector<std::byte> damaged = original;
    const int flips = 1 + static_cast<int>(engine() % 4);
    for (int i = 0; i < flips; ++i) {
      const std::size_t position = engine() % damaged.size();
      damaged[position] = static_cast<std::byte>(engine() & 0xFFu);
    }
    file.write(damaged);

    try {
      File opened = File::open(file.path());
      for (const std::string& name : opened.names()) {
        auto entry = opened.tensor(name);
        // Touch the data, so that a bad range would be noticed.
        if (entry->numel() > 0) {
          (void)entry->getFlat(0);
        }
      }
      ++survived;
    } catch (const Error&) {
      ++rejected;
    }
  }

  // Every attempt must end in one of the two buckets, never in a crash.
  CHECK_EQ(survived + rejected, 400);
  CHECK(rejected > 0);
}

TEST(safetensorsRoundTripsWithoutMetadata) {
  // A file with more than one tensor and no metadata: the header must
  // still be well formed.
  TempFile file("no-metadata.safetensors");

  tensor::Tensor first =
      tensor::Tensor::fromDoubles(tensor::Shape{std::vector<std::int64_t>{2}}, {1, 2});
  tensor::Tensor second =
      tensor::Tensor::fromDoubles(tensor::Shape{std::vector<std::int64_t>{3}}, {3, 4, 5});

  {
    Writer writer(file.path());
    writer.add("first", first);
    writer.add("second", second);
    writer.write();
  }

  File opened = File::open(file.path());
  CHECK_EQ(opened.names().size(), static_cast<std::size_t>(2));
  CHECK(opened.header().metadata.empty());
  CHECK_EQ(opened.tensor("first")->getFlat(1), 2.0);
  CHECK_EQ(opened.tensor("second")->getFlat(2), 5.0);
}

TEST(safetensorsBuildsAHeaderInOrder) {
  std::vector<TensorEntry> entries;
  TensorEntry first;
  first.name = "z";
  first.dtype_name = "F32";
  first.shape = {2};
  first.start = 0;
  first.end = 8;
  entries.push_back(first);

  TensorEntry second;
  second.name = "a";
  second.dtype_name = "F32";
  second.shape = {1};
  second.start = 8;
  second.end = 12;
  entries.push_back(second);

  const std::string header = buildHeader(entries, {});
  // The order of the entries follows the data, not the alphabet.
  CHECK(header.find("\"z\"") < header.find("\"a\""));
}

TEST(safetensorsInspectReportsOnAFile) {
  TempFile file("inspect.safetensors");
  tensor::Tensor tensor =
      tensor::Tensor::fromDoubles(tensor::Shape{std::vector<std::int64_t>{2, 2}}, {1, 2, 3, 4});
  {
    Writer writer(file.path());
    writer.add("weights", tensor);
    writer.setMetadata("format", "milo");
    writer.write();
  }

  std::ostringstream out;
  std::ostringstream err;
  const int status = cli::inspectFile(file.path(), out, err);
  CHECK_EQ(status, 0);

  const std::string report = out.str();
  CHECK_CONTAINS(report, "safetensors");
  CHECK_CONTAINS(report, "weights");
  CHECK_CONTAINS(report, "F32");
  CHECK_CONTAINS(report, "[2, 2]");
  CHECK_CONTAINS(report, "format: milo");
  CHECK(err.str().empty());
}

TEST(safetensorsInspectReportsAMissingFile) {
  std::ostringstream out;
  std::ostringstream err;
  const int status = cli::inspectFile("this-file-does-not-exist.safetensors", out, err);
  CHECK_EQ(status, 1);
  CHECK_CONTAINS(err.str(), "could not open");
}
