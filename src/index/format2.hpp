#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

// The index container of format 2, specified in docs/formats.md ("Index files,
// format 2"). This layer reads and writes the header, the section table and section
// bytes; each index defines its own sections and validates their contents.
namespace rabitqlib::index_detail::format2 {

// A file that is not a valid format-2 container, or uses something this release
// does not support.
class FormatError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

inline constexpr std::array<char, 8> kMagic = {'R', 'A', 'B', 'Q', 'I', 'N', 'D', 'X'};
inline constexpr uint32_t kVersion = 2;
inline constexpr uint32_t kHeaderBytes = 128;
inline constexpr uint64_t kSectionEntryBytes = 32;
inline constexpr uint64_t kSectionAlignment = 64;
inline constexpr size_t kWriterVersionBytes = 16;
// The rotator blob's limit, which bounds every index's working dimension.
inline constexpr uint64_t kMaxPaddedDim = 65536;

// Feature bits this release knows. Version 2 defines none.
inline constexpr uint64_t kKnownIncompatFeatures = 0;
inline constexpr uint64_t kKnownRoCompatFeatures = 0;
inline constexpr uint64_t kKnownCompatFeatures = 0;

enum class IndexKind : uint32_t { Ivf = 1, Hnsw = 2, SymphonyQg = 3 };
enum class Metric : uint32_t { L2 = 1, InnerProduct = 2, Cosine = 3 };
enum class RawType : uint32_t { None = 0, Float32 = 1, Uint8 = 2 };

inline constexpr uint32_t kSectionRequired = 1U << 0;
inline constexpr uint32_t kSectionChecksummed = 1U << 1;

// The section kind every index shares: its rotator blob.
inline constexpr uint32_t kRotatorSection = 1;

struct Header {
    uint64_t incompat_features = 0;
    uint64_t ro_compat_features = 0;
    uint64_t compat_features = 0;
    IndexKind index_kind = IndexKind::Ivf;
    Metric metric = Metric::L2;
    uint64_t dim = 0;
    uint64_t padded_dim = 0;
    uint32_t code_bits = 0;
    RawType raw_type = RawType::None;
    uint32_t id_bits = 32;
    std::optional<uint64_t> seed;
    uint64_t num_points = 0;
};

// A section table entry.
struct Section {
    uint32_t kind = 0;
    uint32_t flags = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t crc32 = 0;
};

class SectionSink;

// One section to write: `write` must pass exactly `size` bytes to the sink. Writers
// mark only small sections checksummed: those are buffered in memory before writing.
struct SectionWrite {
    uint32_t kind = 0;
    uint32_t flags = 0;
    uint64_t size = 0;
    std::function<void(SectionSink&)> write;
};

// Write a complete file from the current position of `output`, which must be the
// start of the file: the header, the section table right after it, then the sections
// in the given order, each at a multiple of 64. The output is written once, front to
// back, so it need not be seekable. Invalid arguments raise std::invalid_argument and
// write nothing; a section writer that passes the wrong number of bytes raises
// std::invalid_argument after part of the file is written.
void write_file(
    std::ostream& output,
    const Header& header,
    const std::vector<SectionWrite>& sections,
    std::string_view writer_version
);

// Receives one section's bytes during write_file; it rejects more bytes than declared.
class SectionSink {
   public:
    void write(const void* data, size_t size);
    [[nodiscard]] uint64_t remaining() const { return remaining_; }

   private:
    friend void write_file(
        std::ostream&, const Header&, const std::vector<SectionWrite>&, std::string_view
    );
    SectionSink(std::ostream& output, uint64_t size) : output_(output), remaining_(size) {}

    std::ostream& output_;
    uint64_t remaining_;
};

// An output stream that passes its bytes to a section sink, for section writers that
// take a std::ostream. A sink error propagates as an exception.
class SinkStream : public std::ostream {
   public:
    explicit SinkStream(SectionSink& sink) : std::ostream(nullptr), buffer_(sink) {
        rdbuf(&buffer_);
        exceptions(std::ios::badbit);
    }

   private:
    class Buffer : public std::streambuf {
       public:
        explicit Buffer(SectionSink& sink) : sink_(sink) {}

       private:
        int_type overflow(int_type c) override {
            if (!traits_type::eq_int_type(c, traits_type::eof())) {
                const char value = traits_type::to_char_type(c);
                sink_.write(&value, 1);
            }
            return traits_type::not_eof(c);
        }
        std::streamsize xsputn(const char* data, std::streamsize count) override {
            sink_.write(data, static_cast<size_t>(count));
            return count;
        }

        SectionSink& sink_;
    };

    Buffer buffer_;
};

// An output stream that only counts its bytes, to size a section before writing it.
class CountingStream : public std::ostream {
   public:
    CountingStream() : std::ostream(nullptr) { rdbuf(&buffer_); }
    [[nodiscard]] uint64_t count() const { return buffer_.count; }

   private:
    class Buffer : public std::streambuf {
       public:
        uint64_t count = 0;

       private:
        int_type overflow(int_type c) override {
            count += traits_type::eq_int_type(c, traits_type::eof()) ? 0 : 1;
            return traits_type::not_eof(c);
        }
        std::streamsize xsputn(const char* /*data*/, std::streamsize n) override {
            count += static_cast<uint64_t>(n);
            return n;
        }
    };

    Buffer buffer_;
};

// Whether the input starts with the format-2 magic. It reads through the stream
// buffer, so the stream's state and exception mask do not matter, and restores the
// position. Input that cannot seek raises FormatError, named for `kind`.
bool has_magic(std::istream& input, IndexKind kind);

// Reads one section front to back in pieces, seeking only once. Other reads of the
// same input move its position, so finish one section before reading another.
class SectionInput {
   public:
    // Read the next `size` bytes of the section.
    void read(void* out, size_t size);

   private:
    friend class Reader;
    SectionInput(std::streambuf& input, const std::string& name, uint64_t size)
        : input_(input), name_(name), remaining_(size) {}

    std::streambuf& input_;
    std::string name_;
    uint64_t remaining_;
};

class Reader {
   public:
    // Read and validate the header and the section table, in the order of "Reading"
    // in the specification: the generic header rules, then `expected_kind`, then
    // `check_header` (the index's own header rules, which throw FormatError), then
    // the sections. Sections whose kind is not in `known_kinds` are rejected when
    // required and skipped otherwise. The reader reads through the stream buffer of
    // `input`, which must outlive it, and ignores the stream's state.
    Reader(
        std::istream& input,
        IndexKind expected_kind,
        const std::vector<uint32_t>& known_kinds,
        const std::function<void(const Header&)>& check_header = {}
    );

    [[nodiscard]] const Header& header() const { return header_; }
    // True when the file uses read-only-compatible features this release does not know:
    // it may be searched, but add, remove and save must refuse it.
    [[nodiscard]] bool read_only() const { return read_only_; }
    [[nodiscard]] const std::string& writer_version() const { return writer_version_; }
    // The entry of a known section kind, or nullptr when the file has none.
    [[nodiscard]] const Section* find(uint32_t kind) const;
    // Read `size` bytes starting `offset` bytes into `section`.
    void read(const Section& section, uint64_t offset, void* out, size_t size);
    // Read `section` from its start in consecutive pieces; for large sections.
    SectionInput open(const Section& section);

   private:
    void read_at(uint64_t offset, void* out, size_t size, const std::string& what);

    std::streambuf& input_;
    std::string component_;
    Header header_;
    bool read_only_ = false;
    std::string writer_version_;
    std::vector<Section> sections_;
};

}  // namespace rabitqlib::index_detail::format2
