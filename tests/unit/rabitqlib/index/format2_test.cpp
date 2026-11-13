#include "format2.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

#include "crc32.hpp"

namespace {
namespace f2 = rabitqlib::index_detail::format2;
using rabitqlib::detail::crc32;

constexpr uint32_t kParams = 256;
constexpr uint32_t kBulk = 261;
constexpr uint32_t kEmpty = 263;
const std::vector<uint32_t> kKnown = {kParams, kBulk, kEmpty};

uint32_t get_u32(const std::string& bytes, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + i])) << (8 * i);
    }
    return value;
}

uint64_t get_u64(const std::string& bytes, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(static_cast<uint8_t>(bytes[offset + i])) << (8 * i);
    }
    return value;
}

void put_u32(std::string& bytes, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        bytes[offset + i] = static_cast<char>(value >> (8 * i));
    }
}

void put_u64(std::string& bytes, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        bytes[offset + i] = static_cast<char>(value >> (8 * i));
    }
}

// Recompute the header checksum after a test edits the header or the table, so that
// the check under test fires rather than the checksum.
void reseal(std::string& bytes) {
    const uint32_t header_bytes = get_u32(bytes, 12);
    const uint64_t table_offset = get_u64(bytes, 96);
    const uint32_t count = get_u32(bytes, 104);
    std::string header = bytes.substr(0, header_bytes);
    put_u32(header, 108, 0);
    uint32_t crc = crc32(header.data(), header.size());
    crc = crc32(bytes.data() + table_offset, size_t{32} * count, crc);
    put_u32(bytes, 108, crc);
}

f2::Header test_header() {
    f2::Header header;
    header.index_kind = f2::IndexKind::Ivf;
    header.metric = f2::Metric::InnerProduct;
    header.dim = 100;
    header.padded_dim = 128;
    header.code_bits = 7;
    header.raw_type = f2::RawType::None;
    header.id_bits = 32;
    header.seed = 42;
    header.num_points = 3;
    return header;
}

const std::string kParamBytes = "params!!";

std::string bulk_bytes() {
    std::string bulk(1000, '\0');
    for (size_t i = 0; i < bulk.size(); ++i) {
        bulk[i] = static_cast<char>(i * 7);
    }
    return bulk;
}

// A checksummed parameter section, an unchecksummed bulk section written in two
// pieces, and an empty section. Layout: header 0-127, table 128-223, parameters at
// 256, bulk at 320 (1000 bytes), the empty section at 1344, which ends the file.
std::vector<f2::SectionWrite> test_sections() {
    const std::string bulk = bulk_bytes();
    return {
        {kParams,
         f2::kSectionRequired | f2::kSectionChecksummed,
         kParamBytes.size(),
         [](f2::SectionSink& sink) { sink.write(kParamBytes.data(), kParamBytes.size()); }},
        {kBulk,
         f2::kSectionRequired,
         bulk.size(),
         [bulk](f2::SectionSink& sink) {
             sink.write(bulk.data(), 400);
             sink.write(bulk.data() + 400, bulk.size() - 400);
         }},
        {kEmpty, f2::kSectionRequired, 0, nullptr}};
}

std::string write(const f2::Header& header, const std::vector<f2::SectionWrite>& sections) {
    std::ostringstream out;
    f2::write_file(out, header, sections, "0.9.0");
    return out.str();
}

std::string valid_file() { return write(test_header(), test_sections()); }

// Byte offset of section table entry `i` when the table follows a 128-byte header.
size_t entry(size_t i) { return 128 + (32 * i); }

f2::Reader open_reader(std::istream& stream, f2::IndexKind kind = f2::IndexKind::Ivf) {
    return f2::Reader(stream, kind, kKnown);
}

void expect_format_error(
    const std::string& bytes,
    const std::string& message,
    f2::IndexKind kind = f2::IndexKind::Ivf
) {
    std::istringstream stream(bytes);
    try {
        f2::Reader reader(stream, kind, kKnown);
        ADD_FAILURE() << "expected FormatError containing: " << message;
    } catch (const f2::FormatError& error) {
        EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
            << "message: " << error.what();
    }
}

// Input and output buffers that cannot seek, like pipes.
class SequentialInput : public std::streambuf {
   public:
    explicit SequentialInput(std::string bytes) : bytes_(std::move(bytes)) {
        setg(bytes_.data(), bytes_.data(), bytes_.data() + bytes_.size());
    }

   private:
    std::string bytes_;
};

class SequentialOutput : public std::streambuf {
   public:
    std::string bytes;

   private:
    int_type overflow(int_type c) override {
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            bytes.push_back(traits_type::to_char_type(c));
        }
        return traits_type::not_eof(c);
    }
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        bytes.append(data, static_cast<size_t>(count));
        return count;
    }
};

TEST(Format2ContainerTest, RoundTripsHeaderAndSections) {
    const std::string bytes = valid_file();
    std::istringstream stream(bytes);
    f2::Reader reader = open_reader(stream);

    const f2::Header& header = reader.header();
    EXPECT_EQ(header.index_kind, f2::IndexKind::Ivf);
    EXPECT_EQ(header.metric, f2::Metric::InnerProduct);
    EXPECT_EQ(header.dim, 100U);
    EXPECT_EQ(header.padded_dim, 128U);
    EXPECT_EQ(header.code_bits, 7U);
    EXPECT_EQ(header.raw_type, f2::RawType::None);
    EXPECT_EQ(header.id_bits, 32U);
    ASSERT_TRUE(header.seed.has_value());
    EXPECT_EQ(*header.seed, 42U);
    EXPECT_EQ(header.num_points, 3U);
    EXPECT_FALSE(reader.read_only());
    EXPECT_EQ(reader.writer_version(), "0.9.0");

    const f2::Section* params = reader.find(kParams);
    const f2::Section* bulk = reader.find(kBulk);
    const f2::Section* empty = reader.find(kEmpty);
    ASSERT_NE(params, nullptr);
    ASSERT_NE(bulk, nullptr);
    ASSERT_NE(empty, nullptr);
    EXPECT_EQ(reader.find(999), nullptr);
    EXPECT_EQ(params->offset, 256U);
    EXPECT_EQ(bulk->offset, 320U);
    EXPECT_EQ(bulk->crc32, 0U);
    EXPECT_EQ(empty->offset, 1344U);  // 320 + 1000, padded to 64
    EXPECT_EQ(bytes.size(), 1344U);   // the empty section ends the file

    std::string read_params(params->size, '\0');
    reader.read(*params, 0, read_params.data(), read_params.size());
    EXPECT_EQ(read_params, kParamBytes);
    std::string read_bulk(bulk->size - 10, '\0');
    reader.read(*bulk, 10, read_bulk.data(), read_bulk.size());
    EXPECT_EQ(read_bulk, bulk_bytes().substr(10));
    EXPECT_THROW(
        reader.read(*bulk, 10, read_bulk.data(), read_bulk.size() + 1), std::out_of_range
    );
}

TEST(Format2ContainerTest, WritesIdenticalBytesTwice) {
    EXPECT_EQ(valid_file(), valid_file());
}

TEST(Format2ContainerTest, WritesAnUnknownSeed) {
    f2::Header header = test_header();
    header.seed.reset();
    const std::string bytes = write(header, {{kEmpty, f2::kSectionRequired, 0, nullptr}});
    EXPECT_EQ(get_u32(bytes, 76), 0U);
    EXPECT_EQ(get_u64(bytes, 80), 0U);
    std::istringstream stream(bytes);
    EXPECT_FALSE(open_reader(stream).header().seed.has_value());
}

TEST(Format2ContainerTest, WritesAndReadsWithoutSeeking) {
    SequentialOutput output_buffer;
    std::ostream output(&output_buffer);
    f2::write_file(output, test_header(), test_sections(), "0.9.0");
    EXPECT_EQ(output_buffer.bytes, valid_file());

    SequentialInput input_buffer(valid_file());
    std::istream input(&input_buffer);
    EXPECT_THROW(f2::has_magic(input, f2::IndexKind::Ivf), f2::FormatError);
    try {
        f2::Reader reader(input, f2::IndexKind::Ivf, kKnown);
        ADD_FAILURE() << "expected FormatError";
    } catch (const f2::FormatError& error) {
        EXPECT_STREQ(error.what(), "IVF index file input must be seekable");
    }
}

TEST(Format2ContainerTest, DetectsTheMagicAndRestoresThePosition) {
    std::istringstream good(valid_file());
    EXPECT_TRUE(f2::has_magic(good, f2::IndexKind::Ivf));
    EXPECT_EQ(good.tellg(), std::streampos(0));
    std::istringstream legacy(std::string("RABQIDX1") + std::string(200, '\0'));
    EXPECT_FALSE(f2::has_magic(legacy, f2::IndexKind::Ivf));

    std::istringstream short_file("RABQ");
    short_file.exceptions(std::ios::failbit | std::ios::badbit);
    EXPECT_FALSE(f2::has_magic(short_file, f2::IndexKind::Ivf));
    EXPECT_EQ(short_file.tellg(), std::streampos(0));
}

TEST(Format2ContainerTest, IgnoresTheStreamExceptionMask) {
    // Truncated in the padding before the last section: the header and the table are
    // read, then a section check fails.
    std::istringstream stream(valid_file().substr(0, 1336));
    stream.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    EXPECT_THROW(open_reader(stream), f2::FormatError);
    std::istringstream valid(valid_file());
    valid.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    EXPECT_NO_THROW(open_reader(valid));
}

TEST(Format2ContainerTest, WriterRejectsInvalidArgumentsBeforeWriting) {
    const f2::SectionWrite empty{kEmpty, f2::kSectionRequired, 0, nullptr};
    const auto expect_rejected = [](const f2::Header& header,
                                    const std::vector<f2::SectionWrite>& sections,
                                    const char* version = "0.9.0") {
        std::ostringstream out;
        EXPECT_THROW(f2::write_file(out, header, sections, version), std::invalid_argument);
        EXPECT_TRUE(out.str().empty());
    };

    f2::Header header = test_header();
    header.padded_dim = 100;
    expect_rejected(header, {empty});
    header = test_header();
    header.index_kind = static_cast<f2::IndexKind>(4);
    expect_rejected(header, {empty});
    for (const int mask : {0, 1, 2}) {
        header = test_header();
        (mask == 0   ? header.incompat_features
         : mask == 1 ? header.ro_compat_features
                     : header.compat_features) = 1;
        expect_rejected(header, {empty});
    }
    expect_rejected(test_header(), {});
    expect_rejected(test_header(), {empty, empty});
    expect_rejected(test_header(), {{0, f2::kSectionRequired, 0, nullptr}});
    expect_rejected(test_header(), {{kEmpty, 4, 0, nullptr}});
    expect_rejected(
        test_header(),
        {{kBulk, 0, 1, nullptr},
         {kEmpty, 0, std::numeric_limits<uint64_t>::max() - 10, nullptr}}
    );
    expect_rejected(test_header(), {empty}, "a version longer than 16");
    expect_rejected(test_header(), {empty}, "0.9\n");
}

TEST(Format2ContainerTest, WriterChecksSectionSizesAndPropagatesFailures) {
    const auto two_bytes = [](f2::SectionSink& sink) { sink.write("ab", 2); };
    for (const uint32_t flags : {0U, f2::kSectionChecksummed}) {
        EXPECT_THROW(
            write(test_header(), {{kBulk, flags, 1, two_bytes}}), std::invalid_argument
        );
        EXPECT_THROW(
            write(test_header(), {{kBulk, flags, 3, two_bytes}}), std::invalid_argument
        );
    }
    const auto fails = [](f2::SectionSink&) { throw std::runtime_error("section failed"); };
    EXPECT_THROW(write(test_header(), {{kBulk, 0, 1, fails}}), std::runtime_error);

    std::ostringstream broken;
    broken.setstate(std::ios::badbit);
    try {
        f2::write_file(broken, test_header(), test_sections(), "0.9.0");
        ADD_FAILURE() << "expected runtime_error";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "IVF index file cannot be written");
    }
}

TEST(Format2ContainerTest, RejectsBrokenHeaders) {
    const std::string good = valid_file();
    expect_format_error(good.substr(0, 127), "IVF index file is too short");

    std::string bytes = good;
    bytes[0] = 'X';
    expect_format_error(bytes, "IVF index file is not in format 2 (bad magic)");

    bytes = good;
    put_u32(bytes, 8, 3);
    reseal(bytes);
    expect_format_error(bytes, "format version 3; this release reads version 2");

    for (const uint32_t size : {120U, 132U, 4096U}) {
        bytes = good;
        put_u32(bytes, 12, size);
        expect_format_error(bytes, "invalid header size " + std::to_string(size));
    }

    bytes = good;
    bytes[50] ^= 1;  // dim, without resealing
    expect_format_error(bytes, "header checksum mismatch");

    bytes = good;
    put_u32(bytes, 104, 0);
    reseal(bytes);
    expect_format_error(bytes, "has no sections");

    for (const uint64_t table_offset : {uint64_t{132}, uint64_t{120}, uint64_t{1344}}) {
        bytes = good;
        put_u64(bytes, 96, table_offset);
        expect_format_error(bytes, "section table lies outside the file");
    }
    bytes = good;
    put_u32(bytes, 104, 1000);
    expect_format_error(bytes, "section table lies outside the file");
}

TEST(Format2ContainerTest, ChecksTheIndexKindBeforeTheSections) {
    std::string bytes = valid_file();
    put_u32(bytes, entry(1), 777);  // an unknown required section
    reseal(bytes);
    expect_format_error(
        bytes, "HNSW index file holds an IVF-RaBitQ index", f2::IndexKind::Hnsw
    );
    expect_format_error(
        bytes,
        "QuantizedGraph index file holds an IVF-RaBitQ index",
        f2::IndexKind::SymphonyQg
    );

    const struct {
        uint32_t kind;
        const char* message;
    } cases[] = {
        {4, "IVF index file holds an IVF-SSD index"},
        {9, "IVF index file has unknown index kind 9"},
        {0, "IVF index file has unknown index kind 0"},
    };
    for (const auto& c : cases) {
        bytes = valid_file();
        put_u32(bytes, 40, c.kind);
        reseal(bytes);
        expect_format_error(bytes, c.message);
    }

    // The index's own header rules also run before the sections.
    bytes = valid_file();
    put_u32(bytes, entry(1), 777);
    reseal(bytes);
    std::istringstream stream(bytes);
    try {
        f2::Reader reader(stream, f2::IndexKind::Ivf, kKnown, [](const f2::Header& header) {
            if (header.id_bits != 64) {
                throw f2::FormatError("IVF index file needs 64-bit IDs");
            }
        });
        ADD_FAILURE() << "expected FormatError";
    } catch (const f2::FormatError& error) {
        EXPECT_STREQ(error.what(), "IVF index file needs 64-bit IDs");
    }
}

TEST(Format2ContainerTest, RejectsInvalidHeaderFields) {
    const std::string good = valid_file();
    const struct {
        size_t offset;
        uint64_t value;
        const char* message;
    } cases[] = {
        {44, 7, "IVF index file has unknown metric 7"},
        {44, 0, "IVF index file has unknown metric 0"},
        {48, 0, "IVF index file has dimension 0"},
        {56, 100, "has padded dimension 100, which must be a multiple of 32"},
        {56,
         96,
         "has padded dimension 96, which must be a multiple of 32 from the dimension 100"},
        {56, 65568, "has padded dimension 65568, which must be a multiple of 32"},
        {64, 10, "has 10 code bits; at most 9 are supported"},
        {68, 5, "has unknown raw element type 5"},
        {72, 16, "has 16-bit IDs; only 32 and 64 are valid"},
        {76, 3, "has unknown header flags 0x3"},
    };
    for (const auto& c : cases) {
        std::string bytes = good;
        if (c.offset == 48 || c.offset == 56) {
            put_u64(bytes, c.offset, c.value);
        } else {
            put_u32(bytes, c.offset, static_cast<uint32_t>(c.value));
        }
        reseal(bytes);
        expect_format_error(bytes, c.message);
    }

    std::string bytes = good;
    put_u32(bytes, 76, 0);  // seed 42 without seed_known
    reseal(bytes);
    expect_format_error(bytes, "has a seed without the seed_known flag");
}

TEST(Format2ContainerTest, AppliesFeatureFlags) {
    const std::string good = valid_file();

    std::string bytes = good;
    put_u64(bytes, 16, 4);
    reseal(bytes);
    expect_format_error(
        bytes,
        "IVF index file uses incompatible features 0x4 that this release does not know; "
        "written by rabitqlib 0.9.0"
    );

    bytes = good;
    put_u64(bytes, 24, 1);
    reseal(bytes);
    std::istringstream read_only(bytes);
    EXPECT_TRUE(open_reader(read_only).read_only());

    bytes = good;
    put_u64(bytes, 32, 0x80);
    reseal(bytes);
    std::istringstream compat(bytes);
    const f2::Reader reader = open_reader(compat);
    EXPECT_FALSE(reader.read_only());
    EXPECT_EQ(reader.header().compat_features, 0x80U);
}

TEST(Format2ContainerTest, SkipsUnknownOptionalSectionsAndRejectsUnknownRequiredOnes) {
    std::string bytes = valid_file();
    put_u32(bytes, entry(1), 777);
    put_u32(bytes, entry(1) + 4, 0);  // optional
    reseal(bytes);
    std::istringstream optional(bytes);
    const f2::Reader reader = open_reader(optional);
    EXPECT_EQ(reader.find(kBulk), nullptr);
    EXPECT_EQ(reader.find(777), nullptr);
    EXPECT_NE(reader.find(kParams), nullptr);

    put_u32(bytes, entry(1) + 4, f2::kSectionRequired);
    reseal(bytes);
    expect_format_error(
        bytes,
        "IVF index file section kind 777 is required but unknown to this release; written "
        "by "
        "rabitqlib 0.9.0"
    );

    // An unknown optional section is not read, so its checksum is not verified.
    bytes = valid_file();
    put_u32(bytes, entry(0), 778);
    put_u32(bytes, entry(0) + 4, f2::kSectionChecksummed);
    reseal(bytes);
    bytes[256] ^= 1;
    std::istringstream unverified(bytes);
    EXPECT_EQ(open_reader(unverified).find(kParams), nullptr);
}

// The valid file with a 136-byte header whose extra bytes come from a later version.
std::string longer_header_file() {
    const std::string good = valid_file();
    std::string bytes = good.substr(0, 128) + std::string(8, '\x5A');
    put_u32(bytes, 12, 136);
    put_u64(bytes, 96, 136);
    bytes += good.substr(128, 96);  // the table
    bytes.resize(256, '\0');        // sections start at 256 in both layouts
    bytes += good.substr(256);
    reseal(bytes);
    return bytes;
}

TEST(Format2ContainerTest, LoadsALongerHeaderAndChecksumsItsExtension) {
    std::string bytes = longer_header_file();
    std::istringstream stream(bytes);
    const f2::Reader reader = open_reader(stream);
    EXPECT_EQ(reader.header().dim, 100U);
    EXPECT_NE(reader.find(kBulk), nullptr);

    bytes[130] ^= 1;
    expect_format_error(bytes, "header checksum mismatch");
}

TEST(Format2ContainerTest, LoadsATableAfterTheSections) {
    const std::string good = valid_file();
    std::string bytes = good.substr(0, 128) + std::string(128, '\0') + good.substr(256);
    put_u64(bytes, 96, bytes.size());  // 1344, a multiple of 8
    bytes += good.substr(128, 96);
    reseal(bytes);
    std::istringstream stream(bytes);
    EXPECT_NE(open_reader(stream).find(kBulk), nullptr);

    // The empty last section now grows into the table.
    put_u64(bytes, 1344 + 64 + 16, 50);
    reseal(bytes);
    expect_format_error(bytes, "section kind 263 overlaps the header or the section table");
}

TEST(Format2ContainerTest, RejectsBrokenSectionEntries) {
    const std::string good = valid_file();
    const struct {
        size_t offset;
        uint64_t value;
        bool wide;
        const char* message;
    } cases[] = {
        {entry(0), 0, false, "IVF index file section kind 0 is reserved"},
        {entry(0) + 4, 0x7, false, "section kind 256 has unknown flags 0x7"},
        {entry(0) + 28, 1, false, "section kind 256 has a non-zero reserved field"},
        {entry(1) + 24, 5, false, "section kind 261 has a checksum but is not checksummed"},
        {entry(1) + 8, 328, true, "section kind 261 is not aligned to 64 bytes"},
        {entry(1) + 16, 1344, true, "section kind 261 lies outside the file"},
        {entry(1) + 16,
         std::numeric_limits<uint64_t>::max(),
         true,
         "261 lies outside the file"},
        {entry(1) + 8,
         std::numeric_limits<uint64_t>::max() - 63,
         true,
         "261 lies outside the file"},
        {entry(1) + 8, 256, true, "section kind 261 overlaps the previous section"},
        {entry(0) + 8,
         128,
         true,
         "section kind 256 overlaps the header or the section table"},
        {entry(0) + 8,
         0,
         true,
         "section kind 256 overlaps the header or the section table"},
        {entry(2), kBulk, false, "section kind 261 is repeated"},
    };
    for (const auto& c : cases) {
        std::string bytes = good;
        if (c.wide) {
            put_u64(bytes, c.offset, c.value);
        } else {
            put_u32(bytes, c.offset, static_cast<uint32_t>(c.value));
        }
        reseal(bytes);
        expect_format_error(bytes, c.message);
    }
}

TEST(Format2ContainerTest, RejectsTruncatedTrailingAndCorruptFiles) {
    const std::string good = valid_file();
    expect_format_error(
        good + std::string(64, '\0'),
        "IVF index file is 1408 bytes, but its sections end at 1344"
    );
    // Truncated inside the padding before the empty last section.
    expect_format_error(good.substr(0, 1336), "section kind 263 lies outside the file");

    std::string bytes = good;
    bytes[256 + 2] ^= 1;  // inside the checksummed parameters
    expect_format_error(bytes, "section kind 256 checksum mismatch (corrupt file)");

    bytes = good;
    bytes[320 + 2] ^= 1;  // bulk data is not checksummed
    std::istringstream stream(bytes);
    EXPECT_NO_THROW(open_reader(stream));
}

}  // namespace
