#ifndef RABITQ_TEST_FORMAT2_FILE_HPP
#define RABITQ_TEST_FORMAT2_FILE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "crc32.hpp"
#include "format2.hpp"

namespace rabitq_test {
namespace format2 = rabitqlib::index_detail::format2;

// Every IVF-RaBitQ section kind, for reading back an IVF file in full.
inline const std::vector<uint32_t> kIvfSections = {
    1, 256, 257, 258, 259, 260, 261, 262, 263};

// A whole format-2 file: the header and each known section with its bytes, in file order.
struct Format2Contents {
    format2::Header header;
    std::string writer_version;
    std::vector<std::pair<format2::Section, std::string>> sections;

    // The bytes of a section; an error if the file has none.
    std::string& bytes(uint32_t kind) {
        for (auto& [section, data] : sections) {
            if (section.kind == kind) {
                return data;
            }
        }
        throw std::runtime_error("test file has no such section");
    }

    void remove(uint32_t kind) {
        sections.erase(
            std::remove_if(
                sections.begin(),
                sections.end(),
                [kind](const auto& entry) { return entry.first.kind == kind; }
            ),
            sections.end()
        );
    }
};

inline Format2Contents read_format2_file(
    const std::string& path,
    format2::IndexKind kind,
    const std::vector<uint32_t>& known_kinds
) {
    Format2Contents contents;
    std::ifstream input(path, std::ios::binary);
    format2::Reader reader(input, kind, known_kinds);
    contents.header = reader.header();
    contents.writer_version = reader.writer_version();
    for (const uint32_t known : known_kinds) {
        if (const format2::Section* section = reader.find(known)) {
            std::string bytes(section->size, '\0');
            reader.read(*section, 0, bytes.data(), bytes.size());
            contents.sections.emplace_back(*section, std::move(bytes));
        }
    }
    std::sort(
        contents.sections.begin(),
        contents.sections.end(),
        [](const auto& a, const auto& b) { return a.first.offset < b.first.offset; }
    );
    return contents;
}

// Write `contents` as a structurally valid file, so that a test can damage what a
// section holds rather than the container.
inline void write_format2_file(const std::string& path, const Format2Contents& contents) {
    std::vector<format2::SectionWrite> writes;
    for (const auto& [section, bytes] : contents.sections) {
        writes.push_back(
            {section.kind,
             section.flags,
             bytes.size(),
             [&data = bytes](format2::SectionSink& sink) {
                 sink.write(data.data(), data.size());
             }}
        );
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    format2::write_file(output, contents.header, writes, contents.writer_version);
}

// The bytes of one section of a format-2 file.
inline std::string read_format2_section(
    const std::string& path,
    format2::IndexKind kind,
    const std::vector<uint32_t>& known_kinds,
    uint32_t section_kind
) {
    return read_format2_file(path, kind, known_kinds).bytes(section_kind);
}

// Rewrite a format-2 file with one section's bytes replaced and every other part kept.
inline void replace_format2_section(
    const std::string& path,
    format2::IndexKind kind,
    const std::vector<uint32_t>& known_kinds,
    uint32_t section_kind,
    const std::string& replacement
) {
    Format2Contents contents = read_format2_file(path, kind, known_kinds);
    contents.bytes(section_kind) = replacement;
    write_format2_file(path, contents);
}

// Overwrite a little-endian header field of `size` bytes and recompute the header
// checksum; the writer refuses some values tests need, such as unknown feature bits.
inline void patch_format2_header(
    const std::string& path, size_t offset, uint64_t value, size_t size
) {
    std::string bytes;
    {
        std::ifstream input(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(input), {});
    }
    for (size_t i = 0; i < size; ++i) {
        bytes[offset + i] = static_cast<char>(value >> (8 * i));
    }
    const auto u32 = [&bytes](size_t at) {
        uint32_t result = 0;
        for (size_t i = 0; i < 4; ++i) {
            result |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[at + i])) << (8 * i);
        }
        return result;
    };
    uint64_t table_offset = 0;
    for (size_t i = 0; i < 8; ++i) {
        table_offset |= static_cast<uint64_t>(static_cast<uint8_t>(bytes[96 + i]))
                        << (8 * i);
    }
    std::string header = bytes.substr(0, u32(12));
    header.replace(108, 4, 4, '\0');
    uint32_t crc = rabitqlib::detail::crc32(header.data(), header.size());
    crc = rabitqlib::detail::crc32(bytes.data() + table_offset, size_t{32} * u32(104), crc);
    for (size_t i = 0; i < 4; ++i) {
        bytes[108 + i] = static_cast<char>(crc >> (8 * i));
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace rabitq_test

#endif  // RABITQ_TEST_FORMAT2_FILE_HPP
