#include "format2.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

#include "../core/crc32.hpp"

namespace rabitqlib::index_detail::format2 {
namespace {

using rabitqlib::detail::crc32;

constexpr uint32_t kHeaderFlagSeedKnown = 1U << 0;
constexpr uint32_t kKnownSectionFlags = kSectionRequired | kSectionChecksummed;
constexpr size_t kChecksumOffset = 108;
constexpr size_t kChunkBytes = size_t{1} << 16;

void put_u32(uint8_t* out, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void put_u64(uint8_t* out, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

uint32_t get_u32(const uint8_t* in) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(in[i]) << (8 * i);
    }
    return value;
}

uint64_t get_u64(const uint8_t* in) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(in[i]) << (8 * i);
    }
    return value;
}

// Round up to a multiple of kSectionAlignment; the caller has bounded `value`.
uint64_t align_section(uint64_t value) {
    return (value + kSectionAlignment - 1) / kSectionAlignment * kSectionAlignment;
}

std::string hex(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << value;
    return out.str();
}

// The error-message component of each index, as in docs/development.md.
const char* component(IndexKind kind) {
    switch (kind) {
        case IndexKind::Ivf:
            return "IVF";
        case IndexKind::Hnsw:
            return "HNSW";
        case IndexKind::SymphonyQg:
            return "QuantizedGraph";
    }
    return "Index";
}

const char* kind_name(uint32_t kind) {
    switch (kind) {
        case 1:
            return "an IVF-RaBitQ";
        case 2:
            return "an HNSW-RaBitQ";
        case 3:
            return "a SymphonyQG";
        case 4:
            return "an IVF-SSD";
        default:
            return nullptr;
    }
}

// The header rules that hold for every index kind, as the text after "index file",
// or an empty string. The index kind itself is checked by the caller.
std::string header_error(const Header& header) {
    const auto metric = static_cast<uint32_t>(header.metric);
    if (metric < 1 || metric > 3) {
        return "has unknown metric " + std::to_string(metric);
    }
    if (header.dim == 0) {
        return "has dimension 0";
    }
    if (header.padded_dim % 32 != 0 || header.padded_dim < header.dim ||
        header.padded_dim > kMaxPaddedDim) {
        return "has padded dimension " + std::to_string(header.padded_dim) +
               ", which must be a multiple of 32 from the dimension " +
               std::to_string(header.dim) + " to 65536";
    }
    if (header.code_bits > 9) {
        return "has " + std::to_string(header.code_bits) +
               " code bits; at most 9 are supported";
    }
    const auto raw_type = static_cast<uint32_t>(header.raw_type);
    if (raw_type > 2) {
        return "has unknown raw element type " + std::to_string(raw_type);
    }
    if (header.id_bits != 32 && header.id_bits != 64) {
        return "has " + std::to_string(header.id_bits) +
               "-bit IDs; only 32 and 64 are valid";
    }
    return {};
}

void encode_header(
    const Header& header,
    uint64_t section_count,
    std::string_view writer_version,
    uint8_t* out
) {
    std::memset(out, 0, kHeaderBytes);
    std::memcpy(out, kMagic.data(), kMagic.size());
    put_u32(out + 8, kVersion);
    put_u32(out + 12, kHeaderBytes);
    put_u64(out + 16, header.incompat_features);
    put_u64(out + 24, header.ro_compat_features);
    put_u64(out + 32, header.compat_features);
    put_u32(out + 40, static_cast<uint32_t>(header.index_kind));
    put_u32(out + 44, static_cast<uint32_t>(header.metric));
    put_u64(out + 48, header.dim);
    put_u64(out + 56, header.padded_dim);
    put_u32(out + 64, header.code_bits);
    put_u32(out + 68, static_cast<uint32_t>(header.raw_type));
    put_u32(out + 72, header.id_bits);
    put_u32(out + 76, header.seed.has_value() ? kHeaderFlagSeedKnown : 0);
    put_u64(out + 80, header.seed.value_or(0));
    put_u64(out + 88, header.num_points);
    put_u64(out + 96, kHeaderBytes);
    put_u32(out + 104, static_cast<uint32_t>(section_count));
    std::memcpy(out + 112, writer_version.data(), writer_version.size());
}

void encode_section(const Section& section, uint8_t* out) {
    std::memset(out, 0, kSectionEntryBytes);
    put_u32(out, section.kind);
    put_u32(out + 4, section.flags);
    put_u64(out + 8, section.offset);
    put_u64(out + 16, section.size);
    put_u32(out + 24, section.crc32);
}

// The header checksum: the header with its checksum field taken as zero, then the table.
uint32_t header_checksum(
    const std::vector<uint8_t>& header_bytes, const std::vector<uint8_t>& table_bytes
) {
    constexpr std::array<uint8_t, 4> kZero{};
    uint32_t crc = crc32(header_bytes.data(), kChecksumOffset);
    crc = crc32(kZero.data(), kZero.size(), crc);
    crc = crc32(
        header_bytes.data() + kChecksumOffset + 4,
        header_bytes.size() - kChecksumOffset - 4,
        crc
    );
    return crc32(table_bytes.data(), table_bytes.size(), crc);
}

void write_zeros(std::ostream& output, uint64_t count) {
    static const std::array<char, kSectionAlignment> kZeros{};
    while (count > 0) {
        const auto chunk = static_cast<size_t>(std::min<uint64_t>(count, kZeros.size()));
        output.write(kZeros.data(), static_cast<std::streamsize>(chunk));
        count -= chunk;
    }
}

}  // namespace

void SectionSink::write(const void* data, size_t size) {
    if (size > remaining_) {
        throw std::invalid_argument("index section writer wrote more bytes than declared");
    }
    output_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    remaining_ -= size;
}

void write_file(
    std::ostream& output,
    const Header& header,
    const std::vector<SectionWrite>& sections,
    std::string_view writer_version
) {
    const std::string name = component(header.index_kind);
    const auto kind = static_cast<uint32_t>(header.index_kind);
    if (kind < 1 || kind > 3) {
        throw std::invalid_argument(
            name + " index file cannot have index kind " + std::to_string(kind)
        );
    }
    if (const std::string error = header_error(header); !error.empty()) {
        throw std::invalid_argument(name + " index file " + error);
    }
    if ((header.incompat_features & ~kKnownIncompatFeatures) != 0 ||
        (header.ro_compat_features & ~kKnownRoCompatFeatures) != 0 ||
        (header.compat_features & ~kKnownCompatFeatures) != 0) {
        throw std::invalid_argument(name + " index file cannot use unknown feature bits");
    }
    if (sections.empty()) {
        throw std::invalid_argument(name + " index file needs at least one section");
    }
    const bool printable =
        std::all_of(writer_version.begin(), writer_version.end(), [](char c) {
            return c >= 0x20 && c <= 0x7E;
        });
    if (writer_version.size() > kWriterVersionBytes || !printable) {
        throw std::invalid_argument(
            name +
            " index file writer version must be at most 16 printable ASCII characters"
        );
    }

    std::vector<Section> entries(sections.size());
    std::unordered_set<uint32_t> kinds;
    const uint64_t table_end = kHeaderBytes + (kSectionEntryBytes * sections.size());
    uint64_t position = table_end;
    for (size_t i = 0; i < sections.size(); ++i) {
        const SectionWrite& section = sections[i];
        if (section.kind == 0) {
            throw std::invalid_argument(name + " index file section kind 0 is reserved");
        }
        if ((section.flags & ~kKnownSectionFlags) != 0) {
            throw std::invalid_argument(name + " index file section has unknown flags");
        }
        if (!kinds.insert(section.kind).second) {
            throw std::invalid_argument(
                name + " index file section kind " + std::to_string(section.kind) +
                " is repeated"
            );
        }
        position = align_section(position);
        if (section.size >
            std::numeric_limits<uint64_t>::max() - kSectionAlignment - position) {
            throw std::invalid_argument(
                name + " index file sections exceed the offset range"
            );
        }
        entries[i].kind = section.kind;
        entries[i].flags = section.flags;
        entries[i].offset = position;
        entries[i].size = section.size;
        position += section.size;
    }

    // Run a section's writer and check that it wrote exactly the declared size.
    const auto write_section = [&name](const SectionWrite& section, SectionSink& sink) {
        if (section.write) {
            section.write(sink);
        }
        if (sink.remaining() != 0) {
            throw std::invalid_argument(
                name + " index file section kind " + std::to_string(section.kind) +
                " writer wrote fewer bytes than declared"
            );
        }
    };

    // Checksummed sections are small: buffer them, so that the header and the table,
    // which hold their checksums, can be written first.
    std::vector<std::string> buffered(sections.size());
    for (size_t i = 0; i < sections.size(); ++i) {
        if ((entries[i].flags & kSectionChecksummed) == 0) {
            continue;
        }
        std::ostringstream buffer;
        SectionSink sink(buffer, entries[i].size);
        write_section(sections[i], sink);
        buffered[i] = buffer.str();
        entries[i].crc32 = crc32(buffered[i].data(), buffered[i].size());
    }

    std::vector<uint8_t> header_bytes(kHeaderBytes);
    encode_header(header, sections.size(), writer_version, header_bytes.data());
    std::vector<uint8_t> table_bytes(kSectionEntryBytes * sections.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        encode_section(entries[i], table_bytes.data() + (i * kSectionEntryBytes));
    }
    put_u32(
        header_bytes.data() + kChecksumOffset, header_checksum(header_bytes, table_bytes)
    );

    output.write(
        reinterpret_cast<const char*>(header_bytes.data()),
        static_cast<std::streamsize>(header_bytes.size())
    );
    output.write(
        reinterpret_cast<const char*>(table_bytes.data()),
        static_cast<std::streamsize>(table_bytes.size())
    );
    uint64_t written = table_end;
    for (size_t i = 0; i < sections.size(); ++i) {
        write_zeros(output, entries[i].offset - written);
        if ((entries[i].flags & kSectionChecksummed) != 0) {
            output.write(
                buffered[i].data(), static_cast<std::streamsize>(buffered[i].size())
            );
        } else {
            SectionSink sink(output, entries[i].size);
            write_section(sections[i], sink);
        }
        written = entries[i].offset + entries[i].size;
    }
    output.flush();
    if (!output) {
        throw std::runtime_error(name + " index file cannot be written");
    }
}

bool has_magic(std::istream& input, IndexKind kind) {
    std::streambuf& buffer = *input.rdbuf();
    const auto start = buffer.pubseekoff(0, std::ios::cur, std::ios::in);
    if (start == std::streampos(std::streamoff(-1))) {
        throw FormatError(
            std::string(component(kind)) + " index file input must be seekable"
        );
    }
    std::array<char, kMagic.size()> magic{};
    const std::streamsize count = buffer.sgetn(magic.data(), magic.size());
    buffer.pubseekpos(start, std::ios::in);
    return count == static_cast<std::streamsize>(magic.size()) && magic == kMagic;
}

Reader::Reader(
    std::istream& input,
    IndexKind expected_kind,
    const std::vector<uint32_t>& known_kinds,
    const std::function<void(const Header&)>& check_header
)
    : input_(*input.rdbuf())
    , component_(std::string(component(expected_kind)) + " index file") {
    const auto end = input_.pubseekoff(0, std::ios::end, std::ios::in);
    if (end == std::streampos(std::streamoff(-1))) {
        throw FormatError(component_ + " input must be seekable");
    }
    const auto file_size = static_cast<uint64_t>(static_cast<std::streamoff>(end));

    // 1. Magic, version and header size.
    if (file_size < kHeaderBytes) {
        throw FormatError(component_ + " is too short for a format-2 header");
    }
    std::vector<uint8_t> header_bytes(kHeaderBytes);
    read_at(0, header_bytes.data(), kHeaderBytes, "header");
    if (std::memcmp(header_bytes.data(), kMagic.data(), kMagic.size()) != 0) {
        throw FormatError(component_ + " is not in format 2 (bad magic)");
    }
    const uint32_t version = get_u32(header_bytes.data() + 8);
    if (version != kVersion) {
        throw FormatError(
            component_ + " has format version " + std::to_string(version) +
            "; this release reads version 2"
        );
    }
    const uint32_t header_size = get_u32(header_bytes.data() + 12);
    if (header_size < kHeaderBytes || header_size % 8 != 0 || header_size > file_size) {
        throw FormatError(
            component_ + " has an invalid header size " + std::to_string(header_size)
        );
    }
    header_bytes.resize(header_size);
    read_at(
        kHeaderBytes,
        header_bytes.data() + kHeaderBytes,
        header_size - kHeaderBytes,
        "header"
    );

    // 2. The section table lies within the file, and the header checksum matches.
    const uint8_t* fields = header_bytes.data();
    const uint64_t table_offset = get_u64(fields + 96);
    const uint32_t section_count = get_u32(fields + 104);
    if (section_count == 0) {
        throw FormatError(component_ + " has no sections");
    }
    if (table_offset % 8 != 0 || table_offset < header_size || table_offset > file_size ||
        section_count > (file_size - table_offset) / kSectionEntryBytes) {
        throw FormatError(component_ + " section table lies outside the file");
    }
    const uint64_t table_end = table_offset + (kSectionEntryBytes * section_count);
    std::vector<uint8_t> table_bytes(kSectionEntryBytes * section_count);
    read_at(table_offset, table_bytes.data(), table_bytes.size(), "section table");
    if (get_u32(fields + kChecksumOffset) != header_checksum(header_bytes, table_bytes)) {
        throw FormatError(component_ + " header checksum mismatch (corrupt file)");
    }

    // 3. Feature flags.
    const char* const writer = reinterpret_cast<const char*>(fields + 112);
    writer_version_.assign(writer, std::find(writer, writer + kWriterVersionBytes, '\0'));
    header_.incompat_features = get_u64(fields + 16);
    header_.ro_compat_features = get_u64(fields + 24);
    header_.compat_features = get_u64(fields + 32);
    const uint64_t unknown_incompat = header_.incompat_features & ~kKnownIncompatFeatures;
    if (unknown_incompat != 0) {
        throw FormatError(
            component_ + " uses incompatible features " + hex(unknown_incompat) +
            " that this release does not know; written by rabitqlib " + writer_version_
        );
    }
    read_only_ = (header_.ro_compat_features & ~kKnownRoCompatFeatures) != 0;

    // 4. The index kind, the header rules of every kind, then the index's own.
    const uint32_t kind = get_u32(fields + 40);
    if (kind != static_cast<uint32_t>(expected_kind)) {
        const char* name = kind_name(kind);
        if (name == nullptr) {
            throw FormatError(
                component_ + " has unknown index kind " + std::to_string(kind)
            );
        }
        throw FormatError(component_ + " holds " + name + " index");
    }
    header_.index_kind = expected_kind;
    header_.metric = static_cast<Metric>(get_u32(fields + 44));
    header_.dim = get_u64(fields + 48);
    header_.padded_dim = get_u64(fields + 56);
    header_.code_bits = get_u32(fields + 64);
    header_.raw_type = static_cast<RawType>(get_u32(fields + 68));
    header_.id_bits = get_u32(fields + 72);
    const uint32_t header_flags = get_u32(fields + 76);
    const uint64_t seed = get_u64(fields + 80);
    header_.num_points = get_u64(fields + 88);
    if ((header_flags & ~kHeaderFlagSeedKnown) != 0) {
        throw FormatError(component_ + " has unknown header flags " + hex(header_flags));
    }
    if ((header_flags & kHeaderFlagSeedKnown) != 0) {
        header_.seed = seed;
    } else if (seed != 0) {
        throw FormatError(component_ + " has a seed without the seed_known flag");
    }
    if (const std::string error = header_error(header_); !error.empty()) {
        throw FormatError(component_ + " " + error);
    }
    if (check_header) {
        check_header(header_);
    }

    // 5. Each entry in table order: its fields, alignment, bounds, order, overlap and
    // repetition, then whether its kind is known.
    std::unordered_set<uint32_t> kinds;
    uint64_t previous_end = 0;
    uint64_t content_end = table_end;
    for (uint32_t i = 0; i < section_count; ++i) {
        const uint8_t* entry = table_bytes.data() + (i * kSectionEntryBytes);
        Section section;
        section.kind = get_u32(entry);
        section.flags = get_u32(entry + 4);
        section.offset = get_u64(entry + 8);
        section.size = get_u64(entry + 16);
        section.crc32 = get_u32(entry + 24);
        const std::string name =
            component_ + " section kind " + std::to_string(section.kind);
        if (section.kind == 0) {
            throw FormatError(name + " is reserved");
        }
        if ((section.flags & ~kKnownSectionFlags) != 0) {
            throw FormatError(name + " has unknown flags " + hex(section.flags));
        }
        if (get_u32(entry + 28) != 0) {
            throw FormatError(name + " has a non-zero reserved field");
        }
        if ((section.flags & kSectionChecksummed) == 0 && section.crc32 != 0) {
            throw FormatError(name + " has a checksum but is not checksummed");
        }
        if (section.offset % kSectionAlignment != 0) {
            throw FormatError(name + " is not aligned to 64 bytes");
        }
        if (section.offset > file_size || section.size > file_size - section.offset) {
            throw FormatError(name + " lies outside the file");
        }
        const uint64_t end_offset = section.offset + section.size;
        if (section.offset < previous_end) {
            throw FormatError(name + " overlaps the previous section or is out of order");
        }
        const bool overlaps_table =
            section.size != 0 && section.offset < table_end && table_offset < end_offset;
        if (section.offset < header_size || overlaps_table) {
            throw FormatError(name + " overlaps the header or the section table");
        }
        if (!kinds.insert(section.kind).second) {
            throw FormatError(name + " is repeated");
        }
        previous_end = end_offset;
        content_end = std::max(content_end, end_offset);

        if (std::find(known_kinds.begin(), known_kinds.end(), section.kind) ==
            known_kinds.end()) {
            if ((section.flags & kSectionRequired) != 0) {
                throw FormatError(
                    name +
                    " is required but unknown to this release; written by rabitqlib " +
                    writer_version_
                );
            }
            continue;
        }
        sections_.push_back(section);
    }

    // 6. The file ends where the table or the last section ends.
    if (content_end != file_size) {
        throw FormatError(
            component_ + " is " + std::to_string(file_size) +
            " bytes, but its sections end at " + std::to_string(content_end)
        );
    }

    // 7. Known checksummed sections, streamed without holding them in memory.
    std::vector<char> chunk;
    for (const Section& section : sections_) {
        if ((section.flags & kSectionChecksummed) == 0) {
            continue;
        }
        chunk.resize(static_cast<size_t>(std::min<uint64_t>(section.size, kChunkBytes)));
        uint32_t crc = 0;
        for (uint64_t done = 0; done < section.size;) {
            const auto count =
                static_cast<size_t>(std::min<uint64_t>(section.size - done, kChunkBytes));
            read(section, done, chunk.data(), count);
            crc = crc32(chunk.data(), count, crc);
            done += count;
        }
        if (crc != section.crc32) {
            throw FormatError(
                component_ + " section kind " + std::to_string(section.kind) +
                " checksum mismatch (corrupt file)"
            );
        }
    }
}

const Section* Reader::find(uint32_t kind) const {
    for (const Section& section : sections_) {
        if (section.kind == kind) {
            return &section;
        }
    }
    return nullptr;
}

void Reader::read(const Section& section, uint64_t offset, void* out, size_t size) {
    if (offset > section.size || size > section.size - offset) {
        throw std::out_of_range(component_ + " read past the end of a section");
    }
    read_at(
        section.offset + offset, out, size, "section kind " + std::to_string(section.kind)
    );
}

SectionInput Reader::open(const Section& section) {
    const std::string name = component_ + " section kind " + std::to_string(section.kind);
    const auto offset = static_cast<std::streamoff>(section.offset);
    if (input_.pubseekpos(offset, std::ios::in) != std::streampos(offset)) {
        throw FormatError(name + " cannot be read");
    }
    return SectionInput(input_, name, section.size);
}

void SectionInput::read(void* out, size_t size) {
    if (size > remaining_) {
        throw std::out_of_range(name_ + " read past its end");
    }
    if (size != 0 &&
        input_.sgetn(static_cast<char*>(out), static_cast<std::streamsize>(size)) !=
            static_cast<std::streamsize>(size)) {
        throw FormatError(name_ + " cannot be read");
    }
    remaining_ -= size;
}

void Reader::read_at(uint64_t offset, void* out, size_t size, const std::string& what) {
    if (size == 0) {
        return;
    }
    const auto position =
        input_.pubseekpos(static_cast<std::streamoff>(offset), std::ios::in);
    if (position != std::streampos(static_cast<std::streamoff>(offset)) ||
        input_.sgetn(static_cast<char*>(out), static_cast<std::streamsize>(size)) !=
            static_cast<std::streamsize>(size)) {
        throw FormatError(component_ + " " + what + " cannot be read");
    }
}

}  // namespace rabitqlib::index_detail::format2
