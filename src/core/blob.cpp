#include "blob.hpp"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

#include "crc32.hpp"

namespace rabitqlib::detail {
namespace {

constexpr std::array<char, 8> kBlobMagic = {'R', 'A', 'B', 'Q', 'B', 'L', 'O', 'B'};
constexpr size_t kCommonHeaderBytes = 40;
constexpr size_t kChecksumOffset = 32;
constexpr uint32_t kRotatorKind = 1;
constexpr uint32_t kRotatorVersion = 1;
constexpr uint32_t kSeedKnown = 1;
constexpr uint32_t kMethodFhtKac = 1;
constexpr uint32_t kMethodMatrix = 2;
constexpr uint32_t kScalarFloat32 = 1;
constexpr uint32_t kScalarFloat64 = 2;
constexpr size_t kMaxRotatorDim = rotator_impl::FhtKacRotator::kMaxDim;

constexpr std::array<uint32_t, 256> make_crc_table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t value = i;
        for (int bit = 0; bit < 8; ++bit) {
            value = (value & 1U) != 0 ? 0xEDB88320U ^ (value >> 1) : value >> 1;
        }
        table[i] = value;
    }
    return table;
}

constexpr std::array<uint32_t, 256> kCrcTable = make_crc_table();

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

// The blob checksum: the whole blob with its crc32 field taken as zero.
uint32_t blob_checksum(const uint8_t* blob, size_t size) {
    constexpr std::array<uint8_t, 4> kZero{};
    uint32_t crc = crc32(blob, kChecksumOffset);
    crc = crc32(kZero.data(), kZero.size(), crc);
    return crc32(blob + kChecksumOffset + 4, size - kChecksumOffset - 4, crc);
}

// The same rules and messages as rabitqlib.quant's Rotator.
void check_dimensions(RotatorType type, size_t dim, size_t padded_dim) {
    if (dim == 0 || dim > kMaxRotatorDim) {
        throw std::invalid_argument(
            "dim must be in [1, " + std::to_string(kMaxRotatorDim) + "]; got " +
            std::to_string(dim)
        );
    }
    if (type == RotatorType::FhtKacRotator && dim < 64) {
        throw std::invalid_argument(
            "dim must be at least 64 for method 'fht_kac'; got " + std::to_string(dim)
        );
    }
    if (padded_dim % 32 != 0 || padded_dim < dim || padded_dim > kMaxRotatorDim) {
        throw std::invalid_argument(
            "padded_dim must be a multiple of 32 in [dim, " +
            std::to_string(kMaxRotatorDim) + "]; got " + std::to_string(padded_dim)
        );
    }
}

size_t body_bytes(RotatorType type, size_t dim, size_t padded_dim) {
    return type == RotatorType::FhtKacRotator ? padded_dim / 2
                                              : sizeof(float) * dim * padded_dim;
}

}  // namespace

uint32_t crc32(const void* data, size_t size, uint32_t crc) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) {
        crc = kCrcTable[(crc ^ bytes[i]) & 0xFFU] ^ (crc >> 8);
    }
    return ~crc;
}

size_t rotator_blob_bytes(RotatorType type, size_t dim, size_t padded_dim) {
    return kRotatorBlobHeaderBytes + body_bytes(type, dim, padded_dim);
}

void encode_rotator_blob(
    const Rotator<float>& rotator,
    RotatorType type,
    size_t dim,
    std::optional<uint32_t> seed,
    uint8_t* out
) {
    const size_t padded_dim = rotator.size();
    check_dimensions(type, dim, padded_dim);
    const size_t body = body_bytes(type, dim, padded_dim);
    if (rotator.dump_bytes() != body) {
        throw std::invalid_argument("rotator does not match its type and dimensions");
    }
    const size_t size = kRotatorBlobHeaderBytes + body;
    std::memset(out, 0, kRotatorBlobHeaderBytes);
    std::memcpy(out, kBlobMagic.data(), kBlobMagic.size());
    put_u32(out + 8, kRotatorKind);
    put_u32(out + 12, kRotatorVersion);
    put_u32(out + 16, static_cast<uint32_t>(kRotatorBlobHeaderBytes));
    put_u32(out + 20, seed.has_value() ? kSeedKnown : 0);
    put_u64(out + 24, body);
    put_u32(out + 40, type == RotatorType::FhtKacRotator ? kMethodFhtKac : kMethodMatrix);
    put_u32(out + 44, kScalarFloat32);
    put_u32(out + 48, static_cast<uint32_t>(dim));
    put_u32(out + 52, static_cast<uint32_t>(padded_dim));
    put_u32(out + 56, seed.value_or(0));
    rotator.save(reinterpret_cast<char*>(out + kRotatorBlobHeaderBytes));
    put_u32(out + kChecksumOffset, blob_checksum(out, size));
}

DecodedRotator decode_rotator_blob(const uint8_t* data, size_t size) {
    if (size < kCommonHeaderBytes) {
        throw std::invalid_argument("data is too short for a rabitqlib blob");
    }
    if (std::memcmp(data, kBlobMagic.data(), kBlobMagic.size()) != 0) {
        throw std::invalid_argument("data is not a rabitqlib blob (bad magic)");
    }
    const uint32_t kind = get_u32(data + 8);
    if (kind != kRotatorKind) {
        throw std::invalid_argument(
            "data is not a rotator blob; got kind " + std::to_string(kind)
        );
    }
    const uint32_t version = get_u32(data + 12);
    if (version != kRotatorVersion) {
        throw std::invalid_argument(
            "data has unsupported rotator format version " + std::to_string(version) +
            "; this release reads version " + std::to_string(kRotatorVersion)
        );
    }
    if (get_u32(data + 16) != kRotatorBlobHeaderBytes) {
        throw std::invalid_argument(
            "data has an invalid header size for rotator version 1"
        );
    }
    const uint64_t body = get_u64(data + 24);
    if (size < kRotatorBlobHeaderBytes || body != size - kRotatorBlobHeaderBytes) {
        throw std::invalid_argument(
            "data length does not match its header (truncated or padded)"
        );
    }
    if (get_u32(data + kChecksumOffset) != blob_checksum(data, size)) {
        throw std::invalid_argument("data checksum mismatch (corrupt blob)");
    }
    const uint32_t flags = get_u32(data + 20);
    if ((flags & ~kSeedKnown) != 0) {
        throw std::invalid_argument("data has unknown flags");
    }
    if (get_u32(data + 36) != 0 || get_u32(data + 60) != 0) {
        throw std::invalid_argument("data has non-zero reserved fields");
    }
    const uint32_t method = get_u32(data + 40);
    if (method != kMethodFhtKac && method != kMethodMatrix) {
        throw std::invalid_argument(
            "data has an unknown rotator method " + std::to_string(method)
        );
    }
    const uint32_t scalar = get_u32(data + 44);
    if (scalar == kScalarFloat64) {
        throw std::invalid_argument(
            "data stores a float64 rotator, which version 1 cannot load"
        );
    }
    if (scalar != kScalarFloat32) {
        throw std::invalid_argument(
            "data has an unknown scalar type " + std::to_string(scalar)
        );
    }

    DecodedRotator result;
    result.type =
        method == kMethodFhtKac ? RotatorType::FhtKacRotator : RotatorType::MatrixRotator;
    result.dim = get_u32(data + 48);
    result.padded_dim = get_u32(data + 52);
    try {
        check_dimensions(result.type, result.dim, result.padded_dim);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            std::string("data has invalid rotator dimensions: ") + error.what()
        );
    }
    const uint32_t seed = get_u32(data + 56);
    if ((flags & kSeedKnown) != 0) {
        result.seed = seed;
    } else if (seed != 0) {
        throw std::invalid_argument("data has a seed without the seed_known flag");
    }
    if (body != body_bytes(result.type, result.dim, result.padded_dim)) {
        throw std::invalid_argument("data body size does not match the rotator parameters");
    }
    result.rotator =
        rotator_impl::make_rotator_for_loading(result.dim, result.type, result.padded_dim);
    result.rotator->load(reinterpret_cast<const char*>(data + kRotatorBlobHeaderBytes));
    return result;
}

}  // namespace rabitqlib::detail
