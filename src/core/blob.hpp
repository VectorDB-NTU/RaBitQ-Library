#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/core/types.hpp"

// The self-describing blobs of docs/formats.md ("Blobs"), for C++ readers and writers
// such as index files; an implementation detail, not installed. rabitqlib.quant keeps
// its own rotator blob code, which uses zlib's faster checksum; both are tested against
// the same golden files.
namespace rabitqlib::detail {

inline constexpr size_t kRotatorBlobHeaderBytes = 64;

// The size of the rotator blob for these parameters, which must be valid.
size_t rotator_blob_bytes(RotatorType type, size_t dim, size_t padded_dim);

// Write the blob of `rotator`, which has the given type and dimensions, to `out`,
// which holds rotator_blob_bytes(type, dim, rotator.size()) bytes. Dimensions the
// format does not allow raise std::invalid_argument.
void encode_rotator_blob(
    const Rotator<float>& rotator,
    RotatorType type,
    size_t dim,
    std::optional<uint32_t> seed,
    uint8_t* out
);

struct DecodedRotator {
    RotatorType type = RotatorType::FhtKacRotator;
    size_t dim = 0;
    size_t padded_dim = 0;
    std::optional<uint32_t> seed;
    std::unique_ptr<Rotator<float>> rotator;
};

// Validate and decode a rotator blob of `size` bytes. Invalid data raises
// std::invalid_argument with the messages rabitqlib.quant uses, which start with "data".
DecodedRotator decode_rotator_blob(const uint8_t* data, size_t size);

}  // namespace rabitqlib::detail
