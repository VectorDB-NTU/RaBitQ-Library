#include "blob.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "crc32.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/core/types.hpp"

namespace {
using rabitqlib::RotatorType;
namespace detail = rabitqlib::detail;

std::vector<uint8_t> read_fixture(const std::string& name) {
    std::ifstream input(
        std::string(RABITQ_QUANT_FIXTURE_ROOT) + "/" + name, std::ios::binary
    );
    EXPECT_TRUE(input.is_open()) << name;
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<uint8_t> encode(
    const rabitqlib::Rotator<float>& rotator,
    RotatorType type,
    size_t dim,
    std::optional<uint32_t> seed
) {
    std::vector<uint8_t> blob(detail::rotator_blob_bytes(type, dim, rotator.size()));
    detail::encode_rotator_blob(rotator, type, dim, seed, blob.data());
    return blob;
}

std::vector<float> rotate(const rabitqlib::Rotator<float>& rotator, size_t dim) {
    std::vector<float> input(dim);
    for (size_t i = 0; i < dim; ++i) {
        input[i] = static_cast<float>(i % 7) - 3.0F;
    }
    std::vector<float> output(rotator.size());
    rotator.rotate(input.data(), output.data());
    return output;
}

void put_u32(std::vector<uint8_t>& blob, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        blob[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

// Recompute the checksum after an edit, so that the check under test fires.
void reseal(std::vector<uint8_t>& blob) {
    put_u32(blob, 32, 0);
    put_u32(blob, 32, detail::crc32(blob.data(), blob.size()));
}

void expect_rejected(const std::vector<uint8_t>& blob, const std::string& message) {
    try {
        detail::decode_rotator_blob(blob.data(), blob.size());
        ADD_FAILURE() << "expected std::invalid_argument: " << message;
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), message);
    }
}

TEST(CoreBlobTest, Crc32MatchesZlib) {
    EXPECT_EQ(detail::crc32("", 0), 0U);
    EXPECT_EQ(detail::crc32("123456789", 9), 0xCBF43926U);
    std::vector<uint8_t> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i);
    }
    EXPECT_EQ(detail::crc32(data.data(), data.size()), 0xB70B4C26U);  // zlib.crc32
    const uint32_t first = detail::crc32(data.data(), 100);
    EXPECT_EQ(detail::crc32(data.data() + 100, data.size() - 100, first), 0xB70B4C26U);
}

// The golden files rabitqlib.quant writes: the C++ codec reads and rewrites them
// byte for byte, and they hold the same rotation as a freshly seeded rotator.
TEST(CoreBlobTest, ReadsAndRewritesTheQuantGoldenRotators) {
    const struct {
        const char* name;
        size_t dim;
        size_t padded_dim;
        uint32_t seed;
    } cases[] = {
        {"rotator_fht_kac_dim64_seed0.bin", 64, 64, 0},
        {"rotator_fht_kac_dim100_padded128_seed42.bin", 100, 128, 42},
    };
    for (const auto& c : cases) {
        const std::vector<uint8_t> golden = read_fixture(c.name);
        const detail::DecodedRotator decoded =
            detail::decode_rotator_blob(golden.data(), golden.size());
        EXPECT_EQ(decoded.type, RotatorType::FhtKacRotator);
        EXPECT_EQ(decoded.dim, c.dim);
        EXPECT_EQ(decoded.padded_dim, c.padded_dim);
        ASSERT_TRUE(decoded.seed.has_value());
        EXPECT_EQ(*decoded.seed, c.seed);
        EXPECT_EQ(
            encode(*decoded.rotator, decoded.type, decoded.dim, decoded.seed), golden
        );

        const std::unique_ptr<rabitqlib::Rotator<float>> fresh(
            rabitqlib::choose_rotator<float>(
                c.dim, RotatorType::FhtKacRotator, c.padded_dim, c.seed
            )
        );
        EXPECT_EQ(rotate(*decoded.rotator, c.dim), rotate(*fresh, c.dim));
    }
}

TEST(CoreBlobTest, RoundTripsAMatrixRotatorWithoutASeed) {
    const std::unique_ptr<rabitqlib::Rotator<float>> rotator(
        rabitqlib::choose_rotator<float>(40, RotatorType::MatrixRotator, 64, 7)
    );
    const std::vector<uint8_t> blob = encode(*rotator, RotatorType::MatrixRotator, 40, {});
    EXPECT_EQ(blob.size(), 64 + (4 * 40 * 64));
    const detail::DecodedRotator decoded =
        detail::decode_rotator_blob(blob.data(), blob.size());
    EXPECT_EQ(decoded.type, RotatorType::MatrixRotator);
    EXPECT_EQ(decoded.dim, 40U);
    EXPECT_EQ(decoded.padded_dim, 64U);
    EXPECT_FALSE(decoded.seed.has_value());
    EXPECT_EQ(rotate(*decoded.rotator, 40), rotate(*rotator, 40));
}

TEST(CoreBlobTest, EncoderRejectsDimensionsTheFormatDoesNotAllow) {
    const std::unique_ptr<rabitqlib::Rotator<float>> rotator(
        rabitqlib::choose_rotator<float>(40, RotatorType::MatrixRotator, 64, 7)
    );
    std::vector<uint8_t> blob(
        detail::rotator_blob_bytes(RotatorType::MatrixRotator, 40, 64)
    );
    EXPECT_THROW(
        detail::encode_rotator_blob(
            *rotator, RotatorType::FhtKacRotator, 40, {}, blob.data()
        ),
        std::invalid_argument
    );
    EXPECT_THROW(
        detail::encode_rotator_blob(
            *rotator, RotatorType::MatrixRotator, 41, {}, blob.data()
        ),
        std::invalid_argument
    );
}

TEST(CoreBlobTest, RejectsInvalidBlobsWithTheQuantMessages) {
    const std::vector<uint8_t> golden = read_fixture("rotator_fht_kac_dim64_seed0.bin");
    const auto edited = [&golden](size_t offset, uint32_t value) {
        std::vector<uint8_t> blob = golden;
        put_u32(blob, offset, value);
        reseal(blob);
        return blob;
    };

    expect_rejected(
        std::vector<uint8_t>(golden.begin(), golden.begin() + 39),
        "data is too short for a rabitqlib blob"
    );
    std::vector<uint8_t> blob = golden;
    blob[0] = 'X';
    expect_rejected(blob, "data is not a rabitqlib blob (bad magic)");
    expect_rejected(edited(8, 3), "data is not a rotator blob; got kind 3");
    expect_rejected(
        edited(12, 2),
        "data has unsupported rotator format version 2; this release reads version 1"
    );
    expect_rejected(
        edited(16, 72), "data has an invalid header size for rotator version 1"
    );
    expect_rejected(
        std::vector<uint8_t>(golden.begin(), golden.end() - 1),
        "data length does not match its header (truncated or padded)"
    );
    blob = golden;
    blob[70] ^= 1;
    expect_rejected(blob, "data checksum mismatch (corrupt blob)");
    expect_rejected(edited(20, 2), "data has unknown flags");
    expect_rejected(edited(60, 1), "data has non-zero reserved fields");
    expect_rejected(edited(40, 3), "data has an unknown rotator method 3");
    expect_rejected(
        edited(44, 2), "data stores a float64 rotator, which version 1 cannot load"
    );
    expect_rejected(edited(44, 9), "data has an unknown scalar type 9");
    expect_rejected(
        edited(48, 32),
        "data has invalid rotator dimensions: dim must be at least 64 for method "
        "'fht_kac'; got 32"
    );
    std::vector<uint8_t> seeded =
        read_fixture("rotator_fht_kac_dim100_padded128_seed42.bin");
    put_u32(seeded, 20, 0);  // seed 42 without seed_known
    reseal(seeded);
    expect_rejected(seeded, "data has a seed without the seed_known flag");
    expect_rejected(edited(52, 96), "data body size does not match the rotator parameters");
}

}  // namespace
