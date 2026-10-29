#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#if defined(RABITQ_TEST_INDEX)
#include "rabitqlib/index/hnsw/hnsw.hpp"
#endif

namespace {
const bool kBeforeMain = [] {
    std::array<float, 64> query{};
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = static_cast<float>(i % 4 + 1);
    }
    std::array<float, 256> lut{};
    lut.fill(-1.0F);
    rabitqlib::fastscan::pack_lut(64, query.data(), lut.data());
    for (size_t i = 0; i < lut.size(); ++i) {
        // The first coordinate occupies the high bit of each four-bit code.
        float expected = 0;
        for (size_t bit = 0; bit < 4; ++bit) {
            if ((i % 16) & (size_t{1} << bit)) {
                expected += static_cast<float>(4 - bit);
            }
        }
        if (lut[i] != expected) {
            return false;
        }
    }
    std::array<uint16_t, 256> integer_lut{};
    // Equal nonzero bytes make the packed result independent of the HACC layout.
    integer_lut.fill(0x0101);
    std::array<uint8_t, 512> packed_lut{};
    packed_lut.fill(0xAB);
    std::array<uint8_t, 256> codes{};
    std::array<int32_t, 32> scores{};
    scores.fill(-1);
    rabitqlib::fastscan::transfer_lut_hacc(integer_lut.data(), 64, packed_lut.data());
    for (const auto value : packed_lut) {
        if (value != 1) {
            return false;
        }
    }
    rabitqlib::fastscan::accumulate_hacc(
        codes.data(), packed_lut.data(), scores.data(), 64
    );
    for (const auto score : scores) {
        // Sixteen four-coordinate tables, each contributing 0x0101.
        if (score != 16 * 0x0101) {
            return false;
        }
    }
#if defined(RABITQ_TEST_INDEX)
    rabitqlib::hnsw::HierarchicalNSW index(2, 64, 4, 2, 16, 42);
    const auto result = index.search(query.data(), 1, 1, 10, 1);
    if (result.size() != 1 || !result[0].empty()) {
        return false;
    }
#endif
    return true;
}();
}  // namespace
int main() { return kBeforeMain ? 0 : 1; }
