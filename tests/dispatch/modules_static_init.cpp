#include <array>
#include <cstdint>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#if defined(RABITQ_TEST_INDEX)
#include "rabitqlib/index/hnsw/hnsw.hpp"
#endif

namespace {
const bool kBeforeMain = [] {
    std::array<float, 64> query{};
    std::array<float, 256> lut{};
    rabitqlib::fastscan::pack_lut(64, query.data(), lut.data());
    std::array<uint16_t, 256> integer_lut{};
    std::array<uint8_t, 512> packed_lut{};
    std::array<uint8_t, 256> codes{};
    std::array<int32_t, 32> scores{};
    rabitqlib::fastscan::transfer_lut_hacc(integer_lut.data(), 64, packed_lut.data());
    rabitqlib::fastscan::accumulate_hacc(
        codes.data(), packed_lut.data(), scores.data(), 64
    );
    for (const auto score : scores) {
        if (score != 0) {
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
