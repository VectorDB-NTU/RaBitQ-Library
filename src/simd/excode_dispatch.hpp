#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

using ExcodeIpTable = std::array<float (*)(const float*, const uint8_t*, size_t), 9>;

ExcodeIpTable resolve_excode_ip_table();

}  // namespace rabitqlib::simd
