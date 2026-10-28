#pragma once

#include <cstdint>

namespace rabitqlib {
enum ScalarQuantizerType : std::uint8_t { RECONSTRUCTION, UNBIASED_ESTIMATION, PLAIN };
enum class RotatorType : uint8_t { MatrixRotator, FhtKacRotator };

namespace quant {
struct RabitqConfig {
    double t_const = -1;
    explicit RabitqConfig() = default;
};
}  // namespace quant
}  // namespace rabitqlib
