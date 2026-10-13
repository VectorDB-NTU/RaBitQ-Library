#pragma once

#include <cstdint>

namespace rabitqlib {
enum ScalarQuantizerType : std::uint8_t { RECONSTRUCTION, UNBIASED_ESTIMATION, PLAIN };
enum class RotatorType : uint8_t { MatrixRotator, FhtKacRotator };

namespace quant {
struct RabitqConfig {
    double t_const = -1;
    explicit RabitqConfig() = default;
    RabitqConfig(RabitqConfig const&) = default;
    RabitqConfig(RabitqConfig&&) = default;
    RabitqConfig& operator=(const RabitqConfig&) = default;
    RabitqConfig& operator=(RabitqConfig&&) = default;
};
}  // namespace quant
}  // namespace rabitqlib
