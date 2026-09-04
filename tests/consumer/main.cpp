#include "rabitqlib/utils/cpu_features.hpp"

int main() {
    const auto& features = rabitqlib::cpu::features();
    return features.avx512vpopcntdq && !features.avx512f ? 1 : 0;
}
