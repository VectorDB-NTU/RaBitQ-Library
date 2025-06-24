#include "rabitq.h"
#include "../rabitqlib/quantization/rabitq.hpp"
#include "../rabitqlib/utils/rotator.hpp"

// 在 C++ 文件中，我们知道 Rotator 是 rabitqlib::Rotator<float> 的别名
// 但这个细节对 C 和 Rust 是隐藏的

extern "C" {

RabitqConfig* rabitq_config_new() {
    return reinterpret_cast<RabitqConfig*>(new rabitqlib::quant::RabitqConfig());
}

void rabitq_config_free(RabitqConfig* config) {
    delete reinterpret_cast<rabitqlib::quant::RabitqConfig*>(config);
}

Rotator* rabitq_rotator_new(size_t dim, size_t padded_dim) {
    auto rotator = new rabitqlib::rotator_impl::FhtKacRotator(dim, padded_dim);
    return reinterpret_cast<Rotator*>(rotator);
}

void rabitq_rotator_free(Rotator* rotator) {
    delete reinterpret_cast<rabitqlib::Rotator<float>*>(rotator);
}

void rabitq_rotator_rotate(const Rotator* rotator, const float* x, float* y) {
    reinterpret_cast<const rabitqlib::Rotator<float>*>(rotator)->rotate(x, y);
}

}