#include <utility>

#include "rabitqlib/index/symqg/qg.hpp"

int symqg_header_smoke() {
    rabitqlib::symqg::QuantizedGraph empty;
    rabitqlib::symqg::QuantizedGraph graph(
        64, 65, 32, rabitqlib::METRIC_L2, rabitqlib::RotatorType::FhtKacRotator, 4, 42
    );
    auto moved = std::move(graph);
    empty = std::move(moved);
    return empty.dimension() == 65 && empty.padded_dim() == 96 &&
                   empty.max_elements() == 64 && empty.num_points() == 0 &&
                   empty.num_stored_points() == 0 && empty.quantization_bits() == 4
               ? 0
               : 1;
}
