#include <type_traits>

#include "rabitqlib/index/ivf/ivf.hpp"

static_assert(std::is_nothrow_default_constructible_v<rabitqlib::ivf::IVF>);

int ivf_header_smoke() {
    rabitqlib::ivf::IVF empty;
    rabitqlib::ivf::IVF index(
        8,
        64,
        2,
        4,
        rabitqlib::METRIC_L2,
        rabitqlib::RotatorType::FhtKacRotator,
        rabitqlib::ivf::InitializerType::Flat
    );
    float data[8 * 64]{};
    float centroids[2 * 64]{};
    rabitqlib::PID assignments[8]{};
    index.construct(data, centroids, assignments);
    return empty.max_elements() == 0 && empty.num_points() == 0 &&
                   empty.num_stored_points() == 0 && index.dimension() == 64 &&
                   index.num_points() == 8 && index.num_stored_points() == 8
               ? 0
               : 1;
}
