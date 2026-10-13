#include "rabitqlib/index/hnsw/hnsw.hpp"

int hnsw_header_smoke() {
    rabitqlib::hnsw::HierarchicalNSW empty;
    rabitqlib::hnsw::HierarchicalNSW index(8, 64, 4, 2, 10, 42);
    float data[8 * 64]{};
    float centroid[64]{};
    rabitqlib::PID assignments[8]{};
    index.construct(1, centroid, 8, data, assignments);
    return empty.num_points() == 0 && empty.num_stored_points() == 0 &&
                   index.dimension() == 64 && index.num_points() == 8 &&
                   index.num_stored_points() == 8 && index.max_elements() == 8
               ? 0
               : 1;
}
