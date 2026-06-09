#include <cstdint>
#include <iostream>

#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::hnsw::HierarchicalNSW;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <arg1> <arg2> <arg3> <arg4> <arg5> <arg6> <arg7> <arg8>\n"
                  << "arg1: path for data file, format .fvecs\n"
                  << "arg2: path for centroids file, format .fvecs\n"
                  << "arg3: path for cluster_ids file, format .ivecs\n"
                  << "arg4: m (degree bound) for hnsw\n"
                  << "arg5: ef for indexing \n"
                  << "arg6: total number of bits for quantization\n"
                  << "arg7: path for saving index\n"
                  << "arg8: metric type (\"l2\" or \"ip\"), l2 by default\n"
                  << "arg9: if use faster quantization (\"true\" or \"false\"), false by "
                     "default\n";
        exit(1);
    }

    char* data_file = argv[1];
    char* centroid_file = argv[2];
    char* cid_file = argv[3];
    size_t m = atoi(argv[4]);
    size_t ef = atoi(argv[5]);
    size_t total_bits = atoi(argv[6]);
    char* index_file = argv[7];

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    if (argc > 8) {
        std::string metric_str(argv[8]);
        if (metric_str == "ip" || metric_str == "IP") {
            metric_type = rabitqlib::METRIC_IP;
        }
    }
    if (metric_type == rabitqlib::METRIC_IP) {
        std::cout << "Metric Type: IP\n";
    } else if (metric_type == rabitqlib::METRIC_L2) {
        std::cout << "Metric Type: L2\n";
    }

    bool faster_quant = false;
    if (argc > 9) {
        std::string faster_str(argv[9]);
        if (faster_str == "true") {
            faster_quant = true;
            std::cout << "Using faster quantize for indexing...\n";
        }
    }

    data_type data;
    data_type centroids;
    gt_type cluster_id;

    rabitqlib::load_vecs<float, data_type>(data_file, data);
    rabitqlib::load_vecs<float, data_type>(centroid_file, centroids);
    rabitqlib::load_vecs<uint32_t, gt_type>(cid_file, cluster_id);

    size_t num_points = data.rows();
    size_t dim = data.cols();

    size_t random_seed = 100;  // by default 100
    auto* hnsw = new rabitqlib::hnsw::HierarchicalNSW(
        num_points, dim, total_bits, m, ef, random_seed, metric_type
    );

    rabitqlib::StopW stopw;
    stopw.reset();

    hnsw->construct(
        centroids.rows(),
        centroids.data(),
        num_points,
        data.data(),
        cluster_id.data(),
        0,
        faster_quant
    );

    float total_time = stopw.get_elapsed_micro();
    total_time /= 1e6;

    std::cout << "indexing time = " << total_time << "s" << '\n';
    hnsw->save(index_file);

    std::cout << "index saved..." << '\n';

    return 0;
}