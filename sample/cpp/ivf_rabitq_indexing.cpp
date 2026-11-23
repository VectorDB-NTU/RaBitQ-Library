#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/ivf_index.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using data_type = rabitqlib::RowMajorArray<float>;
using cluster_ids_type = rabitqlib::RowMajorArray<uint32_t>;

int run(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4> <arg5>\n"
                  << "arg1: path for data file, format .fvecs\n"
                  << "arg2: path for centroids file, format .fvecs\n"
                  << "arg3: path for cluster ids file, format .ivecs\n"
                  << "arg4: total number of bits for quantization\n"
                  << "arg5: path for saving index\n"
                  << "arg6: metric type (\"l2\", \"ip\" or \"cosine\"), l2 by default\n"
                  << "arg7: if use faster quantization (\"true\" or \"false\"), false by "
                     "default\n";
        return 1;
    }

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    const char* metric_name = "L2";
    const std::string metric_str = argc > 6 ? argv[6] : "l2";
    if (metric_str == "ip" || metric_str == "IP") {
        metric_type = rabitqlib::METRIC_IP;
        metric_name = "IP";
    } else if (metric_str == "cosine" || metric_str == "COSINE") {
        metric_type = rabitqlib::METRIC_COSINE;
        metric_name = "Cosine";
    } else if (metric_str != "l2" && metric_str != "L2") {
        throw std::invalid_argument("Metric type must be l2, ip or cosine");
    }
    std::cout << "Metric Type: " << metric_name << '\n';

    bool faster_quant = false;
    if (argc > 7) {
        std::string faster_str(argv[7]);
        if (faster_str == "true") {
            faster_quant = true;
            std::cout << "Using faster quantize for indexing...\n";
        }
    }

    char* data_file = argv[1];
    char* centroids_file = argv[2];
    char* cluster_ids_file = argv[3];
    size_t total_bits = atoi(argv[4]);
    char* index_file = argv[5];

    data_type data;
    data_type centroids;
    cluster_ids_type cluster_ids;

    rabitqlib::load_vecs<float, data_type>(data_file, data);
    rabitqlib::load_vecs<float, data_type>(centroids_file, centroids);
    rabitqlib::load_vecs<PID, cluster_ids_type>(cluster_ids_file, cluster_ids);

    size_t num_points = data.rows();
    size_t dim = data.cols();
    size_t num_clusters = centroids.rows();

    std::cout << "data loaded\n";
    std::cout << "\tN: " << num_points << '\n';
    std::cout << "\tDIM: " << dim << '\n';

    rabitqlib::StopW stopw;
    rabitqlib::IvfOptions options;
    options.metric = metric_type;
    options.nbits = total_bits;
    rabitqlib::IvfIndex ivf(dim, num_clusters, options);
    ivf.set_centroids(centroids.data());
    rabitqlib::AddOptions add_options;
    add_options.cluster_ids = cluster_ids.data();
    add_options.fast_quantization = faster_quant;
    ivf.add(data.data(), num_points, add_options);
    float miniutes = stopw.get_elapsed_mili() / 1000 / 60;
    std::cout << "ivf constructed \n";
    ivf.save(index_file);

    std::cout << "Indexing time " << miniutes << '\n';

    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
