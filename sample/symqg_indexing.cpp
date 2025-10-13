#include <iostream>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::symqg::QuantizedGraph<float>;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for data file, format .fvecs\n"
                  << "arg2: degree bound for symqg, must be a multiple of 32\n"
                  << "arg3: ef for indexing \n"
                  << "arg4: path for saving index\n"
                  << "arg5: metric type (\"l2\" or \"ip\"), l2 by default\n";
        exit(1);
    }

    char* data_file = argv[1];
    size_t degree = atoi(argv[2]);
    size_t ef = atoi(argv[3]);
    char* index_file = argv[4];

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    if (argc > 5) {
        std::string metric_str(argv[5]);
        if (metric_str == "ip" || metric_str == "IP") {
            metric_type = rabitqlib::METRIC_IP;
        }
    }
    if (metric_type == rabitqlib::METRIC_IP) {
        std::cout << "Metric Type: IP\n";
    } else if (metric_type == rabitqlib::METRIC_L2) {
        std::cout << "Metric Type: L2\n";
    }

    data_type data;

    rabitqlib::load_vecs<float, data_type>(data_file, data);

    rabitqlib::StopW stopw;

    index_type qg(data.rows(), data.cols(), degree, metric_type);

    rabitqlib::symqg::QGBuilder builder(qg, ef, data.data());

    // 3 iters, refine at last iter
    builder.build();

    auto milisecs = stopw.get_elapsed_mili();

    std::cout << "Indexing time " << milisecs / 1000.F << " secs\n";

    qg.save(index_file);

    return 0;
}