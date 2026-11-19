#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/symqg_index.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using data_type = rabitqlib::RowMajorArray<float>;

int run(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for data file, format .fvecs\n"
                  << "arg2: degree bound for symqg, must be a multiple of 32\n"
                  << "arg3: ef for indexing \n"
                  << "arg4: path for saving index\n"
                  << "arg5: metric type (\"l2\", \"ip\" or \"cosine\"), l2 by default\n"
                  << "arg6: nbits (32 or 0 for raw vectors, 4 or 8), 32 by default\n"
                  << "arg7: init (pipnn or random), pipnn by default\n"
                  << "arg8: threads, 0 or omitted for all (OMP_NUM_THREADS caps it)\n";
        return 1;
    }

    char* data_file = argv[1];
    size_t degree = atoi(argv[2]);
    size_t ef = atoi(argv[3]);
    char* index_file = argv[4];

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    const char* metric_name = "L2";
    const std::string metric_str = argc > 5 ? argv[5] : "l2";
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
    size_t nbits = argc > 6 ? static_cast<size_t>(atoi(argv[6])) : 32;
    if (nbits == 0) {  // the 0.8 spelling of raw vectors
        nbits = 32;
    }

    data_type data;

    rabitqlib::load_vecs<float, data_type>(data_file, data);

    const std::string init = argc > 7 ? argv[7] : "pipnn";
    if (init != "random" && init != "pipnn") {
        throw std::invalid_argument("Init must be random or pipnn");
    }
    const size_t threads = argc > 8 ? std::stoul(argv[8]) : 0;

    rabitqlib::SymqgOptions options;
    options.metric = metric_type;
    options.nbits = nbits;
    options.max_degree = degree;
    options.ef_construction = ef;
    options.init = init == "pipnn" ? rabitqlib::symqg::QGInitialization::PiPNN
                                   : rabitqlib::symqg::QGInitialization::Random;
    rabitqlib::SymqgIndex qg(data.cols(), options);

    rabitqlib::StopW stopw;
    qg.add(data.data(), data.rows(), threads);
    auto milisecs = stopw.get_elapsed_mili();

    std::cout << "Indexing time " << milisecs / 1000.F << " secs\n";

    qg.save(index_file);

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
