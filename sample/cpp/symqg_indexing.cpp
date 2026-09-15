#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::symqg::QuantizedGraph<float>;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

int run(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for data file, format .fvecs\n"
                  << "arg2: degree bound for symqg, must be a multiple of 32\n"
                  << "arg3: ef for indexing \n"
                  << "arg4: path for saving index\n"
                  << "arg5: metric type (\"l2\" or \"ip\"), l2 by default\n"
                  << "arg6: vector quantization bits (0, 4, or 8), 0 by default\n"
                  << "arg7: init (pipnn or random), pipnn by default\n"
                  << "arg8: construction threads, all available by default\n";
        return 1;
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
    size_t quantization_bits = argc > 6 ? static_cast<size_t>(atoi(argv[6])) : 0;

    data_type data;

    rabitqlib::load_vecs<float, data_type>(data_file, data);

    const std::string init = argc > 7 ? argv[7] : "pipnn";
    if (init != "random" && init != "pipnn") {
        throw std::invalid_argument("Init must be random or pipnn");
    }
    const bool pipnn = init == "pipnn";
    const size_t threads = argc > 8 ? std::stoul(argv[8]) : rabitqlib::total_threads();
    if (threads == 0 || ef == 0 || ef > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(
            "Threads and construction ef must be positive; ef must fit uint32"
        );
    }
    rabitqlib::StopW stopw;
    rabitqlib::StopW stage;
    index_type qg(
        data.rows(),
        data.cols(),
        degree,
        metric_type,
        rabitqlib::RotatorType::FhtKacRotator,
        quantization_bits
    );

    {
        const auto initialization = pipnn ? rabitqlib::symqg::QGInitialization::PiPNN
                                          : rabitqlib::symqg::QGInitialization::Random;
        rabitqlib::symqg::QGBuilder builder(
            qg, static_cast<uint32_t>(ef), data.data(), threads, initialization
        );
        std::cout << "Initialize and encode " << stage.get_elapsed_sec() << " secs\n";

        // QG owns its vectors/codes now; release the input before refinement.
        data = data_type();
        stage.reset();
        builder.build();
        std::cout << (pipnn ? "One refinement " : "Three iterations ")
                  << stage.get_elapsed_sec() << " secs\n";
        std::cout << "Average degree " << builder.avg_degree() << '\n';
    }  // Release builder scratch before saving.

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
