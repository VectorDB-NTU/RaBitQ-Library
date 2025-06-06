#include <cstdint>
#include <iostream>

#include "defines.hpp"
#include "index/ivf/ivf.hpp"
#include "utils/io.hpp"
#include "utils/stopw.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::ivf::IVF;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4> <arg5>\n"
                  << "arg1: path for data file, format .fvecs\n"
                  << "arg2: path for centroids file generated by ivf.py\n"
                  << "arg3: path for cluster ids file generated by ivf.py\n"
                  << "arg4: total number of bits for quantization\n"
                  << "arg5: path for saving index\n"
                  << "arg6: if use faster quantization (\"true\" or \"false\"), false by "
                     "default\n";
        exit(1);
    }

    bool faster_quant = false;
    if (argc > 6) {
        std::string faster_str(argv[6]);
        if (faster_str == "true") {
            faster_quant = true;
            std::cout << "Using faster quantize for indexing...\n";
        }
    }

    char* data_file = argv[1];
    char* centroids_file = argv[2];
    char* cids_file = argv[3];
    size_t total_bits = atoi(argv[4]);
    char* index_file = argv[5];

    data_type data;
    data_type centroids;
    gt_type cids;

    rabitqlib::load_vecs<float, data_type>(data_file, data);
    rabitqlib::load_vecs<float, data_type>(centroids_file, centroids);
    rabitqlib::load_vecs<PID, gt_type>(cids_file, cids);

    size_t num_points = data.rows();
    size_t dim = data.cols();
    size_t k = centroids.rows();

    std::cout << "data loaded\n";
    std::cout << "\tN: " << num_points << '\n';
    std::cout << "\tDIM: " << dim << '\n';

    rabitqlib::StopW stopw;
    index_type ivf(num_points, dim, k, total_bits);
    ivf.construct(data.data(), centroids.data(), cids.data(), faster_quant);
    float miniutes = stopw.get_elapsed_mili() / 1000 / 60;
    std::cout << "ivf constructed \n";
    ivf.save(index_file);

    std::cout << "Indexing time " << miniutes << '\n';

    return 0;
}