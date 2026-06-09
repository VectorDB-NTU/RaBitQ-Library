#include <iostream>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::symqg::QuantizedGraph<float>;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

std::vector<size_t> efs = {
    10, 20, 40, 50, 60, 80, 100, 150, 170, 190, 200, 250, 300, 400, 500, 600, 700, 800, 1500
};
size_t test_round = 3;
size_t topk = 10;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3>\n"
                  << "arg1: path for index \n"
                  << "arg2: path for query file, format .fvecs\n"
                  << "arg3: path for groundtruth file format .ivecs\n";
        exit(1);
    }

    char* index_file = argv[1];
    char* query_file = argv[2];
    char* gt_file = argv[3];

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    size_t nq = query.rows();
    size_t total_count = nq * topk;

    index_type qg;
    qg.load(index_file);

    rabitqlib::StopW stopw;

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(efs.size()));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(efs.size()));

    for (size_t r = 0; r < test_round; r++) {
        for (size_t i = 0; i < efs.size(); ++i) {
            size_t ef = efs[i];
            size_t total_correct = 0;
            float total_time = 0;
            qg.set_ef(ef);
            std::vector<PID> results(topk);
            for (size_t z = 0; z < nq; z++) {
                stopw.reset();
                qg.search(&query(z, 0), topk, results.data());
                total_time += stopw.get_elapsed_micro();
                for (size_t y = 0; y < topk; y++) {
                    for (size_t k = 0; k < topk; k++) {
                        if (gt(z, k) == results[y]) {
                            total_correct++;
                            break;
                        }
                    }
                }
            }
            float qps = static_cast<float>(nq) / (total_time / 1e6F);
            float recall =
                static_cast<float>(total_correct) / static_cast<float>(total_count);

            all_qps[r][i] = qps;
            all_recall[r][i] = recall;
        }
    }

    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);

    std::cout << "EF\tQPS\tRecall\n";
    for (size_t i = 0; i < avg_qps.size(); ++i) {
        std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\n';
    }

    return 0;
}