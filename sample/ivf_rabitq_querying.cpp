#include <iostream>
#include <vector>

#include "defines.hpp"
#include "index/ivf/ivf.hpp"
#include "utils/io.hpp"
#include "utils/stopw.hpp"
#include "utils/tools.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::ivf::IVF;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

static std::vector<size_t> get_nprobes(
    const index_type& ivf,
    const std::vector<size_t>& all_nprobes,
    data_type& query,
    gt_type& gt,
    bool use_hacc
);

static size_t topk = 100;
static size_t test_round = 3;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for index \n"
                  << "arg2: path for query file, format .fvecs\n"
                  << "arg3: path for groundtruth file format .ivecs\n"
                  << "arg4: whether use high accuracy fastscan, (\"true\" or \"false\"), "
                     "true by default\n\n";
        exit(1);
    }

    char* index_file = argv[1];
    char* query_file = argv[2];
    char* gt_file = argv[3];
    bool use_hacc = true;

    if (argc > 4) {
        std::string hacc_str(argv[4]);
        if (hacc_str == "false") {
            use_hacc = false;
            std::cout << "Do not use Hacc FastScan\n";
        }
    }

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    size_t nq = query.rows();
    size_t total_count = nq * topk;

    index_type ivf;
    ivf.load(index_file);

    std::cout << "Load index done" << std::endl;
    std::vector<size_t> all_nprobes;
    all_nprobes.push_back(5);
    for (size_t i = 10; i < 200; i += 10) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 200; i < 400; i += 40) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 400; i <= 1500; i += 100) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 2000; i <= 4000; i += 500) {
        all_nprobes.push_back(i);
    }

    all_nprobes.push_back(6000);
    all_nprobes.push_back(10000);
    all_nprobes.push_back(15000);
    std::cout << "all_nprobes size: " << all_nprobes.size() << std::endl;
    rabitqlib::StopW stopw;

    auto nprobes = get_nprobes(ivf, all_nprobes, query, gt, use_hacc);
    size_t length = nprobes.size();

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(length));
    std::cout << "all_qps size: " << all_qps.size() << std::endl;
    std::cout << "all_recall size: " << all_recall.size() << std::endl;
    for (size_t r = 0; r < test_round; r++) {
        for (size_t l = 0; l < length; ++l) {
            size_t nprobe = nprobes[l];
            std::cout << "Current nprobe: " << nprobe << std::endl;
            if (nprobe > ivf.num_clusters()) {
                std::cout << "nprobe " << nprobe << " is larger than number of clusters, ";
                std::cout << "will use nprobe = num_cluster (" << ivf.num_clusters() << ").\n";
            }
            size_t total_correct = 0;
            float total_time = 0;
            std::vector<PID> results(topk);
            for (size_t i = 0; i < nq; i++) {
                stopw.reset();
                ivf.search(&query(i, 0), topk, nprobe, results.data(), use_hacc);
                total_time += stopw.get_elapsed_micro();
                for (size_t j = 0; j < topk; j++) {
                    for (size_t k = 0; k < topk; k++) {
                        if (gt(i, k) == results[j]) {
                            total_correct++;
                            break;
                        }
                    }
                }
            }
            float qps = static_cast<float>(nq) / (total_time / 1e6F);
            float recall =
                static_cast<float>(total_correct) / static_cast<float>(total_count);

            all_qps[r][l] = qps;
            all_recall[r][l] = recall;
        }
    }

    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);

    std::cout << "nprobe\tQPS\trecall" << '\n';

    for (size_t i = 0; i < length; ++i) {
        size_t nprobe = nprobes[i];
        float qps = avg_qps[i];
        float recall = avg_recall[i];

        std::cout << nprobe << '\t' << qps << '\t' << recall << '\n';
    }

    return 0;
}

static std::vector<size_t> get_nprobes(
    const index_type& ivf,
    const std::vector<size_t>& all_nprobes,
    data_type& query,
    gt_type& gt,
    bool use_hacc
) {
    size_t nq = query.rows();
    size_t total_count = topk * nq;
    float old_recall = 0;
    std::vector<size_t> nprobes;

    for (auto nprobe : all_nprobes) {
        nprobes.push_back(nprobe);

        size_t total_correct = 0;
        std::vector<PID> results(topk);
        for (size_t i = 0; i < nq; i++) {
            ivf.search(&query(i, 0), topk, nprobe, results.data(), use_hacc);
            for (size_t j = 0; j < topk; j++) {
                for (size_t k = 0; k < topk; k++) {
                    if (gt(i, k) == results[j]) {
                        total_correct++;
                        break;
                    }
                }
            }
        }
        float recall = static_cast<float>(total_correct) / static_cast<float>(total_count);
        std::cout << "nprobe: " << nprobe << " recall: " << recall << " old_recall: " << old_recall << std::endl;
        if (recall > 0.997 || recall - old_recall < 1e-5) {
            break;
        }
        std::cout << recall << '\t' << nprobe << std::endl << std::flush;
        old_recall = recall;
    }

    return nprobes;
}