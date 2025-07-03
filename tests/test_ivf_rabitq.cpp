#include <gtest/gtest.h>

#include <rabitqlib/defines.hpp>
#include <rabitqlib/index/ivf/ivf.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/utils/io.hpp>
#include <rabitqlib/utils/stopw.hpp>
#include <contrib/dataset.h>

#include <memory>

using PID = rabitqlib::PID;
using ivf_index = rabitqlib::ivf::IVF;
using IvfIndexPtr = std::unique_ptr<ivf_index>;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;
using DatasetPtr = std::unique_ptr<rabitqlib::test::Dataset>;

static constexpr size_t n_clusters = 128;


DatasetPtr create_dataset(size_t num_points, size_t dim, size_t num_centroids, rabitqlib::test::Dataset::DatasetType type) {
    return std::make_unique<rabitqlib::test::Dataset>(num_points, dim, num_centroids, 42, type);
}

IvfIndexPtr create_ivf_index(rabitqlib::test::Dataset & dataset, size_t total_bits) {
    auto index = std::make_unique<rabitqlib::ivf::IVF>(dataset.get_num_points(), dataset.get_dim(), dataset.get_num_clusters(), total_bits);
    index->construct(dataset.get_data_ptr(), dataset.get_centroids_ptr(), dataset.get_cluster_ids_ptr(), false);
    return index;
}

float calculate_recall(std::vector<std::vector<PID>> & hacc_results, std::unordered_map<PID, std::set<PID>> & results) {
    float recall = 0;
    for (size_t i = 0; i < hacc_results.size(); i++) {
        size_t correct_count = 0;
        if (results.count(i) > 0) {
            for (auto & pid : hacc_results[i]) {
                if (results[i].count(pid) > 0) {
                    correct_count++;
                }
            }
            recall += static_cast<float>(correct_count) / static_cast<float>(results[i].size());
        }
        else {
            recall += 0;
        }
    }
    return recall / static_cast<float>(hacc_results.size());
}

void test_hacc_multi_bits(size_t num_points, size_t dim, size_t num_centroids, size_t total_bits, size_t topk = 100) {
    auto dataset = create_dataset(num_points, dim, num_centroids, rabitqlib::test::Dataset::DatasetType::Random);
    auto ivf_index = create_ivf_index(*dataset, total_bits);
    EXPECT_EQ(ivf_index->num_clusters(), dataset->get_num_clusters());
    EXPECT_EQ(ivf_index->padded_dim(), dataset->get_dim());

    auto results = dataset->get_results(rabitqlib::METRIC_L2, topk);

    auto queries = dataset->get_queries_ptr();
    auto hacc_results = std::vector<std::vector<PID>>(dataset->get_num_queries(), std::vector<PID>(topk));
    auto no_hacc_results = std::vector<std::vector<PID>>(dataset->get_num_queries(), std::vector<PID>(topk));
    for (size_t i = 0; i < dataset->get_num_queries(); i++) {
        ivf_index->search(queries + i * dim, topk, dataset->get_num_clusters() / 2, hacc_results[i].data(), true);
        ivf_index->search(queries + i * dim, topk, dataset->get_num_clusters() / 2, no_hacc_results[i].data(), false);
    }
    auto hacc_recall = calculate_recall(hacc_results, results);
    auto no_hacc_recall = calculate_recall(no_hacc_results, results);
    EXPECT_GT(hacc_recall, 0.9);
    EXPECT_GT(no_hacc_recall, 0.9);
}

TEST(IvfIndexTest, lut_hacc_ex_1bits) {
    test_hacc_multi_bits(5000, 128, n_clusters, 2);
}

TEST(IvfIndexTest, lut_hacc_ex_2bits) {
    test_hacc_multi_bits(5000, 128, n_clusters, 3);
}

TEST(IvfIndexTest, lut_hacc_ex_3bits) {
    test_hacc_multi_bits(5000, 128, n_clusters, 4);
}

TEST(IvfIndexTest, lut_hacc_ex_4bits) {
    test_hacc_multi_bits(5000, 128, n_clusters, 5);
}

TEST(IvfIndexTest, lut_hacc_ex_5bits) {
    test_hacc_multi_bits(8000, 128, n_clusters, 6);
}

TEST(IvfIndexTest, lut_hacc_ex_6bits) {
    test_hacc_multi_bits(10000, 128, n_clusters, 7);
}

TEST(IvfIndexTest, lut_hacc_ex_7bits) {
    test_hacc_multi_bits(20000, 128, n_clusters, 8);
}

TEST(IvfIndexTest, lut_hacc_ex_8bits) {
    test_hacc_multi_bits(100000, 128, n_clusters, 9);
}
