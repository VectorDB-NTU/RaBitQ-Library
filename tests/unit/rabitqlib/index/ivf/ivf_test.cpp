#include "rabitqlib/index/ivf/ivf.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using rabitqlib::PID;
using rabitqlib::ivf::IVF;

constexpr size_t kNumPoints = 64;
constexpr size_t kDimension = 64;
constexpr size_t kNumClusters = 4;
constexpr size_t kTopK = 5;

struct TestData {
    std::vector<float> vectors = std::vector<float>(kNumPoints * kDimension);
    std::vector<float> centroids = std::vector<float>(kNumClusters * kDimension);
    std::vector<PID> cluster_ids = std::vector<PID>(kNumPoints);

    TestData() {
        for (size_t point = 0; point < kNumPoints; ++point) {
            cluster_ids[point] = static_cast<PID>(point % kNumClusters);
            for (size_t dim = 0; dim < kDimension; ++dim) {
                vectors[(point * kDimension) + dim] =
                    std::sin(static_cast<float>((point + 1) * (dim + 3)) * 0.013F);
            }
        }
        for (size_t cluster = 0; cluster < kNumClusters; ++cluster) {
            size_t cluster_size = 0;
            for (size_t point = cluster; point < kNumPoints; point += kNumClusters) {
                ++cluster_size;
                for (size_t dim = 0; dim < kDimension; ++dim) {
                    centroids[(cluster * kDimension) + dim] +=
                        vectors[(point * kDimension) + dim];
                }
            }
            for (size_t dim = 0; dim < kDimension; ++dim) {
                centroids[(cluster * kDimension) + dim] /= static_cast<float>(cluster_size);
            }
        }
    }
};

struct Results {
    std::array<PID, kTopK> ids{};
    std::array<float, kTopK> distances{};
};

Results search(const IVF& index, const TestData& data) {
    Results results;
    index.search(
        data.vectors.data(),
        kTopK,
        kNumClusters,
        results.ids.data(),
        results.distances.data(),
        true
    );
    return results;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open IVF test file");
    }
    return std::vector<char>(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()
    );
}

class IvfArrayStorageTest : public ::testing::Test {
   protected:
    std::string path(const std::string& filename) {
        paths_.push_back(filename);
        return filename;
    }

    void TearDown() override {
        for (const std::string& filename : paths_) {
            std::remove(filename.c_str());
        }
    }

   private:
    std::vector<std::string> paths_;
};

TEST_F(IvfArrayStorageTest, RepeatedConstructionReplacesOwnedStorage) {
    const TestData data;
    IVF index(kNumPoints, kDimension, kNumClusters, 4);

    index.construct(
        data.vectors.data(), data.centroids.data(), data.cluster_ids.data(), false, 1
    );
    const Results first = search(index, data);

    index.construct(
        data.vectors.data(), data.centroids.data(), data.cluster_ids.data(), false, 1
    );
    const Results second = search(index, data);

    EXPECT_EQ(second.ids, first.ids);
    EXPECT_EQ(second.distances, first.distances);
}

TEST_F(IvfArrayStorageTest, OneBitStorageReloadsAndResavesExactly) {
    const TestData data;
    const std::string original_path = path("ivf_array_one_bit.index");
    const std::string resaved_path = path("ivf_array_one_bit_resaved.index");

    IVF index(kNumPoints, kDimension, kNumClusters, 1);
    index.construct(
        data.vectors.data(), data.centroids.data(), data.cluster_ids.data(), false, 1
    );
    const Results before = search(index, data);
    index.save(original_path.c_str());

    index.load(original_path.c_str());
    index.load(original_path.c_str());
    const Results after = search(index, data);
    index.save(resaved_path.c_str());

    EXPECT_EQ(after.ids, before.ids);
    EXPECT_EQ(after.distances, before.distances);
    EXPECT_EQ(read_file(resaved_path), read_file(original_path));
}

TEST_F(IvfArrayStorageTest, FailedLoadPreservesExistingStorage) {
    const TestData data;
    const std::string complete_path = path("ivf_array_complete.index");
    const std::string truncated_path = path("ivf_array_truncated.index");

    IVF index(kNumPoints, kDimension, kNumClusters, 4);
    index.construct(
        data.vectors.data(), data.centroids.data(), data.cluster_ids.data(), false, 1
    );
    const Results before = search(index, data);
    index.save(complete_path.c_str());

    const std::vector<char> complete_file = read_file(complete_path);
    {
        std::ofstream output(truncated_path, std::ios::binary);
        output.write(
            complete_file.data(), static_cast<std::streamsize>(complete_file.size() / 2)
        );
    }
    EXPECT_THROW(index.load(truncated_path.c_str()), std::ios_base::failure);
    const Results after = search(index, data);

    EXPECT_EQ(after.ids, before.ids);
    EXPECT_EQ(after.distances, before.distances);
}

}  // namespace
