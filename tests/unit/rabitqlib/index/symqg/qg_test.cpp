#include "rabitqlib/index/symqg/qg.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using rabitqlib::kPidMax;
using rabitqlib::METRIC_L2;
using rabitqlib::MetricType;
using rabitqlib::PID;
using rabitqlib::QGBatchDataMap;
using rabitqlib::RotatorType;
using rabitqlib::buffer::kSearchBufferMaxPointCount;
using rabitqlib::symqg::QuantizedGraph;

struct GraphFileOptions {
    size_t num_points = 2;
    size_t degree = 32;
    size_t dim = 64;
    size_t padded_dim = 64;
    PID entry_point = 0;
    RotatorType rotator_type = RotatorType::FhtKacRotator;
    MetricType metric_type = METRIC_L2;
    bool metadata_only = false;
    bool invalid_neighbor = false;
    size_t invalid_neighbor_point = 0;
    bool trailing_byte = false;
};

void write_graph_file(const std::string& path, const GraphFileOptions& options) {
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("failed to create test graph");
    }

    output.write(
        reinterpret_cast<const char*>(&options.num_points), sizeof(options.num_points)
    );
    output.write(reinterpret_cast<const char*>(&options.degree), sizeof(options.degree));
    output.write(reinterpret_cast<const char*>(&options.dim), sizeof(options.dim));
    output.write(
        reinterpret_cast<const char*>(&options.padded_dim), sizeof(options.padded_dim)
    );
    output.write(
        reinterpret_cast<const char*>(&options.entry_point), sizeof(options.entry_point)
    );
    output.write(
        reinterpret_cast<const char*>(&options.rotator_type), sizeof(options.rotator_type)
    );
    output.write(
        reinterpret_cast<const char*>(&options.metric_type), sizeof(options.metric_type)
    );
    if (options.metadata_only) {
        return;
    }

    const size_t batch_data_offset = options.dim * sizeof(float);
    const size_t neighbor_offset =
        batch_data_offset +
        (QGBatchDataMap<float>::data_bytes(options.padded_dim) * (options.degree / 32));
    const size_t row_offset = neighbor_offset + (options.degree * sizeof(PID));
    std::vector<char> data(options.num_points * row_offset, 0);
    for (size_t point = 0; point < options.num_points; ++point) {
        for (size_t neighbor = 0; neighbor < options.degree; ++neighbor) {
            const PID neighbor_id = options.invalid_neighbor &&
                                            point == options.invalid_neighbor_point &&
                                            neighbor == 0
                                        ? kPidMax
                                        : static_cast<PID>(neighbor % options.num_points);
            std::memcpy(
                data.data() + (point * row_offset) + neighbor_offset +
                    (neighbor * sizeof(PID)),
                &neighbor_id,
                sizeof(neighbor_id)
            );
        }
    }
    output.write(data.data(), static_cast<std::streamsize>(data.size()));

    const size_t rotator_bytes = options.padded_dim / 2;
    const std::vector<char> rotator(rotator_bytes, 0);
    output.write(rotator.data(), static_cast<std::streamsize>(rotator.size()));
    if (options.trailing_byte) {
        output.put('\0');
    }
}

class QuantizedGraphFileTest : public ::testing::Test {
   protected:
    std::string path(const std::string& name) {
        paths_.push_back(name);
        return name;
    }

    void TearDown() override {
        for (const std::string& file : paths_) {
            std::remove(file.c_str());
        }
    }

   private:
    std::vector<std::string> paths_;
};

TEST(QuantizedGraphStorage, AcceptsAValidContiguousLayout) {
    QuantizedGraph<float> graph(2, 64, 32);
    EXPECT_EQ(graph.num_vertices(), 2);
    EXPECT_EQ(graph.dimension(), 64);
    EXPECT_EQ(graph.degree_bound(), 32);
}

TEST(QuantizedGraphStorage, RejectsInvalidDimensionsAndDegree) {
    EXPECT_THROW((QuantizedGraph<float>(1, 0, 32)), std::invalid_argument);
    EXPECT_THROW((QuantizedGraph<float>(1, 64, 0)), std::invalid_argument);
    EXPECT_THROW((QuantizedGraph<float>(1, 64, 31)), std::invalid_argument);
}

TEST(QuantizedGraphStorage, RejectsStorageLayoutOverflow) {
    const size_t maximum = std::numeric_limits<size_t>::max();
    const size_t overflowing_degree = maximum - (maximum % 32);
    EXPECT_THROW((QuantizedGraph<float>(1, 64, overflowing_degree)), std::length_error);
}

TEST_F(QuantizedGraphFileTest, LoadsAValidFileWithCheckedNeighbors) {
    const std::string file = path("symqg_valid_test.index");
    write_graph_file(file, GraphFileOptions{});

    QuantizedGraph<float> graph;
    graph.load(file.c_str());
    EXPECT_EQ(graph.num_vertices(), 2);
    EXPECT_EQ(graph.dimension(), 64);
    EXPECT_EQ(graph.degree_bound(), 32);
    EXPECT_EQ(graph.entry_point(), 0);
}

TEST_F(QuantizedGraphFileTest, SuccessfulReloadPreservesEfForSearch) {
    const std::string file = path("symqg_reload_test.index");
    write_graph_file(file, GraphFileOptions{});

    QuantizedGraph<float> graph(2, 64, 32);
    graph.set_ef(1);
    graph.load(file.c_str());

    const std::vector<float> query(64, 0.0F);
    PID result = kPidMax;
    graph.search(query.data(), 1, &result);
    EXPECT_LT(result, 2U);
}

TEST_F(QuantizedGraphFileTest, ValidatesNeighborsInTheLastRow) {
    const std::string file = path("symqg_bad_last_neighbor_test.index");
    GraphFileOptions options;
    options.num_points = 3;
    options.invalid_neighbor = true;
    options.invalid_neighbor_point = options.num_points - 1;
    write_graph_file(file, options);

    QuantizedGraph<float> graph;
    EXPECT_THROW(graph.load(file.c_str()), std::runtime_error);
}

TEST_F(QuantizedGraphFileTest, RejectsSizeMismatchBeforeAllocating) {
    const std::string truncated = path("symqg_truncated_test.index");
    GraphFileOptions truncated_options;
    truncated_options.num_points = 1000000000;
    truncated_options.metadata_only = true;
    write_graph_file(truncated, truncated_options);

    QuantizedGraph<float> graph;
    EXPECT_THROW(graph.load(truncated.c_str()), std::ios_base::failure);

    const std::string trailing = path("symqg_trailing_test.index");
    GraphFileOptions trailing_options;
    trailing_options.trailing_byte = true;
    write_graph_file(trailing, trailing_options);
    EXPECT_THROW(graph.load(trailing.c_str()), std::ios_base::failure);
}

TEST_F(QuantizedGraphFileTest, RejectsInvalidMetadata) {
    QuantizedGraph<float> graph;

    const std::string padded = path("symqg_bad_padded_dim_test.index");
    GraphFileOptions padded_options;
    padded_options.padded_dim = 128;
    padded_options.metadata_only = true;
    write_graph_file(padded, padded_options);
    EXPECT_THROW(graph.load(padded.c_str()), std::runtime_error);

    const std::string entry = path("symqg_bad_entry_test.index");
    GraphFileOptions entry_options;
    entry_options.entry_point = 2;
    entry_options.metadata_only = true;
    write_graph_file(entry, entry_options);
    EXPECT_THROW(graph.load(entry.c_str()), std::invalid_argument);

    const std::string metric = path("symqg_bad_metric_test.index");
    GraphFileOptions metric_options;
    metric_options.metric_type = static_cast<MetricType>(255);
    metric_options.metadata_only = true;
    write_graph_file(metric, metric_options);
    EXPECT_THROW(graph.load(metric.c_str()), std::invalid_argument);

    const std::string rotator = path("symqg_bad_rotator_test.index");
    GraphFileOptions rotator_options;
    rotator_options.rotator_type = static_cast<RotatorType>(255);
    rotator_options.metadata_only = true;
    write_graph_file(rotator, rotator_options);
    EXPECT_THROW(graph.load(rotator.c_str()), std::invalid_argument);
}

TEST_F(QuantizedGraphFileTest, EnforcesSearchBufferPointCountLimitBeforeAllocation) {
    const std::string at_limit = path("symqg_point_limit_test.index");
    GraphFileOptions at_limit_options;
    at_limit_options.num_points = kSearchBufferMaxPointCount;
    at_limit_options.metadata_only = true;
    write_graph_file(at_limit, at_limit_options);

    QuantizedGraph<float> graph;
    EXPECT_THROW(graph.load(at_limit.c_str()), std::ios_base::failure);

    if constexpr (std::numeric_limits<size_t>::max() > kSearchBufferMaxPointCount) {
        const std::string over_limit = path("symqg_over_point_limit_test.index");
        GraphFileOptions over_limit_options;
        over_limit_options.num_points = kSearchBufferMaxPointCount + 1;
        over_limit_options.metadata_only = true;
        write_graph_file(over_limit, over_limit_options);
        EXPECT_THROW(graph.load(over_limit.c_str()), std::invalid_argument);
    }
}

TEST_F(QuantizedGraphFileTest, FailedLoadPreservesExistingGraph) {
    QuantizedGraph<float> graph(2, 64, 32);
    graph.set_ep(1);

    const std::string invalid = path("symqg_bad_neighbor_test.index");
    GraphFileOptions options;
    options.invalid_neighbor = true;
    write_graph_file(invalid, options);

    EXPECT_THROW(graph.load(invalid.c_str()), std::runtime_error);
    EXPECT_EQ(graph.num_vertices(), 2);
    EXPECT_EQ(graph.dimension(), 64);
    EXPECT_EQ(graph.degree_bound(), 32);
    EXPECT_EQ(graph.entry_point(), 1);

    EXPECT_THROW(
        graph.load("symqg_file_that_does_not_exist.index"), std::ios_base::failure
    );
    EXPECT_EQ(graph.entry_point(), 1);
}

}  // namespace
