#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/format_error.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/types.hpp"
#include "temp_file.hpp"

namespace rabitqlib {
namespace {
std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

// padded_dim follows the padding rule of the writer that produced the fixture set.
struct HistoricalFixtureCase {
    std::string directory;  // subdirectory under RABITQ_TEST_FIXTURE_ROOT
    std::string name;
    size_t padded_dim;
    size_t count = 33;
    size_t ivf_nprobe = 1;
    ivf::InitializerType initializer = ivf::InitializerType::Flat;
};

void PrintTo(const HistoricalFixtureCase& value, std::ostream* output) {
    *output << value.directory << '/' << value.name;
}

class HistoricalPersistenceTest : public ::testing::TestWithParam<HistoricalFixtureCase> {};

TEST_P(HistoricalPersistenceTest, LoadsSearchesResavesAndPreservesStateAfterFailedLoad) {
    const std::filesystem::path directory =
        std::filesystem::path(RABITQ_TEST_FIXTURE_ROOT) / GetParam().directory;
    const auto name = GetParam().name;
    const auto expected_padded_dim = GetParam().padded_dim;
    const auto path = (directory / (name + ".index")).string();
    std::ifstream expected(directory / (name + ".expected"));
    ASSERT_TRUE(expected.is_open());
    size_t count = 0, dim = 0, nq = 0, k = 0;
    expected >> count >> dim >> nq >> k;
    ASSERT_EQ(count, GetParam().count);
    ASSERT_EQ(dim, 65U);
    ASSERT_EQ(nq, 3U);
    ASSERT_EQ(k, 5U);
    std::vector<float> queries(nq * dim), distances(nq * k), reference_distances(nq * k);
    std::vector<PID> ids(nq * k), reference_ids(nq * k);
    for (auto& value : queries) {
        expected >> value;
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        expected >> reference_ids[i] >> reference_distances[i];
    }
    ASSERT_TRUE(expected.good());
    const auto output =
        std::filesystem::temp_directory_path() /
        rabitq_test::process_unique("historical_" + GetParam().directory + "_" + name);
    const auto compare = [&] {
        EXPECT_EQ(ids, reference_ids);
        for (size_t i = 0; i < ids.size(); ++i) {
            EXPECT_NEAR(
                distances[i],
                reference_distances[i],
                2e-4F * std::max(1.0F, std::abs(reference_distances[i]))
            );
        }
    };
    const auto exercise = [&](auto& index, const auto& search) {
        const size_t removed = name.find("qg") == 0 ? 0 : 1;
        const auto check_counts = [&] {
            // Earlier layouts do not record the seed, and migrating them invents none.
            EXPECT_FALSE(index.seed().has_value());
            EXPECT_EQ(index.num_stored_points(), count);
            EXPECT_EQ(index.num_points(), count - removed);
            EXPECT_GE(index.max_elements(), index.num_stored_points());
        };
        index.load(path.c_str());
        check_counts();
        EXPECT_EQ(index.dimension(), dim);
        const auto metric = name.find("_ip") != std::string::npos ? METRIC_IP : METRIC_L2;
        EXPECT_EQ(index.metric_type(), metric);
        search();
        compare();
        index.save(output.string().c_str());
        // Every index saves format 2.
        EXPECT_EQ(read_bytes(output).substr(0, 8), "RABQINDX");
        index.load(output.string().c_str());
        check_counts();
        search();
        compare();
        // Reading format 2 back and saving again reproduces the file.
        const std::string saved = read_bytes(output);
        index.save(output.string().c_str());
        EXPECT_TRUE(read_bytes(output) == saved);
        std::filesystem::resize_file(output, std::filesystem::file_size(output) - 1);
        EXPECT_THROW(index.load(output.string().c_str()), FormatError);
        check_counts();
        search();
        compare();
        std::filesystem::remove(output);
    };
    if (name.find("ivf") == 0) {
        ivf::IVF index;
        exercise(index, [&] {
            EXPECT_EQ(index.padded_dim(), expected_padded_dim);
            EXPECT_EQ(index.initializer_type(), GetParam().initializer);
            index.search_batch(
                queries.data(), nq, k, GetParam().ivf_nprobe, ids.data(), distances.data()
            );
        });
    } else if (name.find("hnsw") == 0) {
        hnsw::HierarchicalNSW index;
        exercise(index, [&] {
            const auto results = index.search(queries.data(), nq, k, 64, 1);
            for (size_t i = 0; i < nq; ++i) {
                ASSERT_EQ(results[i].size(), k);
                for (size_t j = 0; j < k; ++j) {
                    ids[i * k + j] = results[i][j].second;
                    distances[i * k + j] = results[i][j].first;
                }
            }
        });
    } else {
        symqg::QuantizedGraph<float> index;
        exercise(index, [&] {
            EXPECT_EQ(index.padded_dim(), expected_padded_dim);
            // Earlier layouts do not record the construction settings either.
            EXPECT_FALSE(index.ef_construction().has_value());
            EXPECT_FALSE(index.initialization().has_value());
            index.set_ef(64);
            index.search_batch(queries.data(), nq, k, ids.data(), distances.data(), 1);
        });
    }
}

INSTANTIATE_TEST_SUITE_P(
    V052,
    HistoricalPersistenceTest,
    ::testing::Values(
        HistoricalFixtureCase{"v0.5.2", "ivf4_l2", 128},
        HistoricalFixtureCase{"v0.5.2", "ivf4_ip", 128},
        HistoricalFixtureCase{"v0.5.2", "ivf32_l2", 128},
        HistoricalFixtureCase{"v0.5.2", "ivf32_ip", 128},
        HistoricalFixtureCase{"v0.5.2", "hnsw4_l2", 128},
        HistoricalFixtureCase{"v0.5.2", "hnsw4_ip", 128},
        HistoricalFixtureCase{"v0.5.2", "qg0_l2", 128},
        HistoricalFixtureCase{"v0.5.2", "qg0_ip", 128},
        HistoricalFixtureCase{"v0.5.2", "qg4_l2", 128},
        HistoricalFixtureCase{"v0.5.2", "qg4_ip", 128},
        HistoricalFixtureCase{"v0.5.2", "qg8_l2", 128},
        HistoricalFixtureCase{"v0.5.2", "qg8_ip", 128}
    ),
    [](const auto& info) { return info.param.name; }
);

// Four uneven IVF clusters written before per-cluster storage; dim 65 pads to 96.
INSTANTIATE_TEST_SUITE_P(
    PreIvfPerClusterStorage,
    HistoricalPersistenceTest,
    ::testing::Values(
        HistoricalFixtureCase{"pre-ivf-per-cluster-storage", "ivf4_l2", 96, 130, 4},
        HistoricalFixtureCase{"pre-ivf-per-cluster-storage", "ivf32_l2", 96, 130, 4}
    ),
    [](const auto& info) { return info.param.name; }
);

// The last IVF layout before format 2: RABQIDX1 version 2 and the HNSW sidecar.
INSTANTIATE_TEST_SUITE_P(
    V083,
    HistoricalPersistenceTest,
    ::testing::Values(
        HistoricalFixtureCase{"v0.8.3", "ivf4_flat_l2", 96, 130, 4},
        HistoricalFixtureCase{
            "v0.8.3", "ivf7_flatrabitq_ip", 96, 130, 4, ivf::InitializerType::FlatRaBitQ},
        HistoricalFixtureCase{
            "v0.8.3", "ivf4_hnsw_l2", 96, 130, 4, ivf::InitializerType::HNSW},
        HistoricalFixtureCase{
            "v0.8.3", "ivf32_hnsw_ip", 96, 130, 4, ivf::InitializerType::HNSW},
        HistoricalFixtureCase{"v0.8.3", "ivf1_flat_matrix_l2", 96, 130, 4},
        HistoricalFixtureCase{"v0.8.3", "hnsw1_l2", 96, 130},
        HistoricalFixtureCase{"v0.8.3", "hnsw5_ip", 96, 130},
        HistoricalFixtureCase{"v0.8.3", "hnsw9_l2", 96, 130},
        HistoricalFixtureCase{"v0.8.3", "qg0_ip", 96, 130},
        HistoricalFixtureCase{"v0.8.3", "qg4_l2", 96, 130},
        HistoricalFixtureCase{"v0.8.3", "qg8_ip", 96, 130},
        HistoricalFixtureCase{"v0.8.3", "qg4_matrix_l2", 96, 130}
    ),
    [](const auto& info) { return info.param.name; }
);
// Damaged headers of the earlier layouts that pass the size checks raise FormatError
// rather than the errors of the code that would use them.
TEST(HistoricalPersistenceTest, MalformedEarlierHeadersRaiseFormatError) {
    namespace fs = std::filesystem;
    const fs::path path =
        fs::temp_directory_path() / rabitq_test::process_unique("historical_header");
    struct Case {
        const char* fixture;
        size_t keep_bytes;                                 // 0 keeps the whole file
        std::vector<std::pair<size_t, uint64_t>> patches;  // little-endian u64 fields
        const char* message;
    };
    const uint64_t huge = uint64_t{1} << 62;
    const Case cases[] = {
        // RABQIDX1 v2: the stream fails inside the fixed header.
        {"ivf4_flat_l2", 20, {}, "IVF index file is truncated"},
        // An FHT/Kac dimension below 64 at offset 36.
        {"ivf4_flat_l2",
         0,
         {{36, 32}},
         "Invalid IVF rotator dimensions: Unsupported dimension for FhtKacRotator"},
        // SymphonyQG quantized layout: dimension at 28, padded dimension at 36.
        {"qg4_l2",
         0,
         {{28, 33}},
         "QuantizedGraph file is invalid: Unsupported dimension for FhtKacRotator"},
        {"qg4_l2",
         0,
         {{28, huge}, {36, huge}},
         "QuantizedGraph file describes records too large to address"},
        // The headerless raw layout: dimension at 16, padded dimension at 24.
        {"qg0_ip",
         0,
         {{16, huge}, {24, huge}},
         "QuantizedGraph file describes records too large to address"},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.message);
        fs::copy_file(
            fs::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" /
                (std::string(c.fixture) + ".index"),
            path,
            fs::copy_options::overwrite_existing
        );
        if (c.keep_bytes != 0) {
            fs::resize_file(path, c.keep_bytes);
        }
        {
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            for (const auto& [offset, value] : c.patches) {
                file.seekp(static_cast<std::streamoff>(offset));
                for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                    file.put(static_cast<char>(value >> (8 * i)));
                }
            }
        }
        const std::string file = path.string();
        try {
            if (c.fixture[0] == 'i') {
                ivf::IVF index;
                index.load(file.c_str());
            } else {
                symqg::QuantizedGraph<float> index;
                index.load(file.c_str());
            }
            ADD_FAILURE() << "accepted";
        } catch (const FormatError& error) { EXPECT_STREQ(error.what(), c.message); }
    }
    fs::remove(path);
}

// Truncated files of the earlier layouts raise FormatError like format-2 files.
TEST(HistoricalPersistenceTest, TruncatedEarlierFilesRaiseFormatError) {
    namespace fs = std::filesystem;
    const fs::path path =
        fs::temp_directory_path() / rabitq_test::process_unique("historical_truncated");
    for (const char* name : {"ivf4_flat_l2", "hnsw1_l2", "qg4_l2"}) {
        SCOPED_TRACE(name);
        fs::copy_file(
            fs::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / (std::string(name) + ".index"),
            path,
            fs::copy_options::overwrite_existing
        );
        fs::resize_file(path, fs::file_size(path) / 2);
        const std::string file = path.string();
        if (name[0] == 'i') {
            ivf::IVF index;
            EXPECT_THROW(index.load(file.c_str()), FormatError);
        } else if (name[0] == 'h') {
            hnsw::HierarchicalNSW index;
            EXPECT_THROW(index.load(file.c_str()), FormatError);
        } else {
            symqg::QuantizedGraph<float> index;
            EXPECT_THROW(index.load(file.c_str()), FormatError);
        }
    }
    // A metric byte that is neither L2 nor IP, at offset 61 of an RABQIDX1 v2 file.
    fs::copy_file(
        fs::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / "ivf4_flat_l2.index",
        path,
        fs::copy_options::overwrite_existing
    );
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(61);
        file.put(static_cast<char>(2));
    }
    ivf::IVF bad_metric;
    try {
        bad_metric.load(path.string().c_str());
        ADD_FAILURE() << "an invalid metric was loaded";
    } catch (const FormatError& error) {
        EXPECT_STREQ(error.what(), "Invalid IVF index metadata");
    }
    // Metric byte 2, METRIC_COSINE in memory, never appears in these layouts.
    const struct {
        const char* name;
        std::streamoff offset;
        const char* message;
    } graph_metrics[] = {
        {"hnsw5_ip", 48, "HNSW index file is invalid or truncated"},
        {"qg8_ip", 49, "Metric type must be METRIC_L2 or METRIC_IP"},
        {"qg0_ip", 37, "Metric type must be METRIC_L2 or METRIC_IP"},
    };
    for (const auto& c : graph_metrics) {
        SCOPED_TRACE(c.name);
        fs::copy_file(
            fs::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" /
                (std::string(c.name) + ".index"),
            path,
            fs::copy_options::overwrite_existing
        );
        {
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            file.seekg(c.offset);
            ASSERT_EQ(file.get(), 1);  // the IP metric byte
            file.seekp(c.offset);
            file.put(static_cast<char>(2));
        }
        try {
            if (c.name[0] == 'h') {
                hnsw::HierarchicalNSW index;
                index.load(path.string().c_str());
            } else {
                symqg::QuantizedGraph<float> index;
                index.load(path.string().c_str());
            }
            ADD_FAILURE() << "an invalid metric was loaded";
        } catch (const FormatError& error) { EXPECT_STREQ(error.what(), c.message); }
    }
    fs::remove(path);
    ivf::IVF missing;
    try {
        missing.load((path.string() + ".missing").c_str());
        ADD_FAILURE() << "a missing file was loaded";
    } catch (const FormatError&) {
        ADD_FAILURE() << "a missing file is not a format error";
    } catch (const std::runtime_error&) {  // "Cannot open IVF index file"
    }
}

// The checks of the SymphonyQG layout before format 2, on a copy of its v0.8.3 file.
TEST(HistoricalPersistenceTest, RejectsMalformedEarlierGraphFilesWithoutChangingTarget) {
    namespace fs = std::filesystem;
    const fs::path fixture = fs::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / "qg4_l2.index";
    const fs::path path =
        fs::temp_directory_path() / rabitq_test::process_unique("historical_qg_malformed");
    symqg::QuantizedGraph<float> target(65, 64, 32, METRIC_IP);
    const auto load_patched = [&](std::streamoff offset, size_t value) {
        fs::copy_file(fixture, path, fs::copy_options::overwrite_existing);
        {
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            ASSERT_TRUE(file.is_open());
            file.seekp(offset);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
            ASSERT_TRUE(file.good());
        }
        target.load(path.string().c_str());
    };
    // After the magic and version: point count, degree, dimension, padded dimension.
    constexpr std::streamoff kPointCountOffset = sizeof(uint64_t) + sizeof(uint32_t);
    constexpr std::streamoff kPaddedDimensionOffset =
        kPointCountOffset + 3 * sizeof(size_t);
    try {
        load_patched(kPaddedDimensionOffset, 100);
        ADD_FAILURE() << "an invalid padded dimension was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "QuantizedGraph file has an invalid padded dimension");
    }
    EXPECT_THROW(
        load_patched(kPointCountOffset, std::numeric_limits<size_t>::max()), FormatError
    );
    EXPECT_EQ(target.num_vertices(), 65U);
    EXPECT_EQ(target.metric_type(), METRIC_IP);
    fs::remove(path);
}

// The checks of the HNSW-RaBitQ layout before format 2, on a copy of its v0.8.3 file.
TEST(HistoricalPersistenceTest, RejectsMalformedEarlierHnswFilesWithoutChangingTarget) {
    namespace fs = std::filesystem;
    const fs::path fixture =
        fs::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / "hnsw5_ip.index";
    const fs::path path = fs::temp_directory_path() /
                          rabitq_test::process_unique("historical_hnsw_malformed");
    hnsw::HierarchicalNSW target;
    target.load(fixture.string().c_str());
    const auto expect_rejected = [&](const auto& damage) {
        fs::copy_file(fixture, path, fs::copy_options::overwrite_existing);
        damage();
        try {
            target.load(path.string().c_str());
            ADD_FAILURE() << "a damaged file was accepted";
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "HNSW index file is invalid or truncated");
        }
        EXPECT_EQ(target.num_stored_points(), 130U);
    };
    // The capacity is the first field.
    expect_rejected([&] {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        const size_t capacity = 1;
        file.write(reinterpret_cast<const char*>(&capacity), sizeof(capacity));
    });
    expect_rejected([&] { fs::resize_file(path, fs::file_size(path) - 1); });
    fs::remove(path);
}

// Files before format 2 keep the HNSW routing graph in <index>.hnsw; loading reads it
// from there, through the same path conversion as the index file.
TEST(HistoricalPersistenceTest, LoadsTheHnswSidecarOfEarlierLayouts) {
    namespace fs = std::filesystem;
    const fs::path fixture =
        fs::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / "ivf4_hnsw_l2.index";
    const fs::path directory =
        fs::temp_directory_path() /
        fs::u8path(rabitq_test::process_unique(u8"historical_sidecar_\u6d4b\u8bd5"));
    fs::create_directories(directory);
    const fs::path index_path = directory / fs::u8path(u8"\u7d22\u5f15.index");
    const fs::path sidecar = fs::path(index_path.native() + fs::path(".hnsw").native());
    fs::copy_file(fixture, index_path);
    const std::string path = index_path.u8string();

    ivf::IVF index;
    try {
        index.load(path.c_str());
        ADD_FAILURE() << "an HNSW-routed file without its sidecar was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "Cannot open HNSW centroid sidecar");
    }

    // A graph of another dimension does not fit the index.
    {
        std::vector<float> centroids(4 * 128, 0.5F);
        ivf::HNSWInitializer other(128, 4);
        other.add_vectors(centroids.data(), 1);
        std::ofstream output(sidecar, std::ios::binary);
        other.write(output);
    }
    try {
        index.load(path.c_str());
        ADD_FAILURE() << "a mismatched sidecar was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "HNSW centroid layout does not match IVF dimension");
    }

    fs::copy_file(
        fs::path(fixture.native() + fs::path(".hnsw").native()),
        sidecar,
        fs::copy_options::overwrite_existing
    );
    ASSERT_NO_THROW(index.load(path.c_str()));
    EXPECT_EQ(index.initializer_type(), ivf::InitializerType::HNSW);
    EXPECT_EQ(index.num_points(), 129U);
    fs::remove_all(directory);
}

}  // namespace
}  // namespace rabitqlib
