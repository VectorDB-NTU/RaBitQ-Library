#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"

namespace {
using namespace rabitqlib;
constexpr size_t kCount = 64, kDim = 64;
constexpr size_t kEntryOffset = 6 * sizeof(size_t) + sizeof(int);
constexpr size_t kMaxMOffset = kEntryOffset + sizeof(hnswlib::tableint);
constexpr size_t kHeaderBytes = kMaxMOffset + 4 * sizeof(size_t) + sizeof(double);

template <typename T>
T read_at(const std::vector<char>& bytes, size_t offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        throw std::runtime_error("invalid test fixture offset");
    }
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}
template <typename T>
void write_at(std::vector<char>& bytes, size_t offset, T value) {
    (void)read_at<T>(bytes, offset);
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

struct Fixture {
    std::string path;
    std::vector<float> data = std::vector<float>(kCount * kDim);
    std::vector<char> original;
    std::vector<size_t> upper_offsets, levels;
    size_t level0_record_bytes = 0, upper_layer_links_bytes = 0, max_m = 0, max_m0 = 0;
    size_t upper_node = 0, base_node = 0;

    explicit Fixture(MetricType metric) {
        path =
            (std::filesystem::temp_directory_path() /
             ("rabitq_corrupt_centroids_" + std::to_string(static_cast<int>(metric)) + "_" +
              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                .string();
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i % 113) / 113.0F;
        }
        std::vector<PID> ids(kCount);
        for (size_t i = 0; i < kCount; ++i) {
            ids[i] = static_cast<PID>(i);
        }
        ivf::IVF index(
            kCount,
            kDim,
            kCount,
            4,
            metric,
            RotatorType::FhtKacRotator,
            ivf::InitializerType::HNSW
        );
        index.construct(data.data(), data.data(), ids.data(), false, 1);
        index.save(path.c_str());
        std::ifstream file(path + ".hnsw", std::ios::binary);
        original.assign(std::istreambuf_iterator<char>(file), {});
        level0_record_bytes = read_at<size_t>(original, 3 * sizeof(size_t));
        max_m = read_at<size_t>(original, kMaxMOffset);
        max_m0 = read_at<size_t>(original, kMaxMOffset + sizeof(size_t));
        upper_layer_links_bytes =
            sizeof(hnswlib::linklistsizeint) + max_m * sizeof(hnswlib::tableint);
        size_t pos = kHeaderBytes + kCount * level0_record_bytes;
        for (size_t i = 0; i < kCount; ++i) {
            upper_offsets.push_back(pos);
            const auto bytes = read_at<unsigned int>(original, pos);
            levels.push_back(bytes / upper_layer_links_bytes);
            pos += sizeof(unsigned int) + bytes;
        }
        if (pos != original.size()) {
            throw std::runtime_error("bad fixture length");
        }
        auto upper =
            std::find_if(levels.begin(), levels.end(), [](size_t n) { return n > 0; });
        auto base = std::find(levels.begin(), levels.end(), 0);
        if (upper == levels.end() || base == levels.end()) {
            throw std::runtime_error("fixture needs two levels");
        }
        upper_node = static_cast<size_t>(upper - levels.begin());
        base_node = static_cast<size_t>(base - levels.begin());
    }
    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        std::filesystem::remove(path + ".hnsw", ignored);
    }
    void save(const std::vector<char>& bytes) const {
        std::ofstream file(path + ".hnsw", std::ios::binary | std::ios::trunc);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
};

TEST(IvfHnswSidecarGraphTest, PublicLoadRejectsCorruptStructureAndCanRetry) {
    for (const auto metric : {METRIC_L2, METRIC_IP}) {
        Fixture fixture(metric);
        const auto reject = [&](const char* name, auto corrupt, const char* message) {
            SCOPED_TRACE(name);
            auto bytes = fixture.original;
            corrupt(bytes);
            fixture.save(bytes);
            ivf::IVF loaded;
            try {
                loaded.load(fixture.path.c_str());
                FAIL() << "damaged sidecar was accepted";
            } catch (const std::runtime_error& error) {
                EXPECT_STREQ(error.what(), message);
            }
            fixture.save(fixture.original);
            ASSERT_NO_THROW(loaded.load(fixture.path.c_str()));
            PID id = 0;
            float distance = 0;
            ASSERT_NO_THROW(loaded.search(fixture.data.data(), 1, 1, &id, &distance));
        };
        const size_t upper =
            fixture.upper_offsets[fixture.upper_node] + sizeof(unsigned int);
        reject(
            "count mismatch",
            [](auto& b) { write_at<size_t>(b, 2 * sizeof(size_t), kCount - 1); },
            "HNSW centroid count does not match IVF cluster count"
        );
        reject(
            "capacity below count",
            [](auto& b) { write_at<size_t>(b, sizeof(size_t), kCount - 1); },
            "HNSW centroid count does not match IVF cluster count"
        );
        reject(
            "invalid level multiplier",
            [](auto& b) {
                write_at(
                    b,
                    kMaxMOffset + 3 * sizeof(size_t),
                    std::numeric_limits<double>::quiet_NaN()
                );
            },
            "Invalid HNSW centroid graph parameters"
        );
        reject(
            "invalid construction breadth",
            [](auto& b) {
                write_at<size_t>(b, kMaxMOffset + 3 * sizeof(size_t) + sizeof(double), 0);
            },
            "Invalid HNSW centroid graph parameters"
        );
        reject(
            "truncated base records",
            [&](auto& b) {
                b.resize(kHeaderBytes + kCount * fixture.level0_record_bytes - 1);
            },
            "Truncated HNSW centroid base records"
        );
        reject(
            "entry out of range",
            [](auto& b) {
                write_at(b, kEntryOffset, static_cast<hnswlib::tableint>(kCount));
            },
            "Invalid HNSW centroid entry point or maximum level"
        );
        reject(
            "negative level",
            [](auto& b) { write_at(b, 6 * sizeof(size_t), -1); },
            "Invalid HNSW centroid entry point or maximum level"
        );
        reject(
            "wrong maximum level",
            [](auto& b) {
                write_at(b, 6 * sizeof(size_t), read_at<int>(b, 6 * sizeof(size_t)) + 1);
            },
            "Invalid HNSW centroid level structure"
        );
        reject(
            "entry below maximum level",
            [&](auto& b) {
                write_at(
                    b, kEntryOffset, static_cast<hnswlib::tableint>(fixture.base_node)
                );
            },
            "Invalid HNSW centroid level structure"
        );
        reject(
            "base neighbor",
            [](auto& b) {
                write_at<uint16_t>(b, kHeaderBytes, 1);
                write_at(
                    b,
                    kHeaderBytes + sizeof(hnswlib::linklistsizeint),
                    static_cast<hnswlib::tableint>(kCount)
                );
            },
            "Invalid HNSW centroid neighbor or target level"
        );
        reject(
            "base degree",
            [&](auto& b) {
                write_at(b, kHeaderBytes, static_cast<uint16_t>(fixture.max_m0 + 1));
            },
            "Invalid HNSW centroid neighbor count"
        );
        reject(
            "upper neighbor",
            [&](auto& b) {
                write_at<uint16_t>(b, upper, 1);
                write_at(
                    b,
                    upper + sizeof(hnswlib::linklistsizeint),
                    static_cast<hnswlib::tableint>(kCount)
                );
            },
            "Invalid HNSW centroid neighbor or target level"
        );
        reject(
            "upper degree",
            [&](auto& b) { write_at(b, upper, static_cast<uint16_t>(fixture.max_m + 1)); },
            "Invalid HNSW centroid neighbor count"
        );
        reject(
            "neighbor missing upper layer",
            [&](auto& b) {
                write_at<uint16_t>(b, upper, 1);
                write_at(
                    b,
                    upper + sizeof(hnswlib::linklistsizeint),
                    static_cast<hnswlib::tableint>(fixture.base_node)
                );
            },
            "Invalid HNSW centroid neighbor or target level"
        );
        reject(
            "partial upper layer",
            [&](auto& b) {
                const auto offset = fixture.upper_offsets[fixture.upper_node];
                write_at(b, offset, read_at<unsigned int>(b, offset) - 1);
            },
            "Invalid HNSW centroid upper-link length"
        );
        reject(
            "truncated upper length",
            [&](auto& b) { b.resize(fixture.upper_offsets.back() + 2); },
            "Truncated HNSW centroid sidecar"
        );
        reject(
            "truncated upper links",
            [&](auto& b) {
                const auto offset = fixture.upper_offsets[fixture.upper_node];
                b.resize(upper + read_at<unsigned int>(b, offset) - 1);
            },
            "Invalid HNSW centroid upper-link length"
        );
        reject(
            "invalid graph stride",
            [](auto& b) { write_at<size_t>(b, kMaxMOffset, 0); },
            "Invalid HNSW centroid graph parameters"
        );
        reject(
            "invalid label offset",
            [](auto& b) {
                write_at(b, 4 * sizeof(size_t), std::numeric_limits<size_t>::max());
            },
            "HNSW centroid layout does not match IVF dimension"
        );
        reject(
            "deleted centroid",
            [](auto& b) { b[kHeaderBytes + 2] = 1; },
            "Invalid HNSW centroid entry point or deletion state"
        );
        reject(
            "truncated header",
            [](auto& b) { b.resize(4); },
            "Truncated HNSW centroid sidecar"
        );
        reject(
            "trailing byte",
            [](auto& b) { b.push_back(0); },
            "Invalid HNSW centroid level structure"
        );
    }
}

TEST(IvfHnswSidecarGraphTest, DisconnectedGraphCannotReadEmptyResults) {
    for (const auto metric : {METRIC_L2, METRIC_IP}) {
        Fixture fixture(metric);
        auto bytes = fixture.original;
        for (size_t i = 0; i < kCount; ++i) {
            write_at<uint16_t>(bytes, kHeaderBytes + i * fixture.level0_record_bytes, 0);
            for (size_t level = 0; level < fixture.levels[i]; ++level) {
                write_at<uint16_t>(
                    bytes,
                    fixture.upper_offsets[i] + sizeof(unsigned int) +
                        level * fixture.upper_layer_links_bytes,
                    0
                );
            }
        }
        fixture.save(bytes);
        ivf::IVF loaded;
        ASSERT_NO_THROW(loaded.load(fixture.path.c_str()));
        PID id = 0;
        float distance = 0;
        try {
            loaded.search(fixture.data.data(), 1, kCount, &id, &distance);
            FAIL() << "insufficient candidates must be rejected";
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "HNSW centroid graph returned too few candidates");
        }
    }
}
}  // namespace
