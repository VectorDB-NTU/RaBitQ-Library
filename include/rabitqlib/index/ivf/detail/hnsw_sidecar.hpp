#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>

#include "rabitqlib/third/hnswlib/hnswlib.h"
#include "rabitqlib/utils/path.hpp"

namespace rabitqlib::ivf::detail {

// The HNSW initializer stores its graph at the index path plus this suffix. Saving,
// loading and atomic replacement must all agree on it.
inline constexpr char kHnswSidecarSuffix[] = ".hnsw";

// hnswlib's native sidecar has no version tag. Check the existing writer's
// header and record lengths before its loader uses offsets or allocates links.
// As with the main index file, callers must not mutate the file during loading.
inline void validate_hnsw_sidecar_file(
    const std::string& path, size_t expected_count, size_t vector_bytes
) {
    std::ifstream input(io_impl::filesystem_path(path), std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open HNSW centroid sidecar");
    }
    const auto end = input.tellg();
    input.seekg(0);
    const auto read = [&](auto& value) {
        if (!input.read(reinterpret_cast<char*>(&value), sizeof(value))) {
            throw std::runtime_error("Truncated HNSW centroid sidecar");
        }
    };
    size_t level0_offset = 0, capacity = 0, count = 0, level0_record_bytes = 0;
    size_t label_offset = 0, data_offset = 0;
    int max_level = 0;
    hnswlib::tableint entry = 0;
    size_t max_m = 0, max_m0 = 0, m = 0, ef_construction = 0;
    double mult = 0;
    read(level0_offset);
    read(capacity);
    read(count);
    read(level0_record_bytes);
    read(label_offset);
    read(data_offset);
    read(max_level);
    read(entry);
    read(max_m);
    read(max_m0);
    read(m);
    read(mult);
    read(ef_construction);
    if (count != expected_count || count == 0 || capacity < count ||
        count > std::numeric_limits<hnswlib::tableint>::max()) {
        throw std::runtime_error("HNSW centroid count does not match IVF cluster count");
    }
    constexpr size_t kMaxM = 10000;  // hnswlib caps M at this value.
    if (m < 2 || m > kMaxM || max_m != m || max_m0 != 2 * m || !std::isfinite(mult) ||
        mult <= 0 || ef_construction < m) {
        throw std::runtime_error("Invalid HNSW centroid graph parameters");
    }
    const size_t upper_layer_links_bytes =
        sizeof(hnswlib::linklistsizeint) + max_m * sizeof(hnswlib::tableint);
    const size_t level0_links_bytes =
        sizeof(hnswlib::linklistsizeint) + max_m0 * sizeof(hnswlib::tableint);
    if (level0_offset != 0 || data_offset != level0_links_bytes ||
        data_offset > label_offset || label_offset > level0_record_bytes ||
        label_offset - data_offset != vector_bytes ||
        level0_record_bytes - label_offset != sizeof(hnswlib::labeltype)) {
        throw std::runtime_error("HNSW centroid layout does not match IVF dimension");
    }
    if (max_level < 0 || entry >= count) {
        throw std::runtime_error("Invalid HNSW centroid entry point or maximum level");
    }
    auto remaining = end - input.tellg();
    if (remaining < 0 || count > static_cast<uintmax_t>(remaining) / level0_record_bytes) {
        throw std::runtime_error("Truncated HNSW centroid base records");
    }
    const auto level0_records_bytes =
        static_cast<std::streamoff>(count * level0_record_bytes);
    input.seekg(level0_records_bytes, std::ios::cur);
    remaining -= level0_records_bytes;
    size_t highest_level = 0, entry_level = 0;
    for (size_t i = 0; i < count; ++i) {
        unsigned int link_bytes = 0;  // Native type written by hnswlib::saveIndex.
        if (remaining < static_cast<std::streamoff>(sizeof(link_bytes))) {
            throw std::runtime_error("Truncated HNSW centroid sidecar");
        }
        read(link_bytes);
        remaining -= static_cast<std::streamoff>(sizeof(link_bytes));
        if (link_bytes % upper_layer_links_bytes != 0 ||
            link_bytes > static_cast<uintmax_t>(remaining)) {
            throw std::runtime_error("Invalid HNSW centroid upper-link length");
        }
        const size_t levels = link_bytes / upper_layer_links_bytes;
        highest_level = std::max(highest_level, levels);
        if (i == entry) {
            entry_level = levels;
        }
        // Most nodes have no upper links. Keep the stream's read buffer intact
        // and track the validated byte budget instead of querying each position.
        if (link_bytes != 0) {
            input.seekg(static_cast<std::streamoff>(link_bytes), std::ios::cur);
        }
        remaining -= static_cast<std::streamoff>(link_bytes);
    }
    if (!input || remaining != 0 || highest_level != static_cast<size_t>(max_level) ||
        entry_level != highest_level) {
        throw std::runtime_error("Invalid HNSW centroid level structure");
    }
}

inline void validate_hnsw_centroid_graph(const hnswlib::HierarchicalNSW<float>& graph) {
    const size_t count = graph.cur_element_count;
    if (graph.enterpoint_node_ >= count || graph.maxlevel_ < 0 ||
        graph.element_levels_[graph.enterpoint_node_] != graph.maxlevel_ ||
        graph.num_deleted_ != 0) {
        throw std::runtime_error("Invalid HNSW centroid entry point or deletion state");
    }
    for (size_t i = 0; i < count; ++i) {
        if (graph.element_levels_[i] < 0 || graph.element_levels_[i] > graph.maxlevel_) {
            throw std::runtime_error("Invalid HNSW centroid node level");
        }
        for (int level = 0; level <= graph.element_levels_[i]; ++level) {
            const char* links =
                level == 0 ? graph.data_level0_memory_ + i * graph.size_data_per_element_ +
                                 graph.offsetLevel0_
                           : graph.linkLists_[i] + (static_cast<size_t>(level) - 1) *
                                                       graph.size_links_per_element_;
            uint16_t degree = 0;  // The high two bytes hold flags, not the degree.
            std::memcpy(&degree, links, sizeof(degree));
            if (degree > (level == 0 ? graph.maxM0_ : graph.maxM_)) {
                throw std::runtime_error("Invalid HNSW centroid neighbor count");
            }
            for (size_t j = 0; j < degree; ++j) {
                hnswlib::tableint neighbor = 0;
                std::memcpy(
                    &neighbor,
                    links + sizeof(hnswlib::linklistsizeint) + j * sizeof(neighbor),
                    sizeof(neighbor)
                );
                if (neighbor >= count || graph.element_levels_[neighbor] < level) {
                    throw std::runtime_error(
                        "Invalid HNSW centroid neighbor or target level"
                    );
                }
            }
        }
    }
}

}  // namespace rabitqlib::ivf::detail
