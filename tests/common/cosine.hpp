#ifndef RABITQ_TEST_COSINE_HPP
#define RABITQ_TEST_COSINE_HPP

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "format2_file.hpp"

namespace rabitq_test {

// The normalization the cosine metric specifies: the squared norm summed in double in
// coordinate order, then each coordinate times 1 / sqrt of it, rounded to float. A zero
// row stays zero.
inline std::vector<float> normalized_rows(const std::vector<float>& rows, size_t dim) {
    std::vector<float> normalized(rows.size());
    for (size_t begin = 0; begin < rows.size(); begin += dim) {
        double squared_norm = 0;
        for (size_t i = begin; i < begin + dim; ++i) {
            squared_norm += static_cast<double>(rows[i]) * rows[i];
        }
        const double scale = squared_norm > 0 ? 1.0 / std::sqrt(squared_norm) : 0.0;
        for (size_t i = begin; i < begin + dim; ++i) {
            normalized[i] = static_cast<float>(rows[i] * scale);
        }
    }
    return normalized;
}

// Scale row i by 1 + i % 7, so that the rows' norms differ.
inline std::vector<float> rescaled_rows(std::vector<float> rows, size_t dim) {
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] *= static_cast<float>(1 + ((i / dim) % 7));
    }
    return rows;
}

// A cosine index stores what an inner-product index of the normalized vectors stores:
// the two files differ only in the header's metric field.
inline void expect_same_sections_except_metric(
    const std::string& cosine_path,
    const std::string& ip_path,
    format2::IndexKind kind,
    const std::vector<uint32_t>& known_kinds
) {
    const auto cosine = read_format2_file(cosine_path, kind, known_kinds);
    const auto ip = read_format2_file(ip_path, kind, known_kinds);
    EXPECT_EQ(cosine.header.metric, format2::Metric::Cosine);
    EXPECT_EQ(ip.header.metric, format2::Metric::InnerProduct);
    ASSERT_EQ(cosine.sections.size(), ip.sections.size());
    for (size_t i = 0; i < cosine.sections.size(); ++i) {
        EXPECT_EQ(cosine.sections[i].first.kind, ip.sections[i].first.kind);
        EXPECT_EQ(cosine.sections[i].second, ip.sections[i].second)
            << "section kind " << cosine.sections[i].first.kind;
    }
}

}  // namespace rabitq_test

#endif  // RABITQ_TEST_COSINE_HPP
