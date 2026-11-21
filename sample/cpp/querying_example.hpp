#pragma once

#include <cstddef>
#include <vector>

namespace querying_example {

// The average of each column over the rounds. The querying samples store one row per
// round, at least one, with one value per search setting, so every row has the same
// length.
inline std::vector<float> column_averages(const std::vector<std::vector<float>>& rows) {
    std::vector<float> averages(rows.front().size(), 0.0F);
    for (const auto& row : rows) {
        for (size_t column = 0; column < averages.size(); ++column) {
            averages[column] += row[column];
        }
    }
    for (auto& average : averages) {
        average /= static_cast<float>(rows.size());
    }
    return averages;
}

}  // namespace querying_example
