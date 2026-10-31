#pragma once

#include <cstdint>
#include <stdexcept>

namespace rabitqlib {
enum MetricType : uint8_t { METRIC_L2, METRIC_IP };

inline void validate_metric_type(MetricType metric_type) {
    if (metric_type != METRIC_L2 && metric_type != METRIC_IP) {
        throw std::invalid_argument("Metric type must be METRIC_L2 or METRIC_IP");
    }
}

}  // namespace rabitqlib
