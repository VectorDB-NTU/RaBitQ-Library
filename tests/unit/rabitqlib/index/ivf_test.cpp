#include "rabitqlib/index/ivf/ivf.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace rabitqlib::ivf {
namespace {

TEST(IvfConfigurationTest, RejectsUnsupportedMetric) {
    EXPECT_THROW(
        (IVF(8, 64, 1, 1, static_cast<MetricType>(255), RotatorType::MatrixRotator)),
        std::invalid_argument
    );
}

}  // namespace
}  // namespace rabitqlib::ivf
