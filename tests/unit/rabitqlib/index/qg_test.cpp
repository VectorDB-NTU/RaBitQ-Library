#include "rabitqlib/index/symqg/qg.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace rabitqlib::symqg {
namespace {

TEST(QuantizedGraphConfigurationTest, RejectsDegreeNotAlignedForFastScan) {
    EXPECT_THROW(
        (QuantizedGraph<float>(64, 64, 16, METRIC_L2, RotatorType::MatrixRotator)),
        std::invalid_argument
    );
}

TEST(QuantizedGraphConfigurationTest, RejectsDegreeThatCannotExcludeSelf) {
    EXPECT_THROW(
        (QuantizedGraph<float>(32, 64, 32, METRIC_L2, RotatorType::MatrixRotator)),
        std::invalid_argument
    );
}

TEST(QuantizedGraphLifecycleTest, DestroysConcreteRotatorThroughBasePointer) {
    QuantizedGraph<float> graph(33, 64, 32, METRIC_L2, RotatorType::MatrixRotator);
    EXPECT_EQ(graph.num_vertices(), 33U);
}

}  // namespace
}  // namespace rabitqlib::symqg
