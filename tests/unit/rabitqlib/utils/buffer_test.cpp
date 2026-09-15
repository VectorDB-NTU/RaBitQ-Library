#include "rabitqlib/utils/buffer.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "rabitqlib/defines.hpp"

namespace rabitqlib::buffer {
namespace {

TEST(SearchBufferTest, RejectsCapacityWhoseSentinelWouldOverflow) {
    EXPECT_THROW(
        (SearchBuffer<float>(std::numeric_limits<size_t>::max())), std::length_error
    );

    SearchBuffer<float> buffer;
    EXPECT_TRUE(buffer.is_full());
    EXPECT_FALSE(buffer.has_next());
    EXPECT_EQ(buffer.size(), 0U);
    buffer.insert(7, 1.0F);
    EXPECT_EQ(buffer.size(), 0U);
    EXPECT_THROW(buffer.resize(std::numeric_limits<size_t>::max()), std::length_error);

    SearchBuffer<float> populated(2);
    populated.insert(7, 1.0F);
    EXPECT_THROW(populated.resize(std::numeric_limits<size_t>::max()), std::length_error);
    EXPECT_EQ(populated.size(), 1U);
    PID result = 0;
    float distance = 0;
    populated.copy_results(&result, &distance);
    EXPECT_EQ(result, 7U);
    EXPECT_FLOAT_EQ(distance, 1.0F);
}

}  // namespace
}  // namespace rabitqlib::buffer
