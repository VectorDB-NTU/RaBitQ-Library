#include "rabitqlib/utils/io.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

#include "rabitqlib/defines.hpp"

namespace rabitqlib {
namespace {

template <typename T>
void write_value(std::ofstream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

TEST(VectorIoTest, LoadsCompleteConsistentRecords) {
    const std::string path = ::testing::TempDir() + "rabitq_valid.fvecs";
    {
        std::ofstream output(path, std::ios::binary);
        const uint32_t cols = 2;
        const float values[] = {1.0F, 2.0F, 3.0F, 4.0F};
        write_value(output, cols);
        output.write(reinterpret_cast<const char*>(values), 2 * sizeof(float));
        write_value(output, cols);
        output.write(reinterpret_cast<const char*>(values + 2), 2 * sizeof(float));
    }

    RowMajorMatrix<float> matrix;
    load_vecs<float>(path.c_str(), matrix);
    ASSERT_EQ(matrix.rows(), 2);
    ASSERT_EQ(matrix.cols(), 2);
    EXPECT_FLOAT_EQ(matrix(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(matrix(1, 1), 4.0F);
    std::remove(path.c_str());
}

TEST(BinaryMatrixIoTest, LoadsUtf8PathsAndChecksTheirSize) {
    const auto native_path =
        std::filesystem::temp_directory_path() /
        std::filesystem::u8path(u8"rabitq_\u6d4b\u8bd5_\U0001f680.bin");
    {
        std::ofstream output(native_path, std::ios::binary);
        write_value(output, uint32_t{1});
        write_value(output, uint32_t{1});
        write_value(output, 3.0F);
    }
    const auto path = native_path.u8string();
    EXPECT_TRUE(file_exists(path.c_str()));
    EXPECT_EQ(get_filesize(path.c_str()), 12U);
    RowMajorMatrix<float> matrix;
    load_bin<float>(path.c_str(), matrix);
    ASSERT_EQ(matrix.size(), 1);
    EXPECT_FLOAT_EQ(matrix(0, 0), 3.0F);
    std::filesystem::remove(native_path);
}

#if !defined(_WIN32)
TEST(FilesystemPathTest, PreservesNativePosixBytes) {
    const std::string filename = "rabitq_\xff.bin";
    EXPECT_EQ(io_impl::filesystem_path(filename).native(), filename);
}
#endif

#if defined(__linux__)
TEST(BinaryMatrixIoTest, LoadsNonUtf8NativePaths) {
    const auto native_path = std::filesystem::temp_directory_path() / "rabitq_\xff.bin";
    {
        std::ofstream output(native_path, std::ios::binary);
        ASSERT_TRUE(output.is_open());
        write_value(output, uint32_t{1});
        write_value(output, uint32_t{1});
        write_value(output, 3.0F);
    }
    const auto path = native_path.native();
    EXPECT_TRUE(file_exists(path.c_str()));
    EXPECT_EQ(get_filesize(path.c_str()), 12U);
    RowMajorMatrix<float> matrix;
    load_bin<float>(path.c_str(), matrix);
    ASSERT_EQ(matrix.size(), 1);
    EXPECT_FLOAT_EQ(matrix(0, 0), 3.0F);
    std::filesystem::remove(native_path);
}
#endif

TEST(VectorIoTest, RejectsInconsistentAndTruncatedRecordsWithoutReplacingOutput) {
    const std::string path = ::testing::TempDir() + "rabitq_invalid.fvecs";
    RowMajorMatrix<float> matrix(1, 1);
    matrix(0, 0) = 7.0F;

    {
        std::ofstream output(path, std::ios::binary);
        const uint32_t cols = 2;
        const uint32_t wrong_cols = 3;
        const float values[] = {1.0F, 2.0F, 3.0F, 4.0F};
        write_value(output, cols);
        output.write(reinterpret_cast<const char*>(values), 2 * sizeof(float));
        write_value(output, wrong_cols);
        output.write(reinterpret_cast<const char*>(values + 2), 2 * sizeof(float));
    }
    EXPECT_THROW(load_vecs<float>(path.c_str(), matrix), std::runtime_error);
    ASSERT_EQ(matrix.rows(), 1);
    EXPECT_FLOAT_EQ(matrix(0, 0), 7.0F);

    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const uint32_t cols = 2;
        const float value = 1.0F;
        write_value(output, cols);
        write_value(output, value);
    }
    EXPECT_THROW(load_vecs<float>(path.c_str(), matrix), std::runtime_error);
    ASSERT_EQ(matrix.rows(), 1);
    EXPECT_FLOAT_EQ(matrix(0, 0), 7.0F);
    std::remove(path.c_str());
}

TEST(BinaryMatrixIoTest, ValidatesPayloadBeforeReplacingOutput) {
    const std::string path = ::testing::TempDir() + "rabitq_matrix.bin";
    RowMajorMatrix<float> matrix(1, 1);
    matrix(0, 0) = 7.0F;

    {
        std::ofstream output(path, std::ios::binary);
        const uint32_t rows = 2;
        const uint32_t cols = 2;
        const float values[] = {1.0F, 2.0F, 3.0F};
        write_value(output, rows);
        write_value(output, cols);
        output.write(reinterpret_cast<const char*>(values), sizeof(values));
    }
    EXPECT_THROW(load_bin<float>(path.c_str(), matrix), std::runtime_error);
    ASSERT_EQ(matrix.rows(), 1);
    EXPECT_FLOAT_EQ(matrix(0, 0), 7.0F);

    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const uint32_t rows = 2;
        const uint32_t cols = 2;
        const float values[] = {1.0F, 2.0F, 3.0F, 4.0F};
        write_value(output, rows);
        write_value(output, cols);
        output.write(reinterpret_cast<const char*>(values), sizeof(values));
    }
    load_bin<float>(path.c_str(), matrix);
    ASSERT_EQ(matrix.rows(), 2);
    ASSERT_EQ(matrix.cols(), 2);
    EXPECT_FLOAT_EQ(matrix(1, 1), 4.0F);
    std::remove(path.c_str());
}

}  // namespace
}  // namespace rabitqlib
