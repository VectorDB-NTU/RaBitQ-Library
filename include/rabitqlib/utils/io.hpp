#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace rabitqlib {
namespace io_impl {

inline size_t checked_add(size_t lhs, size_t rhs) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        throw std::length_error("File layout exceeds size_t");
    }
    return lhs + rhs;
}

inline size_t checked_multiply(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::length_error("File layout exceeds size_t");
    }
    return lhs * rhs;
}

inline void read_exact(
    std::ifstream& input, void* destination, size_t bytes, const char* description
) {
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("File field is too large to read");
    }
    input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error(
            std::string("Unexpected end of file while reading ") + description
        );
    }
}

template <typename T>
void read_value(std::ifstream& input, T& value, const char* description) {
    read_exact(input, &value, sizeof(value), description);
}

}  // namespace io_impl

// get num of bytes
inline size_t get_filesize(const char* filename) {
    const auto file_size = std::filesystem::file_size(filename);
    if (file_size > std::numeric_limits<size_t>::max()) {
        throw std::length_error("File is too large to address");
    }
    return static_cast<size_t>(file_size);
}

inline bool file_exists(const char* filename) { return std::filesystem::exists(filename); }

// load .*vecs file to a matrix (e.g., RowMajorFloatMat)
template <typename T, class M>
void load_vecs(const char* filename, M& row_mat) {
    if (!file_exists(filename)) {
        throw std::runtime_error("File does not exist: " + std::string(filename));
    }

    static_assert(std::is_same_v<T*, std::decay_t<decltype(row_mat.data())>>);

    const size_t file_size = get_filesize(filename);
    if (file_size < sizeof(uint32_t)) {
        throw std::runtime_error("Vector file is too small to contain a dimension");
    }
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open vector file: " + std::string(filename));
    }

    uint32_t stored_cols = 0;
    io_impl::read_value(input, stored_cols, "vector dimension");
    if (stored_cols == 0) {
        throw std::runtime_error("Vector dimension must be positive");
    }

    const size_t cols = stored_cols;
    const size_t record_bytes =
        io_impl::checked_add(sizeof(uint32_t), io_impl::checked_multiply(cols, sizeof(T)));
    if (file_size % record_bytes != 0) {
        throw std::runtime_error("Vector file size is not a whole number of records");
    }
    const size_t rows = file_size / record_bytes;
    M loaded(rows, cols);

    input.seekg(0, std::ifstream::beg);

    for (size_t i = 0; i < rows; i++) {
        uint32_t row_cols = 0;
        io_impl::read_value(input, row_cols, "row dimension");
        if (row_cols != stored_cols) {
            throw std::runtime_error("Vector file contains inconsistent row dimensions");
        }
        io_impl::read_exact(
            input, &loaded(i, 0), io_impl::checked_multiply(sizeof(T), cols), "vector data"
        );
    }

    row_mat = std::move(loaded);
}

// load .*bin file to a matrix (e.g., RowMajorFloatMat)
template <typename T, class M>
void load_bin(const char* filename, M& row_mat) {
    if (!file_exists(filename)) {
        throw std::runtime_error("File does not exist: " + std::string(filename));
    }

    static_assert(std::is_same_v<T*, std::decay_t<decltype(row_mat.data())>>);

    const size_t file_size = get_filesize(filename);
    if (file_size < 2 * sizeof(uint32_t)) {
        throw std::runtime_error("Binary matrix file is too small to contain its header");
    }
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error(
            "Cannot open binary matrix file: " + std::string(filename)
        );
    }

    uint32_t stored_rows = 0;
    uint32_t stored_cols = 0;
    io_impl::read_value(input, stored_rows, "matrix row count");
    io_impl::read_value(input, stored_cols, "matrix column count");
    if (stored_rows == 0 || stored_cols == 0) {
        throw std::runtime_error("Binary matrix dimensions must be positive");
    }

    const size_t rows = stored_rows;
    const size_t cols = stored_cols;
    const size_t payload_bytes =
        io_impl::checked_multiply(io_impl::checked_multiply(rows, cols), sizeof(T));
    const size_t expected_file_size =
        io_impl::checked_add(2 * sizeof(uint32_t), payload_bytes);
    if (file_size != expected_file_size) {
        throw std::runtime_error("Binary matrix payload size does not match its header");
    }

    M loaded(rows, cols);
    io_impl::read_exact(input, loaded.data(), payload_bytes, "matrix data");

    row_mat = std::move(loaded);
}
}  // namespace rabitqlib
