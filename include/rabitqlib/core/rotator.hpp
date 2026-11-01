#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/detail/simd/matrix_dispatch.hpp"
#include "rabitqlib/core/detail/simd/rotator_dispatch.hpp"
#include "rabitqlib/core/types.hpp"  // IWYU pragma: export
#include "rabitqlib/third/Eigen/QR"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib {

template <typename T>
class Rotator {
   protected:
    size_t dim_ = 0;
    size_t padded_dim_ = 0;

   public:
    explicit Rotator() = default;
    explicit Rotator(size_t dim, size_t padded_dim) : dim_(dim), padded_dim_(padded_dim) {}
    virtual ~Rotator() = default;
    virtual void rotate(const T* data, T* rotated_data) const = 0;
    virtual void load(std::ifstream&) = 0;
    virtual void save(std::ofstream&) const = 0;
    // Buffer I/O
    virtual void load(const char* data) = 0;
    virtual void save(char* data) const = 0;
    [[nodiscard]] virtual size_t dump_bytes() const = 0;
    [[nodiscard]] size_t size() const { return padded_dim_; }
};

namespace rotator_impl {

inline size_t padding_requirement(size_t dim, RotatorType type) {
    if (type == RotatorType::MatrixRotator) {
        return dim;
    }
    if (type == RotatorType::FhtKacRotator) {
        if (dim > std::numeric_limits<size_t>::max() - 31) {
            throw std::invalid_argument("Rotator dimension is too large to pad");
        }
        return round_up_to_multiple(dim, 32);
    }
    throw std::invalid_argument("Invalid rotator type in padding_requirement()");
}

inline size_t checked_padded_dimension(size_t dim, RotatorType type, size_t padded_dim) {
    if (dim == 0) {
        throw std::invalid_argument("Rotator dimension must be positive");
    }
    if (padded_dim == 0) {
        padded_dim = padding_requirement(dim, type);
    }

    if (padded_dim < dim) {
        throw std::invalid_argument(
            "Padded rotator dimension must not be smaller than input"
        );
    }

    if (padded_dim != padding_requirement(padded_dim, type)) {
        throw std::invalid_argument("Invalid padded dimension for the rotator type");
    }

    return padded_dim;
}

// Internal loading path: allocate storage, then restore the saved rotation before use.
struct AllocateOnly {};

template <typename T = float>
class MatrixRotator : public Rotator<T> {
   private:
    RowMajorMatrix<T> rand_mat_;

   public:
    // The seed makes the matrix reproducible only with the same standard library
    // (std::normal_distribution) and build (QR rounding); persist it with save().
    explicit MatrixRotator(
        size_t dim, size_t padded_dim, uint32_t seed = std::random_device{}()
    )
        : MatrixRotator(dim, padded_dim, AllocateOnly{}) {
        RowMajorMatrix<T> rand = random_gaussian_matrix<T>(padded_dim, padded_dim, seed);
        Eigen::HouseholderQR<RowMajorMatrix<T>> qr(rand);
        RowMajorMatrix<T> q_inv =
            qr.householderQ()
                .transpose();  // The inverse of an orthogonal matrix is its transpose.

        // the random matrix only need the first dim rows, since we just pad zeros for
        // the vector to be rotated to padded dimension
        std::memcpy(&rand_mat_(0, 0), &q_inv(0, 0), sizeof(T) * dim * padded_dim);
    }
    MatrixRotator(size_t dim, size_t padded_dim, AllocateOnly)
        : Rotator<T>(dim, padded_dim), rand_mat_(dim, padded_dim) {}

    MatrixRotator() = default;
    ~MatrixRotator() override = default;

    MatrixRotator& operator=(const MatrixRotator&) = default;

    void load(std::ifstream& input) override {
        input.read(
            reinterpret_cast<char*>(rand_mat_.data()),
            static_cast<std::streamsize>(MatrixRotator<T>::dump_bytes())
        );
    }

    void save(std::ofstream& output) const override {
        output.write(
            reinterpret_cast<const char*>(rand_mat_.data()),
            static_cast<std::streamsize>(MatrixRotator<T>::dump_bytes())
        );
    }

    void load(const char* data) override {
        std::memcpy(rand_mat_.data(), data, MatrixRotator<T>::dump_bytes());
    }

    void save(char* data) const override {
        std::memcpy(data, rand_mat_.data(), MatrixRotator<T>::dump_bytes());
    }

    [[nodiscard]] size_t dump_bytes() const override {
        return sizeof(T) * this->dim_ * this->padded_dim_;
    }

    void rotate(const T* data, T* rotated_data) const override {
        if constexpr (std::is_same_v<T, float>) {
            // Match Eigen's alias-safe assignment: callers may rotate in place.
            std::vector<float> result(this->padded_dim_);
            simd::matrix_product(
                data, rand_mat_.data(), result.data(), 1, this->dim_, this->padded_dim_
            );
            std::copy(result.begin(), result.end(), rotated_data);
        } else {
            ConstRowMajorMatrixMap<T> v(data, 1, static_cast<Eigen::Index>(this->dim_));
            RowMajorMatrixMap<T> rv(
                rotated_data, 1, static_cast<Eigen::Index>(this->padded_dim_)
            );
            rv = v * rand_mat_;
        }
    }
};

// Internal linkage: the per-ISA HNSW search sources include this header.
static inline void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    simd::flip_sign(flip, data, dim);
}

class FhtKacRotator : public Rotator<float> {
   private:
    std::vector<uint8_t> flip_;
    size_t trunc_dim_ = 0;
    float scale_factor_ = 0;

    static constexpr size_t kByteLen = 8;

   public:
    static constexpr size_t kMaxDim = 65536;

    explicit FhtKacRotator(
        size_t dim, size_t padded_dim, uint32_t seed = std::random_device{}()
    )
        : FhtKacRotator(dim, padded_dim, AllocateOnly{}) {
        // Frozen, portable mapping: the top byte of each mt19937 output. Its output
        // sequence is fixed by the standard, unlike std::uniform_int_distribution;
        // GCC 11+ libstdc++'s uniform_int_distribution<int>(0, 255) yields these same
        // bytes; older libstdc++ occasionally drew different ones.
        std::mt19937 gen(seed);
        for (auto& i : flip_) {
            i = static_cast<uint8_t>(gen() >> 24);
        }
    }

    FhtKacRotator(size_t dim, size_t padded_dim, AllocateOnly)
        : Rotator<float>(dim, padded_dim) {
        if (dim < 64 || padded_dim < dim || padded_dim % 32 != 0 || padded_dim > kMaxDim) {
            throw std::invalid_argument("Unsupported dimension for FhtKacRotator");
        }
        flip_.resize((padded_dim / kByteLen) * 4);
        size_t bottom_log_dim = floor_log2(dim);
        trunc_dim_ = size_t{1} << bottom_log_dim;
        scale_factor_ = 1.0F / std::sqrt(static_cast<float>(trunc_dim_));
    }
    FhtKacRotator() = default;
    ~FhtKacRotator() override = default;

    void load(std::ifstream& input) override {
        input.read(
            reinterpret_cast<char*>(flip_.data()),
            static_cast<std::streamsize>(FhtKacRotator::dump_bytes())
        );
    }

    void save(std::ofstream& output) const override {
        output.write(
            reinterpret_cast<const char*>(flip_.data()),
            static_cast<std::streamsize>(FhtKacRotator::dump_bytes())
        );
    }

    void load(const char* data) override {
        std::memcpy(flip_.data(), data, FhtKacRotator::dump_bytes());
    }

    void save(char* data) const override {
        std::memcpy(data, flip_.data(), FhtKacRotator::dump_bytes());
    }

    [[nodiscard]] size_t dump_bytes() const override {
        return sizeof(uint8_t) * flip_.size();
    }

    FhtKacRotator(const FhtKacRotator&) = default;

    FhtKacRotator& operator=(const FhtKacRotator& other) {
        // Preserve the old dimensions if copying the sign storage fails.
        flip_ = other.flip_;
        dim_ = other.dim_;
        padded_dim_ = other.padded_dim_;
        trunc_dim_ = other.trunc_dim_;
        scale_factor_ = other.scale_factor_;
        return *this;
    }

    static void kacs_walk(float* data, size_t len) { simd::kacs_walk(data, len); }

    void rotate(const float* data, float* rotated_data) const override {
        simd::fht_rotate(
            data, rotated_data, dim_, padded_dim_, trunc_dim_, scale_factor_, flip_.data()
        );
    }
};
inline std::unique_ptr<Rotator<float>> make_rotator_for_loading(
    size_t dim, RotatorType type, size_t padded_dim
) {
    padded_dim = checked_padded_dimension(dim, type, padded_dim);
    if (type == RotatorType::FhtKacRotator) {
        return std::make_unique<FhtKacRotator>(dim, padded_dim, AllocateOnly{});
    }
    return std::make_unique<MatrixRotator<float>>(dim, padded_dim, AllocateOnly{});
}
}  // namespace rotator_impl

// Return an owning pointer; wrap it in RAII and use size() for the output dimension.
// FHT/Kac supports float; its seed gives the same rotation on every platform.
// Matrix rotation supports float/double; its seed reproduces the matrix only with
// the same standard library and build, so persist matrix rotators with save().
template <typename T>
Rotator<T>* choose_rotator(
    size_t dim,
    RotatorType type = RotatorType::FhtKacRotator,
    size_t padded_dim = 0,
    uint32_t seed = std::random_device{}()
) {
    padded_dim = rotator_impl::checked_padded_dimension(dim, type, padded_dim);

    if (type == RotatorType::FhtKacRotator) {
        if constexpr (std::is_same_v<T, float>) {
            return ::new rotator_impl::FhtKacRotator(dim, padded_dim, seed);
        } else {
            throw std::invalid_argument("FhtKacRotator only supports float");
        }
    }

    if (type == RotatorType::MatrixRotator) {
        return ::new rotator_impl::MatrixRotator<T>(dim, padded_dim, seed);
    }

    throw std::invalid_argument("Invalid rotator type in choose_rotator()");
}
}  // namespace rabitqlib
