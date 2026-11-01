#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep (std::optional seed)

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/detail/fast_scale_table.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/pack_excode.hpp"
#include "rabitqlib/core/provisional/quantize.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/core/types.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {
namespace {

// Raised for corrupt or incompatible serialized bytes; a ValueError in Python.
struct FormatError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

constexpr size_t kMaxRotatorDim = 65536;
constexpr size_t kMaxTotalBits = 9;

static_assert(sizeof(quant::v1::Factors) == 3 * sizeof(float));

// --- Structured dtypes, byte-compatible with the planned C structs ---------------

py::dtype float_record_dtype(const std::array<const char*, 3>& names) {
    py::list name_list;
    py::list formats;
    py::list offsets;
    for (size_t i = 0; i < names.size(); ++i) {
        name_list.append(names[i]);
        formats.append("<f4");
        offsets.append(i * sizeof(float));
    }
    return {name_list, formats, offsets, 3 * sizeof(float)};
}

py::dtype factors_dtype() { return float_record_dtype({"add", "rescale", "error"}); }

py::dtype estimate_dtype() {
    return float_record_dtype({"distance", "error", "refine_input"});
}

py::dtype query_terms_dtype() {
    return float_record_dtype({"g_add", "g_error", "g_k1xsumq"});
}

// --- Argument checks ------------------------------------------------------------

void check_nbits(size_t nbits, size_t low, size_t high, const char* name = "nbits") {
    if (nbits < low || nbits > high) {
        throw std::invalid_argument(
            std::string(name) + " must be in [" + std::to_string(low) + ", " +
            std::to_string(high) + "]; got " + std::to_string(nbits)
        );
    }
}

// Layer calls work on rotated, padded vectors: d is a positive multiple of 32.
size_t checked_working_dim(size_t d, const py::array& array, const char* name) {
    if (d == 0 || d % 32 != 0) {
        throw std::invalid_argument(
            std::string(name) +
            " must have a last dimension that is a positive multiple of 32; got shape " +
            array_shape(array)
        );
    }
    return d;
}

template <typename TP>
void check_code_values(const RowView<TP>& codes, size_t nbits, const char* name) {
    const auto maximum = static_cast<TP>((1U << nbits) - 1);
    for (size_t i = 0; i < codes.rows; ++i) {
        const TP* row = codes.row(i);
        for (size_t j = 0; j < codes.cols; ++j) {
            if (row[j] > maximum) {
                throw std::invalid_argument(
                    std::string(name) + " values must be below 2**nbits = " +
                    std::to_string(size_t{1} << nbits) + "; got " + std::to_string(row[j]) +
                    " at (" + std::to_string(i) + ", " + std::to_string(j) + ")"
                );
            }
        }
    }
}

template <typename T>
py::array_t<T> new_rows(size_t rows, size_t cols) {
    return py::array_t<T>(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
}

// The output rows: a new array, or the caller's out= after its exact checks.
template <typename T>
std::pair<py::array, RowView<T>> output_rows(
    const py::object& out, size_t rows, size_t cols
) {
    if (out.is_none()) {
        py::array array = new_rows<T>(rows, cols);
        auto view = make_row_view<T>(array);
        return {std::move(array), std::move(view)};
    }
    auto view = checked_output_rows<T>(out, rows, cols);
    return {py::reinterpret_borrow<py::array>(out), std::move(view)};
}

// A (d,) float32 vector, copied if misaligned.
py::array_t<float, py::array::c_style | py::array::forcecast> ensure_vector(
    py::handle value, size_t d, const char* name
) {
    auto vector = ensure_1d_array<float>(value, name);
    if (static_cast<size_t>(vector.shape(0)) != d) {
        throw std::invalid_argument(
            std::string(name) + " must have shape (" + std::to_string(d) +
            ",); got shape " + array_shape(vector)
        );
    }
    if (reinterpret_cast<std::uintptr_t>(vector.data()) % alignof(float) != 0) {
        py::array_t<float, py::array::c_style | py::array::forcecast> aligned(
            vector.shape(0)
        );
        std::memcpy(
            aligned.mutable_data(), vector.data(), static_cast<size_t>(vector.nbytes())
        );
        return aligned;
    }
    return vector;
}

// --- Layer 1 --------------------------------------------------------------------

size_t padded_dim(size_t dim) {
    if (dim == 0) {
        throw std::invalid_argument("dim must be positive");
    }
    if (dim > std::numeric_limits<size_t>::max() - 31) {
        throw std::invalid_argument("dim is too large to pad");
    }
    return round_up_to_multiple(dim, 32);
}

template <typename TP>
void quantize_rows(
    const float* residuals,
    size_t rows,
    size_t d,
    size_t nbits,
    double t_const,
    const RowView<TP>& out,
    size_t num_threads
) {
    index_detail::parallel_for(0, rows, num_threads, [&](size_t i, size_t /*thread*/) {
        quant::provisional::combined_code(
            residuals + (i * d), d, nbits, out.row(i), t_const
        );
    });
}

py::array quantize(
    py::handle residuals,
    size_t nbits,
    bool fast_quantization,
    size_t num_threads,
    const py::object& out
) {
    check_nbits(nbits, 1, kMaxTotalBits);
    auto input = ensure_2d_array<float>(residuals, "residuals");
    const auto rows = static_cast<size_t>(input.shape(0));
    const size_t d =
        checked_working_dim(static_cast<size_t>(input.shape(1)), input, "residuals");

    double t_const = -1;
    if (fast_quantization && nbits > 1) {
        const std::optional<double> frozen =
            quant::rabitq_impl::ex_bits::frozen_const_scaling_factor(d, nbits - 1);
        if (!frozen.has_value()) {
            throw std::invalid_argument(
                "fast_quantization requires a working dimension of at most " +
                std::to_string(quant::rabitq_impl::ex_bits::kFastScaleMaxDim) +
                " (the frozen scale table); got shape " + array_shape(input)
            );
        }
        t_const = *frozen;
    }

    const auto run = [&](auto& result) {
        auto [array, view] = std::move(result);
        {
            py::gil_scoped_release release;
            check_coordinates(input.data(), rows * d, d, "residuals");
            quantize_rows(input.data(), rows, d, nbits, t_const, view, num_threads);
        }
        return array;
    };
    if (nbits == kMaxTotalBits) {
        auto result = output_rows<uint16_t>(out, rows, d);
        return run(result);
    }
    auto result = output_rows<uint8_t>(out, rows, d);
    return run(result);
}

template <typename TP>
py::tuple split_rows(const RowView<TP>& codes, size_t nbits, size_t prefix_bits) {
    const size_t ex_bits = nbits - prefix_bits;
    const auto mask = static_cast<TP>((1U << ex_bits) - 1);
    auto prefix = new_rows<uint8_t>(codes.rows, codes.cols);
    auto ex = new_rows<uint8_t>(codes.rows, codes.cols);
    uint8_t* prefix_data = prefix.mutable_data();
    uint8_t* ex_data = ex.mutable_data();
    {
        py::gil_scoped_release release;
        check_code_values(codes, nbits, "codes");
        for (size_t i = 0; i < codes.rows; ++i) {
            const TP* row = codes.row(i);
            for (size_t j = 0; j < codes.cols; ++j) {
                prefix_data[(i * codes.cols) + j] = static_cast<uint8_t>(row[j] >> ex_bits);
                ex_data[(i * codes.cols) + j] = static_cast<uint8_t>(row[j] & mask);
            }
        }
    }
    return py::make_tuple(prefix, ex);
}

py::tuple split(py::handle codes, size_t nbits, size_t prefix_bits) {
    check_nbits(nbits, 2, kMaxTotalBits);
    check_nbits(prefix_bits, 1, nbits - 1, "prefix_bits");
    if (nbits == kMaxTotalBits) {
        const auto view = exact_dtype_rows<uint16_t>(codes, "codes");
        checked_working_dim(view.cols, view.owner, "codes");
        return split_rows(view, nbits, prefix_bits);
    }
    const auto view = exact_dtype_rows<uint8_t>(codes, "codes");
    checked_working_dim(view.cols, view.owner, "codes");
    return split_rows(view, nbits, prefix_bits);
}

template <typename TP>
py::array code_factors_rows(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& residuals,
    const RowView<TP>& codes,
    size_t nbits,
    MetricType metric,
    const float* centroid,
    size_t num_threads,
    const py::object& out
) {
    const size_t rows = codes.rows;
    const size_t d = codes.cols;
    const py::dtype dtype = factors_dtype();
    py::array result;
    if (out.is_none()) {
        result = py::array(dtype, std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows)});
    } else {
        const std::string expected =
            "out must be a writable, C-contiguous NumPy array of "
            "dtype quant.FACTORS_DTYPE and shape (" +
            std::to_string(rows) + ",)";
        if (!py::isinstance<py::array>(out)) {
            throw std::invalid_argument(expected + "; got " + array_input_type(out));
        }
        result = py::reinterpret_borrow<py::array>(out);
        if (!result.dtype().attr("__eq__")(dtype).cast<bool>() || result.ndim() != 1 ||
            static_cast<size_t>(result.shape(0)) != rows || !result.writeable() ||
            (rows > 1 && result.strides(0) != static_cast<py::ssize_t>(dtype.itemsize())) ||
            reinterpret_cast<std::uintptr_t>(result.data()) % alignof(float) != 0) {
            throw std::invalid_argument(expected + "; got shape " + array_shape(result));
        }
    }
    auto* factors = static_cast<quant::v1::Factors*>(const_cast<void*>(result.data()));
    {
        py::gil_scoped_release release;
        check_coordinates(residuals.data(), rows * d, d, "residuals");
        if (centroid != nullptr) {
            check_coordinates(centroid, d, d, "centroid");
        }
        check_code_values(codes, nbits, "codes");
        const auto factors_of = [&](size_t i) {
            return quant::provisional::code_factors(
                residuals.data() + (i * d), centroid, d, codes.row(i), nbits, metric
            );
        };
        // Report the lowest mismatched row, whatever the thread schedule.
        std::atomic<size_t> first_rejected{rows};
        index_detail::parallel_for(0, rows, num_threads, [&](size_t i, size_t /*thread*/) {
            try {
                factors[i] = factors_of(i);
            } catch (const std::invalid_argument&) {
                size_t current = first_rejected.load();
                while (i < current && !first_rejected.compare_exchange_weak(current, i)) {}
            }
        });
        if (const size_t i = first_rejected.load(); i < rows) {
            try {
                (void)factors_of(i);
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument(
                    "codes row " + std::to_string(i) + " for residuals row " +
                    std::to_string(i) + ": " + error.what()
                );
            }
        }
    }
    return result;
}

py::array code_factors(
    py::handle residuals,
    py::handle codes,
    size_t nbits,
    const std::string& metric,
    const py::object& centroid,
    size_t num_threads,
    const py::object& out
) {
    check_nbits(nbits, 1, kMaxTotalBits);
    const MetricType metric_type = metric_from_string(metric);
    auto input = ensure_2d_array<float>(residuals, "residuals");
    const size_t d =
        checked_working_dim(static_cast<size_t>(input.shape(1)), input, "residuals");
    const auto rows = static_cast<size_t>(input.shape(0));

    py::array_t<float, py::array::c_style | py::array::forcecast> centroid_array;
    const float* centroid_data = nullptr;
    if (!centroid.is_none()) {
        centroid_array = ensure_vector(centroid, d, "centroid");
        centroid_data = centroid_array.data();
    }

    const auto run = [&](const auto& view) {
        if (view.rows != rows || view.cols != d) {
            throw std::invalid_argument(
                "codes must have the shape of residuals, " + array_shape(input) +
                "; got shape " + array_shape(view.owner)
            );
        }
        return code_factors_rows(
            input, view, nbits, metric_type, centroid_data, num_threads, out
        );
    };
    if (nbits == kMaxTotalBits) {
        return run(exact_dtype_rows<uint16_t>(codes, "codes"));
    }
    return run(exact_dtype_rows<uint8_t>(codes, "codes"));
}

py::array pack_codes(
    py::handle codes, size_t nbits, size_t num_threads, const py::object& out
) {
    check_nbits(nbits, 1, 8);
    const auto view = exact_dtype_rows<uint8_t>(codes, "codes");
    const size_t d = checked_working_dim(view.cols, view.owner, "codes");
    auto [array, packed] = output_rows<uint8_t>(out, view.rows, d * nbits / 8);
    {
        py::gil_scoped_release release;
        check_code_values(view, nbits, "codes");
        index_detail::parallel_for(
            0, view.rows, num_threads, [&](size_t i, size_t /*thread*/) {
                quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                    view.row(i), packed.row(i), d, nbits
                );
            }
        );
    }
    return array;
}

py::array unpack_codes(
    py::handle packed, size_t nbits, size_t num_threads, const py::object& out
) {
    check_nbits(nbits, 1, 8);
    const auto view = exact_dtype_rows<uint8_t>(packed, "packed");
    const size_t bits = view.cols * 8;
    if (view.cols == 0 || bits % nbits != 0 || (bits / nbits) % 32 != 0) {
        throw std::invalid_argument(
            "packed must have d * nbits / 8 bytes per row for a positive multiple of 32 "
            "d; got shape " +
            array_shape(view.owner) + " with nbits " + std::to_string(nbits)
        );
    }
    const size_t d = bits / nbits;
    auto [array, codes] = output_rows<uint8_t>(out, view.rows, d);
    {
        py::gil_scoped_release release;
        index_detail::parallel_for(
            0, view.rows, num_threads, [&](size_t i, size_t /*thread*/) {
                quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(
                    view.row(i), codes.row(i), d, nbits
                );
            }
        );
    }
    return array;
}

// --- Blob format (docs/formats.md) ------------------------------------------------

constexpr std::array<char, 8> kBlobMagic = {'R', 'A', 'B', 'Q', 'B', 'L', 'O', 'B'};
constexpr size_t kCommonHeaderBytes = 40;
constexpr uint32_t kRotatorKind = 1;
constexpr uint32_t kRotatorVersion = 1;
constexpr size_t kRotatorHeaderBytes = 64;
constexpr uint32_t kSeedKnown = 1;
constexpr uint32_t kMethodFhtKac = 1;
constexpr uint32_t kMethodMatrix = 2;
constexpr uint32_t kScalarFloat32 = 1;
constexpr uint32_t kScalarFloat64 = 2;

// CRC-32 with the IEEE polynomial, as zlib.crc32 computes it.
uint32_t crc32(const uint8_t* data, size_t size) {
    static const std::array<uint32_t, 256> kTable = [] {
        std::array<uint32_t, 256> table{};
        for (uint32_t i = 0; i < table.size(); ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value & 1U) != 0 ? 0xEDB88320U ^ (value >> 1U) : value >> 1U;
            }
            table[i] = value;
        }
        return table;
    }();
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < size; ++i) {
        crc = kTable[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8U);
    }
    return crc ^ 0xFFFFFFFFU;
}

// Checksum of a blob with its crc32 field (offset 32) taken as zero.
uint32_t blob_checksum(const uint8_t* blob, size_t size) {
    std::vector<uint8_t> copy(blob, blob + size);
    std::fill_n(copy.begin() + 32, 4, uint8_t{0});
    return crc32(copy.data(), copy.size());
}

void put_u32(uint8_t* out, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void put_u64(uint8_t* out, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

uint32_t get_u32(const uint8_t* in) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(in[i]) << (8 * i);
    }
    return value;
}

uint64_t get_u64(const uint8_t* in) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(in[i]) << (8 * i);
    }
    return value;
}

// A contiguous, read-only byte view of any buffer-protocol object, held for one call.
class ByteView {
   private:
    py::buffer_info info_;

   public:
    explicit ByteView(py::handle value) {
        const std::string message =
            "data must be a contiguous bytes-like object; got " + array_input_type(value);
        if (!py::isinstance<py::buffer>(value)) {
            throw py::type_error(message);
        }
        info_ = py::reinterpret_borrow<py::buffer>(value).request();
        // C-contiguous: each stride is the size of everything after it.
        py::ssize_t expected = info_.itemsize;
        for (py::ssize_t axis = info_.ndim - 1; axis >= 0; --axis) {
            const auto index = static_cast<size_t>(axis);
            if (info_.shape[index] > 1 && info_.strides[index] != expected) {
                throw py::type_error(message);
            }
            expected *= info_.shape[index];
        }
    }

    [[nodiscard]] const uint8_t* data() const {
        return static_cast<const uint8_t*>(info_.ptr);
    }
    [[nodiscard]] size_t size() const {
        return static_cast<size_t>(info_.size * info_.itemsize);
    }
};

// --- Rotator --------------------------------------------------------------------

RotatorType rotator_method(const std::string& method) {
    if (method == "fht_kac" || method == "fht") {
        return RotatorType::FhtKacRotator;
    }
    if (method == "matrix") {
        return RotatorType::MatrixRotator;
    }
    throw std::invalid_argument("method must be 'fht_kac' (alias 'fht') or 'matrix'");
}

const char* method_name(RotatorType type) {
    return type == RotatorType::FhtKacRotator ? "fht_kac" : "matrix";
}

size_t rotator_body_bytes(RotatorType type, size_t dim, size_t padded_dim) {
    return type == RotatorType::FhtKacRotator ? padded_dim / 2
                                              : sizeof(float) * dim * padded_dim;
}

// Python's quant.Rotator: immutable, so copies and threads may share it.
class QuantRotator {
   private:
    std::shared_ptr<const Rotator<float>> rotator_;
    RotatorType type_ = RotatorType::FhtKacRotator;
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    std::optional<uint32_t> seed_;

    QuantRotator() = default;

    static void check_dimensions(RotatorType type, size_t dim, size_t padded_dim) {
        if (dim == 0 || dim > kMaxRotatorDim) {
            throw std::invalid_argument(
                "dim must be in [1, " + std::to_string(kMaxRotatorDim) + "]; got " +
                std::to_string(dim)
            );
        }
        if (type == RotatorType::FhtKacRotator && dim < 64) {
            throw std::invalid_argument(
                "dim must be at least 64 for method 'fht_kac'; got " + std::to_string(dim)
            );
        }
        if (padded_dim % 32 != 0 || padded_dim < dim || padded_dim > kMaxRotatorDim) {
            throw std::invalid_argument(
                "padded_dim must be a multiple of 32 in [dim, " +
                std::to_string(kMaxRotatorDim) + "]; got " + std::to_string(padded_dim)
            );
        }
    }

    static size_t checked_padded_dim_argument(const py::object& value) {
        const std::string message =
            "padded_dim must be an integer multiple of 32 in [dim, " +
            std::to_string(kMaxRotatorDim) + "]";
        if (py::isinstance<py::bool_>(value)) {
            throw py::type_error(message);
        }
        py::object integer;
        try {
            integer = py::module_::import("operator").attr("index")(value);
        } catch (const py::error_already_set&) { throw py::type_error(message); }
        if (integer < py::int_(0) || integer > py::int_(kMaxRotatorDim)) {
            throw std::invalid_argument(message);
        }
        return integer.cast<size_t>();
    }

   public:
    QuantRotator(
        size_t dim,
        const std::string& method,
        const py::object& padded,
        const py::object& seed
    )
        : type_(rotator_method(method)), dim_(dim) {
        padded_dim_ = padded.is_none() ? (dim == 0 ? 0 : padded_dim(dim))
                                       : checked_padded_dim_argument(padded);
        check_dimensions(type_, dim_, padded_dim_);
        seed_ = seed.is_none() ? std::random_device{}()
                               : static_cast<uint32_t>(checked_seed(
                                     seed, std::numeric_limits<uint32_t>::max()
                                 ));
        py::gil_scoped_release release;
        // A matrix rotation runs a QR factorization; keep the GIL released for it.
        std::unique_ptr<Rotator<float>> created(
            choose_rotator<float>(dim_, type_, padded_dim_, *seed_)
        );
        rotator_ = std::move(created);
    }

    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t padded_dim_value() const { return padded_dim_; }
    [[nodiscard]] std::string method() const { return method_name(type_); }
    [[nodiscard]] std::optional<uint32_t> seed() const { return seed_; }

    [[nodiscard]] py::array rotate(
        py::handle x, size_t num_threads, const py::object& out
    ) const {
        auto input = ensure_2d_array<float>(x, "x");
        ensure_dimension(input, dim_, "x dimension does not match dim");
        const auto rows = static_cast<size_t>(input.shape(0));
        auto [array, view] = output_rows<float>(out, rows, padded_dim_);
        {
            py::gil_scoped_release release;
            check_coordinates(input.data(), rows * dim_, dim_, "x");
            index_detail::parallel_for(
                0, rows, num_threads, [&](size_t i, size_t /*thread*/) {
                    rotator_->rotate(input.data() + (i * dim_), view.row(i));
                }
            );
        }
        return array;
    }

    [[nodiscard]] py::bytes to_bytes() const {
        const size_t body = rotator_->dump_bytes();
        std::vector<uint8_t> blob(kRotatorHeaderBytes + body, 0);
        std::memcpy(blob.data(), kBlobMagic.data(), kBlobMagic.size());
        put_u32(&blob[8], kRotatorKind);
        put_u32(&blob[12], kRotatorVersion);
        put_u32(&blob[16], static_cast<uint32_t>(kRotatorHeaderBytes));
        put_u32(&blob[20], seed_.has_value() ? kSeedKnown : 0);
        put_u64(&blob[24], body);
        put_u32(
            &blob[40], type_ == RotatorType::FhtKacRotator ? kMethodFhtKac : kMethodMatrix
        );
        put_u32(&blob[44], kScalarFloat32);
        put_u32(&blob[48], static_cast<uint32_t>(dim_));
        put_u32(&blob[52], static_cast<uint32_t>(padded_dim_));
        put_u32(&blob[56], seed_.value_or(0));
        rotator_->save(reinterpret_cast<char*>(blob.data() + kRotatorHeaderBytes));
        put_u32(&blob[32], blob_checksum(blob.data(), blob.size()));
        return {reinterpret_cast<const char*>(blob.data()), blob.size()};
    }

    static QuantRotator from_bytes(py::handle data) {
        const ByteView bytes(data);
        const uint8_t* blob = bytes.data();
        const size_t size = bytes.size();
        if (size < kCommonHeaderBytes) {
            throw FormatError("data is too short for a rabitqlib blob");
        }
        if (std::memcmp(blob, kBlobMagic.data(), kBlobMagic.size()) != 0) {
            throw FormatError("data is not a rabitqlib blob (bad magic)");
        }
        const uint32_t kind = get_u32(blob + 8);
        if (kind != kRotatorKind) {
            throw FormatError(
                "data is not a rotator blob; got kind " + std::to_string(kind)
            );
        }
        const uint32_t version = get_u32(blob + 12);
        if (version != kRotatorVersion) {
            throw FormatError(
                "data has unsupported rotator format version " + std::to_string(version) +
                "; this release reads version " + std::to_string(kRotatorVersion)
            );
        }
        if (get_u32(blob + 16) != kRotatorHeaderBytes) {
            throw FormatError("data has an invalid header size for rotator version 1");
        }
        const uint64_t body = get_u64(blob + 24);
        if (size < kRotatorHeaderBytes || body != size - kRotatorHeaderBytes) {
            throw FormatError(
                "data length does not match its header (truncated or padded)"
            );
        }
        if (get_u32(blob + 32) != blob_checksum(blob, size)) {
            throw FormatError("data checksum mismatch (corrupt blob)");
        }
        const uint32_t flags = get_u32(blob + 20);
        if ((flags & ~kSeedKnown) != 0) {
            throw FormatError("data has unknown flags");
        }
        if (get_u32(blob + 36) != 0 || get_u32(blob + 60) != 0) {
            throw FormatError("data has non-zero reserved fields");
        }
        const uint32_t method = get_u32(blob + 40);
        if (method != kMethodFhtKac && method != kMethodMatrix) {
            throw FormatError(
                "data has an unknown rotator method " + std::to_string(method)
            );
        }
        const uint32_t scalar = get_u32(blob + 44);
        if (scalar == kScalarFloat64) {
            throw FormatError("data stores a float64 rotator, which version 1 cannot load");
        }
        if (scalar != kScalarFloat32) {
            throw FormatError("data has an unknown scalar type " + std::to_string(scalar));
        }
        QuantRotator result;
        result.type_ = method == kMethodFhtKac ? RotatorType::FhtKacRotator
                                               : RotatorType::MatrixRotator;
        result.dim_ = get_u32(blob + 48);
        result.padded_dim_ = get_u32(blob + 52);
        try {
            check_dimensions(result.type_, result.dim_, result.padded_dim_);
        } catch (const std::invalid_argument& error) {
            throw FormatError(
                std::string("data has invalid rotator dimensions: ") + error.what()
            );
        }
        const uint32_t seed = get_u32(blob + 56);
        if ((flags & kSeedKnown) != 0) {
            result.seed_ = seed;
        } else if (seed != 0) {
            throw FormatError("data has a seed without the seed_known flag");
        }
        if (body != rotator_body_bytes(result.type_, result.dim_, result.padded_dim_)) {
            throw FormatError("data body size does not match the rotator parameters");
        }
        auto rotator = rotator_impl::make_rotator_for_loading(
            result.dim_, result.type_, result.padded_dim_
        );
        if (rotator->dump_bytes() != body) {
            throw std::logic_error("Rotator body size disagrees with the blob format");
        }
        rotator->load(reinterpret_cast<const char*>(blob + kRotatorHeaderBytes));
        result.rotator_ = std::move(rotator);
        return result;
    }

    [[nodiscard]] std::string repr() const {
        return "Rotator(dim=" + std::to_string(dim_) +
               ", padded_dim=" + std::to_string(padded_dim_) + ", method='" + method() +
               "', seed=" + (seed_.has_value() ? std::to_string(*seed_) : "None") + ")";
    }
};

}  // namespace
}  // namespace rabitqlib::python_bindings

void register_quant(py::module_& m) {
    using namespace rabitqlib::python_bindings;

    auto quant = m.def_submodule(
        "_quant",
        "Native backend of rabitqlib.quant. Import rabitqlib.quant instead of this module."
    );
    py::register_exception<FormatError>(
        quant, "FormatError", py::module_::import("builtins").attr("ValueError")
    );
    quant.attr("FormatError").attr("__doc__") =
        "Corrupt or incompatible serialized bytes. A subclass of ValueError.";

    quant.attr("FACTORS_DTYPE") = factors_dtype();
    quant.attr("ESTIMATE_DTYPE") = estimate_dtype();
    quant.attr("QUERY_TERMS_DTYPE") = query_terms_dtype();

    quant.def(
        "padded_dim",
        &padded_dim,
        py::arg("dim"),
        R"doc(Return the working dimension for dim: dim rounded up to a multiple of 32.

Parameters
----------
dim : int
    Original vector dimension, positive.

Returns
-------
padded_dim : int
    The smallest multiple of 32 that is at least dim.)doc"
    );

    quant.def(
        "quantize",
        &quantize,
        py::arg("residuals"),
        py::kw_only(),
        py::arg("nbits"),
        py::arg("fast_quantization") = false,
        py::arg("num_threads") = 0,
        py::arg("out") = py::none(),
        R"doc(Quantize rotated residuals into unsigned nbits-bit codes.

Parameters
----------
residuals : array_like of float32, shape (n, d)
    Rotated, padded residuals (vector minus centroid); d is a positive
    multiple of 32. Converted to contiguous float32.
nbits : int
    Total bits per coordinate, 1 to 9, including the sign bit.
fast_quantization : bool, optional
    Use the frozen fast-mode scale instead of the exact search. Default is
    False. Rejected for d above 65536.
num_threads : int, optional
    Worker threads; 0 (the default) uses every available CPU. Results do not
    depend on the thread count.
out : numpy.ndarray, optional
    Output of the exact dtype and shape, with contiguous rows.

Returns
-------
codes : numpy.ndarray, shape (n, d)
    Codes in [0, 2**nbits): uint8, or uint16 when nbits is 9.)doc"
    );

    quant.def(
        "split",
        &split,
        py::arg("codes"),
        py::kw_only(),
        py::arg("nbits"),
        py::arg("prefix_bits"),
        R"doc(Split nbits-bit codes into a prefix and extra bits.

Parameters
----------
codes : numpy.ndarray, shape (n, d)
    Codes from quantize: uint8, or uint16 when nbits is 9. Never cast; rows
    may have any stride.
nbits : int
    Width of codes, 2 to 9.
prefix_bits : int
    Width of the prefix, 1 to nbits - 1.

Returns
-------
prefix, ex : tuple of numpy.ndarray of uint8, shape (n, d)
    prefix = codes >> ex_bits and ex = codes & (2**ex_bits - 1), where
    ex_bits = nbits - prefix_bits.)doc"
    );

    quant.def(
        "code_factors",
        &code_factors,
        py::arg("residuals"),
        py::arg("codes"),
        py::kw_only(),
        py::arg("nbits"),
        py::arg("metric"),
        py::arg("centroid") = py::none(),
        py::arg("num_threads") = 0,
        py::arg("out") = py::none(),
        R"doc(Return the factors of nbits-bit codes for their residuals.

Pass the full codes with their width for the full-code factors, or a prefix
from split with nbits=prefix_bits for the prefix factors.

Parameters
----------
residuals : array_like of float32, shape (n, d)
    The rotated, padded residuals the codes were quantized from.
codes : numpy.ndarray, shape (n, d)
    uint8 codes, or uint16 when nbits is 9. Never cast; rows may have any
    stride. Each row must quantize its residual.
nbits : int
    Width of codes, 1 to 9.
metric : {'l2', 'ip'}
    Distance metric. Required: factors for the wrong metric give wrong
    distances without an error.
centroid : array_like of float32, shape (d,), optional
    The centroid in working space; None means zero. It only affects add.
num_threads : int, optional
    Worker threads; 0 (the default) uses every available CPU.
out : numpy.ndarray, optional
    A C-contiguous FACTORS_DTYPE array of shape (n,).

Returns
-------
factors : numpy.ndarray of FACTORS_DTYPE, shape (n,)
    The add, rescale and error factors of each code.)doc"
    );

    quant.def(
        "pack_codes",
        &pack_codes,
        py::arg("codes"),
        py::kw_only(),
        py::arg("nbits"),
        py::arg("num_threads") = 0,
        py::arg("out") = py::none(),
        R"doc(Pack nbits-bit codes into the packed-codes format (version 1).

At nbits=1 this is the sign bitset, equal to
numpy.packbits(codes, axis=1, bitorder="little").

Parameters
----------
codes : numpy.ndarray of uint8, shape (n, d)
    Codes in [0, 2**nbits). Never cast; rows may have any stride.
nbits : int
    Width of codes, 1 to 8. Split 9-bit codes first.
num_threads : int, optional
    Worker threads; 0 (the default) uses every available CPU.
out : numpy.ndarray, optional
    Output of the exact dtype and shape, with contiguous rows.

Returns
-------
packed : numpy.ndarray of uint8, shape (n, d * nbits // 8))doc"
    );

    quant.def(
        "unpack_codes",
        &unpack_codes,
        py::arg("packed"),
        py::kw_only(),
        py::arg("nbits"),
        py::arg("num_threads") = 0,
        py::arg("out") = py::none(),
        R"doc(Unpack codes written by pack_codes; the inverse is lossless.

Parameters
----------
packed : numpy.ndarray of uint8, shape (n, d * nbits // 8)
    Packed rows; d, a positive multiple of 32, follows from the row size.
    Never cast; rows may have any stride.
nbits : int
    Width of the packed codes, 1 to 8.
num_threads : int, optional
    Worker threads; 0 (the default) uses every available CPU.
out : numpy.ndarray, optional
    Output of the exact dtype and shape, with contiguous rows.

Returns
-------
codes : numpy.ndarray of uint8, shape (n, d))doc"
    );

    static constexpr char kRotatorDoc[] =
        R"doc(A seeded rotation from the original dimension to the padded working space.

Parameters
----------
dim : int
    Original vector dimension: 64 to 65536 for 'fht_kac', 1 to 65536 for
    'matrix'.
method : {'fht_kac', 'fht', 'matrix'}, optional
    FHT/Kac rotation (default; 'fht' is an alias) or a dense random matrix.
padded_dim : int, optional
    Working dimension, a multiple of 32 in [dim, 65536]. Default is
    padded_dim(dim).
seed : int, optional
    Seed in [0, 2**32 - 1]; None (the default) draws one. A seeded FHT/Kac
    rotation is identical on every platform; a matrix rotation only with the
    same standard library and build, so persist rotators with to_bytes.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import quant
>>> rot = quant.Rotator(128, seed=0)
>>> rot.rotate(np.ones((2, 128), dtype=np.float32)).shape
(2, 128))doc";

    py::class_<QuantRotator>(quant, "Rotator", kRotatorDoc)
        .def(
            py::init<size_t, const std::string&, const py::object&, const py::object&>(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("method") = "fht_kac",
            py::arg("padded_dim") = py::none(),
            py::arg("seed") = py::none(),
            kRotatorDoc
        )
        .def_property_readonly(
            "dim", &QuantRotator::dim, R"doc(Original vector dimension.)doc"
        )
        .def_property_readonly(
            "padded_dim",
            &QuantRotator::padded_dim_value,
            R"doc(Working dimension of rotated vectors.)doc"
        )
        .def_property_readonly(
            "method",
            &QuantRotator::method,
            R"doc(Canonical method name, 'fht_kac' or 'matrix'.)doc"
        )
        .def_property_readonly(
            "seed",
            &QuantRotator::seed,
            R"doc(The seed the rotation was drawn from, or None if unknown.)doc"
        )
        .def(
            "rotate",
            &QuantRotator::rotate,
            py::arg("x"),
            py::kw_only(),
            py::arg("num_threads") = 0,
            py::arg("out") = py::none(),
            R"doc(Rotate vectors into the padded working space.

Parameters
----------
x : array_like of float32, shape (n, dim)
    Vectors in the original space. Converted to contiguous float32.
num_threads : int, optional
    Worker threads; 0 (the default) uses every available CPU.
out : numpy.ndarray, optional
    A float32 array of shape (n, padded_dim) with contiguous rows.

Returns
-------
rotated : numpy.ndarray of float32, shape (n, padded_dim))doc"
        )
        .def(
            "to_bytes",
            &QuantRotator::to_bytes,
            R"doc(Serialize the rotation as a versioned rotator blob (docs/formats.md).)doc"
        )
        .def_static(
            "from_bytes",
            &QuantRotator::from_bytes,
            py::arg("data"),
            R"doc(Load a rotator blob written by to_bytes.

Parameters
----------
data : bytes-like
    Any contiguous buffer, read without an intermediate copy.

Raises
------
FormatError
    If the bytes are corrupt, truncated or of an unsupported version.)doc"
        )
        .def(
            "__copy__",
            [](const py::object& self) { return self; },
            "Return self; rotators are immutable."
        )
        .def(
            "__deepcopy__",
            [](const py::object& self, const py::object& /*memo*/) { return self; },
            py::arg("memo"),
            "Return self; rotators are immutable."
        )
        .def("__repr__", &QuantRotator::repr, "Return the rotator's parameters.")
        .def(
            py::pickle(
                [](const QuantRotator& self) { return self.to_bytes(); },
                [](const py::bytes& state) { return QuantRotator::from_bytes(state); }
            )
        );
}
