#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/gil_safe_call_once.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep (std::optional seed)

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/detail/fast_scale_table.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/pack_excode.hpp"
#include "rabitqlib/core/provisional/quantize.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/core/types.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/estimator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {
namespace {

// Raised for corrupt or incompatible serialized bytes; a ValueError in Python.
struct FormatError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// The blob format and both rotation methods share the FHT/Kac limit.
constexpr size_t kMaxRotatorDim = rotator_impl::FhtKacRotator::kMaxDim;
constexpr size_t kMaxTotalBits = 9;

// --- Records and structured dtypes, byte-compatible with the planned C structs -----

// An estimate record, the layout of ESTIMATE_DTYPE; factor records are v1::Factors.
struct Estimate {
    float distance;
    float error;
    float refine_input;
};

static_assert(sizeof(quant::v1::Factors) == 3 * sizeof(float));
static_assert(sizeof(Estimate) == 3 * sizeof(float));

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

// Each dtype is built once and shared, since every estimate call needs one.
template <typename Names>
py::dtype stored_dtype(py::gil_safe_call_once_and_store<py::dtype>& storage, Names names) {
    return storage.call_once_and_store_result(
                      [&] { return float_record_dtype(names()); }
    ).get_stored();
}

py::dtype factors_dtype() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::dtype> storage;
    return stored_dtype(storage, [] {
        return std::array<const char*, 3>{"add", "rescale", "error"};
    });
}

py::dtype estimate_dtype() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::dtype> storage;
    return stored_dtype(storage, [] {
        return std::array<const char*, 3>{"distance", "error", "refine_input"};
    });
}

py::dtype query_terms_dtype() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::dtype> storage;
    return stored_dtype(storage, [] {
        return std::array<const char*, 3>{"g_add", "g_error", "g_k1xsumq"};
    });
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

// The rows a call reads: all of them, or the caller's indices, checked against n.
struct Selection {
    py::object owner;  // Keeps the converted indices alive; none for all rows.
    const int64_t* indices = nullptr;
    size_t count = 0;

    [[nodiscard]] size_t row(size_t k) const {
        return indices == nullptr ? k : static_cast<size_t>(indices[k]);
    }
};

Selection all_rows(size_t rows) {
    Selection selected;
    selected.count = rows;
    return selected;
}

Selection selection(const py::object& indices, size_t rows) {
    if (indices.is_none()) {
        return all_rows(rows);
    }
    Selection selected;
    auto array = ensure_1d_integer_array(indices, "indices");
    selected.indices = array.data();
    selected.count = static_cast<size_t>(array.shape(0));
    selected.owner = std::move(array);
    for (size_t k = 0; k < selected.count; ++k) {
        const int64_t i = selected.indices[k];
        if (i < 0 || static_cast<uint64_t>(i) >= rows) {
            throw std::invalid_argument(
                "indices must be in [0, " + std::to_string(rows) + "); got " +
                std::to_string(i) + " at position " + std::to_string(k)
            );
        }
    }
    return selected;
}

template <typename TP>
void check_code_values(
    const RowView<TP>& codes, const Selection& selected, size_t nbits, const char* name
) {
    const auto maximum = static_cast<TP>((1U << nbits) - 1);
    for (size_t k = 0; k < selected.count; ++k) {
        const size_t i = selected.row(k);
        const TP* row = codes.row(i);
        for (size_t j = 0; j < codes.cols; ++j) {
            if (row[j] > maximum) {
                throw std::invalid_argument(
                    std::string(name) + " values must be below 2**" +
                    std::to_string(nbits) + " = " + std::to_string(size_t{1} << nbits) +
                    "; got " + std::to_string(row[j]) + " at (" + std::to_string(i) + ", " +
                    std::to_string(j) + ")"
                );
            }
        }
    }
}

// A float32 vector copied into owned storage and checked like index inputs. Its length
// must be expected_d, or with expected_d == 0 a positive multiple of 32.
std::vector<float> owned_vector(py::handle value, const char* name, size_t expected_d) {
    auto array = ensure_1d_array<float>(value, name);
    const auto d = static_cast<size_t>(array.shape(0));
    if (expected_d == 0 && (d == 0 || d % 32 != 0)) {
        throw std::invalid_argument(
            std::string(name) +
            " must have a length that is a positive multiple of 32; got shape " +
            array_shape(array)
        );
    }
    if (expected_d != 0 && d != expected_d) {
        throw std::invalid_argument(
            std::string(name) + " must have shape (" + std::to_string(expected_d) +
            ",); got shape " + array_shape(array)
        );
    }
    std::vector<float> vector(d);
    std::memcpy(vector.data(), array.data(), d * sizeof(float));
    check_coordinates(vector.data(), d, d, name);
    return vector;
}

// A centroid in the working space, or an empty vector for none.
std::vector<float> centroid_vector(const py::object& centroid, size_t d) {
    return centroid.is_none() ? std::vector<float>{}
                              : owned_vector(centroid, "centroid", d);
}

// --- Outputs ----------------------------------------------------------------------

template <typename T>
py::array_t<T> new_rows(size_t rows, size_t cols) {
    return py::array_t<T>(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
}

// An input that an out= must not overlap: its name and the array the call reads.
struct NamedInput {
    const char* name;
    py::handle array;  // Null or None when the input is absent.
};
using Inputs = std::initializer_list<NamedInput>;

// The bytes an array spans, from its lowest to past its highest element.
std::pair<std::uintptr_t, std::uintptr_t> byte_extent(const py::array& array) {
    if (array.size() == 0) {
        return {0, 0};
    }
    auto low = reinterpret_cast<std::uintptr_t>(array.data());
    auto high = low;
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
        const py::ssize_t span = (array.shape(axis) - 1) * array.strides(axis);
        if (span < 0) {
            low -= static_cast<std::uintptr_t>(-span);
        } else {
            high += static_cast<std::uintptr_t>(span);
        }
    }
    return {low, high + static_cast<std::uintptr_t>(array.itemsize())};
}

// Reject an out= that shares memory with an input, as the C++ API rejects overlapping
// buffers: rows written first would change rows read later. Exact aliasing is rejected
// too, so the rule does not depend on how a call orders its reads and writes.
void check_output_disjoint(const py::array& out, Inputs inputs) {
    const auto [out_low, out_high] = byte_extent(out);
    if (out_low == out_high) {
        return;
    }
    for (const NamedInput& input : inputs) {
        if (!input.array || input.array.is_none()) {
            continue;
        }
        const auto [low, high] =
            byte_extent(py::reinterpret_borrow<py::array>(input.array));
        if (low < out_high && out_low < high) {
            throw std::invalid_argument(std::string("out must not overlap ") + input.name);
        }
    }
}

// The output rows: a new array, or the caller's out= after its exact checks. The
// view's owner is the array to return.
template <typename T>
RowView<T> output_rows(const py::object& out, size_t rows, size_t cols, Inputs inputs) {
    if (out.is_none()) {
        return make_row_view<T>(new_rows<T>(rows, cols));
    }
    auto view = checked_output_rows<T>(out, rows, cols);
    check_output_disjoint(view.owner, inputs);
    return view;
}

// A new 1-D array of dtype, or the caller's out= of exactly that dtype and shape.
// dtype_name is how messages name the dtype, such as "quant.ESTIMATE_DTYPE".
template <typename T>
std::pair<py::array, T*> vector_output(
    const py::object& out,
    const py::dtype& dtype,
    const char* dtype_name,
    size_t rows,
    Inputs inputs
) {
    static_assert(alignof(T) == alignof(float));
    if (out.is_none()) {
        py::array result(dtype, std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows)});
        auto* data = static_cast<T*>(result.mutable_data());
        return {std::move(result), data};
    }
    const std::string expected = std::string("out must be a writable, C-contiguous ") +
                                 "NumPy array of dtype " + dtype_name + " and shape (" +
                                 std::to_string(rows) + ",)";
    if (!py::isinstance<py::array>(out)) {
        throw std::invalid_argument(expected + "; got " + array_input_type(out));
    }
    auto result = py::reinterpret_borrow<py::array>(out);
    if (!result.dtype().equal(dtype) || result.ndim() != 1 ||
        static_cast<size_t>(result.shape(0)) != rows || !result.writeable() ||
        (rows > 1 && result.strides(0) != static_cast<py::ssize_t>(sizeof(T))) ||
        reinterpret_cast<std::uintptr_t>(result.data()) % alignof(T) != 0) {
        throw std::invalid_argument(expected + "; got shape " + array_shape(result));
    }
    check_output_disjoint(result, inputs);
    auto* data = static_cast<T*>(result.mutable_data());
    return {std::move(result), data};
}

std::pair<py::array, Estimate*> estimate_output(
    const py::object& out, size_t rows, Inputs inputs
) {
    return vector_output<Estimate>(
        out, estimate_dtype(), "quant.ESTIMATE_DTYPE", rows, inputs
    );
}

// --- Record inputs ----------------------------------------------------------------

// A 1-D array of one of the float-record dtypes, read as contiguous aligned records.
// Strided or misaligned input is copied; other dtypes are rejected, never cast.
template <typename T>
struct Records {
    py::array owner;
    const T* data = nullptr;
    size_t rows = 0;

    [[nodiscard]] const T& operator[](size_t i) const { return data[i]; }
};

template <typename T>
Records<T> record_input(
    py::handle value, const py::dtype& dtype, const char* dtype_name, const char* name
) {
    const std::string expected =
        std::string(name) + " must be a 1D NumPy array of dtype quant." + dtype_name;
    auto array = py::array::ensure(value);
    if (!array) {
        throw std::invalid_argument(expected + "; got " + array_input_type(value));
    }
    if (!array.dtype().equal(dtype)) {
        throw std::invalid_argument(
            expected + "; got dtype " + py::str(array.dtype()).cast<std::string>()
        );
    }
    if (array.ndim() != 1) {
        throw std::invalid_argument(expected + "; got shape " + array_shape(array));
    }
    const auto rows = static_cast<size_t>(array.shape(0));
    if ((rows > 1 && array.strides(0) != static_cast<py::ssize_t>(sizeof(T))) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0) {
        array = py::array(array.attr("copy")());
    }
    Records<T> records;
    records.rows = rows;
    records.data = static_cast<const T*>(array.data());
    records.owner = std::move(array);
    return records;
}

// The factors argument: FACTORS_DTYPE records, one per row of the named input.
Records<quant::v1::Factors> factors_input(py::handle value, size_t rows, const char* of) {
    auto records = record_input<quant::v1::Factors>(
        value, factors_dtype(), "FACTORS_DTYPE", "factors"
    );
    if (records.rows != rows) {
        throw std::invalid_argument(
            std::string("factors must have one record per row of ") + of + ", " +
            std::to_string(rows) + "; got shape " + array_shape(records.owner)
        );
    }
    return records;
}

// Factors are read only for the selected rows; call with the GIL released.
void check_finite_factors(
    const Records<quant::v1::Factors>& factors, const Selection& selected
) {
    for (size_t k = 0; k < selected.count; ++k) {
        const quant::v1::Factors& f = factors[selected.row(k)];
        if (!std::isfinite(f.add) || !std::isfinite(f.rescale) || !std::isfinite(f.error)) {
            throw std::invalid_argument("factors must contain only finite values");
        }
    }
}

// --- Layer 1 --------------------------------------------------------------------

size_t padded_dim(size_t dim) {
    if (dim == 0) {
        throw std::invalid_argument("dim must be positive");
    }
    return rotator_impl::padding_requirement(dim, RotatorType::FhtKacRotator);
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

    // The whole-array pass applies the index coordinate bound; combined_code's own
    // per-row checks guard its public contract and cost O(d) against the code search.
    const auto run = [&](const auto& view) {
        {
            py::gil_scoped_release release;
            check_coordinates(input.data(), rows * d, d, "residuals");
            index_detail::parallel_for(
                0, rows, num_threads, [&](size_t i, size_t /*thread*/) {
                    quant::provisional::combined_code(
                        input.data() + (i * d), d, nbits, view.row(i), t_const
                    );
                }
            );
        }
        return view.owner;
    };
    if (nbits == kMaxTotalBits) {
        return run(output_rows<uint16_t>(out, rows, d, {{"residuals", input}}));
    }
    return run(output_rows<uint8_t>(out, rows, d, {{"residuals", input}}));
}

template <typename TP>
py::tuple split_rows(const RowView<TP>& codes, size_t nbits, size_t prefix_bits) {
    const size_t ex_bits = nbits - prefix_bits;
    const auto mask = static_cast<TP>((1U << ex_bits) - 1);
    auto prefix = new_rows<uint8_t>(codes.rows, codes.cols);
    auto ex = new_rows<uint8_t>(codes.rows, codes.cols);
    uint8_t* prefix_data = prefix.mutable_data();
    uint8_t* ex_data = ex.mutable_data();
    const Selection every_row = all_rows(codes.rows);
    {
        py::gil_scoped_release release;
        check_code_values(codes, every_row, nbits, "codes");
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
    const std::vector<float>& centroid,
    size_t num_threads,
    const py::object& out
) {
    const size_t rows = codes.rows;
    const size_t d = codes.cols;
    auto [result, data] = vector_output<quant::v1::Factors>(
        out,
        factors_dtype(),
        "quant.FACTORS_DTYPE",
        rows,
        {{"residuals", residuals}, {"codes", codes.owner}}
    );
    {
        py::gil_scoped_release release;
        check_coordinates(residuals.data(), rows * d, d, "residuals");
        // A caller's out= is written only after every row succeeds.
        std::vector<quant::v1::Factors> staged(out.is_none() ? 0 : rows);
        quant::v1::Factors* factors = out.is_none() ? data : staged.data();
        // code_factors checks each row's code values and that the code quantizes the
        // residual; the binding adds the row number to its message.
        const auto factors_of = [&](size_t i) {
            return quant::provisional::code_factors(
                residuals.data() + (i * d), centroid.data(), d, codes.row(i), nbits, metric
            );
        };
        // Report the lowest rejected row, whatever the thread schedule.
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
        std::copy(staged.begin(), staged.end(), data);
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
    // One zero vector for every row when there is no centroid.
    std::vector<float> center = centroid_vector(centroid, d);
    center.resize(d, 0.0F);

    const auto run = [&](const auto& view) {
        if (view.rows != rows || view.cols != d) {
            throw std::invalid_argument(
                "codes must have the shape of residuals, " + array_shape(input) +
                "; got shape " + array_shape(view.owner)
            );
        }
        return code_factors_rows(input, view, nbits, metric_type, center, num_threads, out);
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
    const auto packed =
        output_rows<uint8_t>(out, view.rows, d * nbits / 8, {{"codes", view.owner}});
    const Selection every_row = all_rows(view.rows);
    {
        py::gil_scoped_release release;
        check_code_values(view, every_row, nbits, "codes");
        index_detail::parallel_for(
            0, view.rows, num_threads, [&](size_t i, size_t /*thread*/) {
                quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                    view.row(i), packed.row(i), d, nbits
                );
            }
        );
    }
    return packed.owner;
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
    const auto codes = output_rows<uint8_t>(out, view.rows, d, {{"packed", view.owner}});
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
    return codes.owner;
}

// --- Layer 2 --------------------------------------------------------------------

// The query-side terms of the reference formula (docs/formats.md, core/estimator.hpp).
struct QueryTerms {
    float g_add = 0;
    float g_error = 0;
    float g_k1xsumq = 0;
};

QueryTerms compute_query_terms(
    const float* query, const float* centroid, size_t d, MetricType metric
) {
    const float distance_sqr =
        centroid == nullptr ? l2norm_sqr(query, d) : euclidean_sqr(query, centroid, d);
    QueryTerms terms;
    terms.g_add = metric == METRIC_L2
                      ? distance_sqr
                      : (centroid == nullptr ? 0.0F : -dot_product(query, centroid, d));
    terms.g_error = std::sqrt(distance_sqr);
    terms.g_k1xsumq =
        std::accumulate(query, query + d, 0.0F) * fastscan::detail::code_center(1);
    if (!std::isfinite(terms.g_add) || !std::isfinite(terms.g_error) ||
        !std::isfinite(terms.g_k1xsumq)) {
        throw std::overflow_error("Query terms are not representable in float32");
    }
    return terms;
}

inline Estimate estimate_of(
    float ip, size_t bits, const quant::v1::Factors& f, const QueryTerms& terms
) {
    return {
        quant::detail::full_est_dist_from_ip(
            ip, bits, f.add, f.rescale, terms.g_add, terms.g_k1xsumq
        ),
        f.error * terms.g_error,
        ip};
}

// <q, code> for unpacked codes, summed in order as quant::v1::compute does. This is
// the portable reference path, not a SIMD kernel.
template <typename TP>
inline float unpacked_ip(const float* query, const TP* code, size_t d) {
    float ip = 0;
    for (size_t j = 0; j < d; ++j) {
        ip += query[j] * static_cast<float>(code[j]);
    }
    return ip;
}

py::object query_terms(
    py::handle rotated_query, const std::string& metric, const py::object& centroid
) {
    const MetricType metric_type = metric_from_string(metric);
    const auto query = owned_vector(rotated_query, "rotated_query", 0);
    const auto center = centroid_vector(centroid, query.size());
    const QueryTerms terms = compute_query_terms(
        query.data(), center.empty() ? nullptr : center.data(), query.size(), metric_type
    );
    py::array result(query_terms_dtype(), std::vector<py::ssize_t>{1});
    auto* data = static_cast<float*>(result.mutable_data());
    data[0] = terms.g_add;
    data[1] = terms.g_error;
    data[2] = terms.g_k1xsumq;
    return result.attr("__getitem__")(0);
}

py::array estimate_from_ip(
    py::handle ip,
    py::handle factors,
    py::handle terms,
    size_t nbits,
    size_t num_threads,
    const py::object& out
) {
    check_nbits(nbits, 1, kMaxTotalBits);
    auto ips = ensure_1d_array<float>(ip, "ip");
    const auto rows = static_cast<size_t>(ips.shape(0));
    const auto records = factors_input(factors, rows, "ip");
    auto terms_array = py::array::ensure(terms);
    if (!terms_array || !terms_array.dtype().equal(query_terms_dtype()) ||
        terms_array.size() != 1) {
        throw std::invalid_argument(
            "terms must be one record of dtype quant.QUERY_TERMS_DTYPE, as query_terms "
            "returns; got " +
            (terms_array ? "dtype " + py::str(terms_array.dtype()).cast<std::string>() +
                               " and shape " + array_shape(terms_array)
                         : array_input_type(terms))
        );
    }
    // Copied before anything is written, so terms may share memory with out.
    std::array<float, 3> values{};
    std::memcpy(values.data(), terms_array.data(), sizeof(values));
    if (!all_finite(values.data(), values.size())) {
        throw std::invalid_argument("terms must contain only finite values");
    }
    const QueryTerms query{values[0], values[1], values[2]};
    auto [result, estimates] =
        estimate_output(out, rows, {{"ip", ips}, {"factors", records.owner}});
    const Selection every_row = all_rows(rows);
    {
        py::gil_scoped_release release;
        if (!all_finite(ips.data(), rows)) {
            throw std::invalid_argument("ip must contain only finite values");
        }
        check_finite_factors(records, every_row);
        index_detail::parallel_for(0, rows, num_threads, [&](size_t i, size_t /*thread*/) {
            estimates[i] = estimate_of(ips.data()[i], nbits, records[i], query);
        });
    }
    return result;
}

// --- Layer 3 --------------------------------------------------------------------

// Bit-reversal of every byte value.
constexpr std::array<uint8_t, 256> kReversedBytes = [] {
    std::array<uint8_t, 256> table{};
    for (size_t value = 0; value < table.size(); ++value) {
        uint8_t reversed = 0;
        for (size_t bit = 0; bit < 8; ++bit) {
            reversed =
                static_cast<uint8_t>(reversed | (((value >> bit) & 1U) << (7 - bit)));
        }
        table[value] = reversed;
    }
    return table;
}();

// FastScan blocks (docs/formats.md): 32 vectors per block, the last one zero-padded.
py::array pack_fastscan_blocks(
    py::handle bitsets, py::handle factors, size_t num_threads, const py::object& out
) {
    const auto codes = exact_dtype_rows<uint8_t>(bitsets, "bitsets");
    if (codes.cols == 0 || (codes.cols * 8) % 32 != 0) {
        throw std::invalid_argument(
            "bitsets must have d // 8 bytes per row for a positive multiple of 32 d; got "
            "shape " +
            array_shape(codes.owner)
        );
    }
    const size_t d = codes.cols * 8;
    const auto records = factors_input(factors, codes.rows, "bitsets");
    const size_t n = codes.rows;
    const size_t num_blocks = div_round_up(n, fastscan::kBatchSize);
    const auto blocks = output_rows<uint8_t>(
        out,
        num_blocks,
        BatchDataMap<float>::data_bytes(d),
        {{"bitsets", codes.owner}, {"factors", records.owner}}
    );
    const Selection every_row = all_rows(n);
    {
        py::gil_scoped_release release;
        check_finite_factors(records, every_row);
        // One block of sign bytes per worker, in the order the FastScan packer reads:
        // the first coordinate of each byte in its highest bit.
        const size_t workers =
            std::max<size_t>(1, std::min(resolve_num_threads(num_threads), num_blocks));
        std::vector<uint8_t> scratch(workers * fastscan::kBatchSize * codes.cols);
        index_detail::parallel_for(
            0, num_blocks, num_threads, [&](size_t b, size_t thread) {
                const size_t begin = b * fastscan::kBatchSize;
                const size_t lanes = std::min(fastscan::kBatchSize, n - begin);
                uint8_t* msb_first =
                    scratch.data() + (thread * fastscan::kBatchSize * codes.cols);
                for (size_t lane = 0; lane < lanes; ++lane) {
                    const uint8_t* row = codes.row(begin + lane);
                    for (size_t byte = 0; byte < codes.cols; ++byte) {
                        msb_first[(lane * codes.cols) + byte] = kReversedBytes[row[byte]];
                    }
                }
                uint8_t* block = blocks.row(b);
                fastscan::pack_codes(d, msb_first, lanes, block);
                BatchDataMap<float> map(reinterpret_cast<char*>(block), d);
                for (size_t lane = 0; lane < fastscan::kBatchSize; ++lane) {
                    const quant::v1::Factors f =
                        lane < lanes ? records[begin + lane] : quant::v1::Factors{0, 0, 0};
                    map.f_add()[lane] = f.add;
                    map.f_rescale()[lane] = f.rescale;
                    map.f_error()[lane] = f.error;
                }
            }
        );
    }
    return blocks.owner;
}

// --- Query ------------------------------------------------------------------------

// The FastScan query of a Query, built on the first estimate_fastscan call. It borrows
// the Query's own vector, so a copied or moved Query starts without one and builds its
// own.
struct LazyBatchQuery {
    std::unique_ptr<SplitBatchQuery<float>> query;

    LazyBatchQuery() = default;
    LazyBatchQuery(const LazyBatchQuery& /*other*/) {}
    LazyBatchQuery(LazyBatchQuery&& /*other*/) noexcept {}
    LazyBatchQuery& operator=(const LazyBatchQuery&) = delete;
    LazyBatchQuery& operator=(LazyBatchQuery&&) = delete;
    ~LazyBatchQuery() = default;
};

// A Query's guard. A copy gets a fresh, idle guard rather than the original's state.
struct QueryAccess : IndexAccess {
    QueryAccess()
        : IndexAccess("Query is busy: set_centroid cannot overlap another call") {}
    QueryAccess(const QueryAccess& /*other*/) : QueryAccess() {}
    QueryAccess(QueryAccess&& /*other*/) noexcept : QueryAccess() {}
    QueryAccess& operator=(const QueryAccess&) = delete;
    QueryAccess& operator=(QueryAccess&&) = delete;
    ~QueryAccess() = default;
};

// A rotated query in the working space, with the terms of its current centroid. It owns
// copies of the query and centroid, so copies and unpickled queries never share storage.
class QuantQuery {
   private:
    std::vector<float> query_;
    std::vector<float> centroid_;  // Empty means no centroid (zero).
    MetricType metric_;
    QueryTerms terms_;
    LazyBatchQuery batch_query_;
    QueryAccess access_;

    [[nodiscard]] size_t d() const { return query_.size(); }

    // SplitBatchQuery::set_g_add uses the -<q, c> IP convention of the terms here.
    void apply_batch_terms() {
        if (metric_ == METRIC_L2) {
            batch_query_.query->set_g_add(terms_.g_error);
        } else {
            batch_query_.query->set_g_add(terms_.g_error, -terms_.g_add);
        }
    }

    // Called with the GIL held, so concurrent readers cannot both build it.
    const SplitBatchQuery<float>& batch_query() {
        if (!batch_query_.query) {
            batch_query_.query = std::make_unique<SplitBatchQuery<float>>(
                query_.data(), d(), 0, metric_, true
            );
            apply_batch_terms();
        }
        return *batch_query_.query;
    }

    // The guard is taken before any input conversion, so conversion cannot race
    // set_centroid.
    template <typename Score>
    py::array scored(Score&& score) {
        const auto operation = access_.read();
        return score();
    }

    static void check_row_length(
        const py::array& array,
        size_t cols,
        size_t expected,
        const char* name,
        const char* what
    ) {
        if (cols != expected) {
            throw std::invalid_argument(
                std::string(name) + " must have " + what + " = " +
                std::to_string(expected) + " columns for this query; got shape " +
                array_shape(array)
            );
        }
    }

   public:
    QuantQuery(std::vector<float> query, MetricType metric, std::vector<float> centroid)
        : query_(std::move(query))
        , centroid_(std::move(centroid))
        , metric_(metric)
        , terms_(compute_query_terms(
              query_.data(), centroid_.empty() ? nullptr : centroid_.data(), d(), metric
          )) {}

    static QuantQuery create(
        py::handle rotated_query, const std::string& metric, const py::object& centroid
    ) {
        const MetricType metric_type = metric_from_string(metric);
        auto query = owned_vector(rotated_query, "rotated_query", 0);
        auto center = centroid_vector(centroid, query.size());
        return {std::move(query), metric_type, std::move(center)};
    }

    [[nodiscard]] size_t dim() const { return d(); }
    [[nodiscard]] std::string metric() const { return metric_to_string(metric_); }

    void set_centroid(const py::object& centroid) {
        const auto operation = access_.write();
        auto center = centroid_vector(centroid, d());
        const QueryTerms terms = compute_query_terms(
            query_.data(), center.empty() ? nullptr : center.data(), d(), metric_
        );
        centroid_ = std::move(center);
        terms_ = terms;
        if (batch_query_.query) {
            apply_batch_terms();
        }
    }

    [[nodiscard]] QuantQuery copy() {
        const auto operation = access_.read();
        return QuantQuery(*this);
    }

    py::tuple state() {
        const auto operation = access_.read();
        py::array_t<float> query(static_cast<py::ssize_t>(d()));
        std::memcpy(query.mutable_data(), query_.data(), d() * sizeof(float));
        py::object centroid = py::none();
        if (!centroid_.empty()) {
            py::array_t<float> center(static_cast<py::ssize_t>(d()));
            std::memcpy(center.mutable_data(), centroid_.data(), d() * sizeof(float));
            centroid = std::move(center);
        }
        return py::make_tuple(query, metric(), centroid);
    }

    static QuantQuery from_state(const py::tuple& state) {
        if (state.size() != 3 || !py::isinstance<py::str>(state[1])) {
            throw std::invalid_argument("Invalid Query pickle state");
        }
        return create(state[0], state[1].cast<std::string>(), state[2]);
    }

    [[nodiscard]] std::string repr() {
        const auto operation = access_.read();
        return "Query(dim=" + std::to_string(d()) + ", metric='" + metric() +
               "', centroid=" + (centroid_.empty() ? "None" : "set") + ")";
    }

    py::array estimate_bitset(
        py::handle bitsets,
        py::handle factors,
        const py::object& indices,
        size_t num_threads,
        const py::object& out
    ) {
        return scored([&] {
            const auto codes = exact_dtype_rows<uint8_t>(bitsets, "bitsets");
            check_row_length(codes.owner, codes.cols, d() / 8, "bitsets", "d // 8");
            const auto records = factors_input(factors, codes.rows, "bitsets");
            const Selection selected = selection(indices, codes.rows);
            auto [result, estimates] = estimate_output(
                out,
                selected.count,
                {{"bitsets", codes.owner},
                 {"factors", records.owner},
                 {"indices", selected.owner}}
            );
            const auto ip_func = select_excode_ipfunc(1);
            {
                py::gil_scoped_release release;
                check_finite_factors(records, selected);
                index_detail::parallel_for(
                    0, selected.count, num_threads, [&](size_t k, size_t /*thread*/) {
                        const size_t i = selected.row(k);
                        estimates[k] = estimate_of(
                            ip_func(query_.data(), codes.row(i), d()), 1, records[i], terms_
                        );
                    }
                );
            }
            return result;
        });
    }

    py::array estimate_codes(
        py::handle codes,
        py::handle factors,
        size_t nbits,
        const py::object& indices,
        size_t num_threads,
        const py::object& out
    ) {
        check_nbits(nbits, 1, kMaxTotalBits);
        return scored([&] {
            const auto run = [&](const auto& view) {
                check_row_length(view.owner, view.cols, d(), "codes", "d");
                const auto records = factors_input(factors, view.rows, "codes");
                const Selection selected = selection(indices, view.rows);
                auto [result, estimates] = estimate_output(
                    out,
                    selected.count,
                    {{"codes", view.owner},
                     {"factors", records.owner},
                     {"indices", selected.owner}}
                );
                {
                    py::gil_scoped_release release;
                    check_code_values(view, selected, nbits, "codes");
                    check_finite_factors(records, selected);
                    index_detail::parallel_for(
                        0, selected.count, num_threads, [&](size_t k, size_t /*thread*/) {
                            const size_t i = selected.row(k);
                            estimates[k] = estimate_of(
                                unpacked_ip(query_.data(), view.row(i), d()),
                                nbits,
                                records[i],
                                terms_
                            );
                        }
                    );
                }
                return result;
            };
            if (nbits == kMaxTotalBits) {
                return run(exact_dtype_rows<uint16_t>(codes, "codes"));
            }
            return run(exact_dtype_rows<uint8_t>(codes, "codes"));
        });
    }

    // refine (Packed = false) reads unpacked extra bits; refine_packed reads
    // pack_codes(ex, nbits=ex_bits) with the dispatched kernel.
    template <bool Packed>
    py::array refine(
        py::handle ex,
        py::handle factors,
        py::handle first,
        size_t ex_bits,
        size_t prefix_bits,
        const py::object& indices,
        size_t num_threads,
        const py::object& out
    ) {
        check_nbits(prefix_bits, 1, kMaxTotalBits - 1, "prefix_bits");
        check_nbits(ex_bits, 1, kMaxTotalBits - prefix_bits, "ex_bits");
        const size_t total_bits = prefix_bits + ex_bits;
        const char* name = Packed ? "packed" : "ex";
        return scored([&] {
            const auto codes = exact_dtype_rows<uint8_t>(ex, name);
            if constexpr (Packed) {
                check_row_length(
                    codes.owner, codes.cols, d() * ex_bits / 8, name, "d * ex_bits // 8"
                );
            } else {
                check_row_length(codes.owner, codes.cols, d(), name, "d");
            }
            const auto records = factors_input(factors, codes.rows, name);
            const Selection selected = selection(indices, codes.rows);
            const auto estimates =
                record_input<Estimate>(first, estimate_dtype(), "ESTIMATE_DTYPE", "first");
            if (estimates.rows != selected.count) {
                throw std::invalid_argument(
                    std::string("first must have one record per ") +
                    (indices.is_none() ? "row of " + std::string(name)
                                       : "entry of indices") +
                    ", " + std::to_string(selected.count) + "; got shape " +
                    array_shape(estimates.owner)
                );
            }
            auto [result, distances] = vector_output<float>(
                out,
                py::dtype::of<float>(),
                "float32",
                selected.count,
                {{name, codes.owner},
                 {"factors", records.owner},
                 {"first", estimates.owner},
                 {"indices", selected.owner}}
            );
            const auto ip_func = select_excode_ipfunc(ex_bits);
            const auto scale = static_cast<float>(1U << ex_bits);
            {
                py::gil_scoped_release release;
                if constexpr (!Packed) {
                    check_code_values(codes, selected, ex_bits, name);
                }
                check_finite_factors(records, selected);
                for (size_t k = 0; k < estimates.rows; ++k) {
                    if (!std::isfinite(estimates[k].refine_input)) {
                        throw std::invalid_argument(
                            "first['refine_input'] must contain only finite values"
                        );
                    }
                }
                index_detail::parallel_for(
                    0, selected.count, num_threads, [&](size_t k, size_t /*thread*/) {
                        const size_t i = selected.row(k);
                        const float ip_ex =
                            Packed ? ip_func(query_.data(), codes.row(i), d())
                                   : unpacked_ip(query_.data(), codes.row(i), d());
                        const quant::v1::Factors& f = records[i];
                        distances[k] = quant::detail::full_est_dist_from_ip(
                            (scale * estimates[k].refine_input) + ip_ex,
                            total_bits,
                            f.add,
                            f.rescale,
                            terms_.g_add,
                            terms_.g_k1xsumq
                        );
                    }
                );
            }
            return result;
        });
    }

    py::array estimate_fastscan(
        py::handle blocks, size_t num_points, size_t num_threads, const py::object& out
    ) {
        return scored([&] {
            const auto view = exact_dtype_rows<uint8_t>(blocks, "blocks");
            const size_t block_bytes = BatchDataMap<float>::data_bytes(d());
            const size_t num_blocks = div_round_up(num_points, fastscan::kBatchSize);
            if (view.rows != num_blocks || view.cols != block_bytes) {
                throw std::invalid_argument(
                    "blocks must have shape (" + std::to_string(num_blocks) + ", " +
                    std::to_string(block_bytes) + ") for num_points " +
                    std::to_string(num_points) + " and d " + std::to_string(d()) +
                    "; got shape " + array_shape(view.owner)
                );
            }
            auto [result, estimates] =
                estimate_output(out, num_points, {{"blocks", view.owner}});
            const SplitBatchQuery<float>& query = batch_query();
            {
                py::gil_scoped_release release;
                // Hand-built blocks get the finite-factor check of the other calls.
                for (size_t b = 0; b < num_blocks; ++b) {
                    const ConstBatchDataMap<float> map(
                        reinterpret_cast<const char*>(view.row(b)), d()
                    );
                    std::array<float, fastscan::kBatchSize> values{};
                    const size_t lanes = std::min(
                        fastscan::kBatchSize, num_points - (b * fastscan::kBatchSize)
                    );
                    for (const auto& field :
                         {map.f_add(), map.f_rescale(), map.f_error()}) {
                        field.copy_to(values.data(), values.size());
                        if (!all_finite(values.data(), lanes)) {
                            throw std::invalid_argument(
                                "blocks must contain only finite factors; block " +
                                std::to_string(b) + " does not"
                            );
                        }
                    }
                }
                index_detail::parallel_for(
                    0, num_blocks, num_threads, [&](size_t b, size_t /*thread*/) {
                        std::array<float, fastscan::kBatchSize> distances{};
                        std::array<float, fastscan::kBatchSize> lower_bounds{};
                        std::array<float, fastscan::kBatchSize> ips{};
                        std::array<float, fastscan::kBatchSize> errors{};
                        const auto* block = reinterpret_cast<const char*>(view.row(b));
                        split_batch_estdist(
                            block,
                            query,
                            d(),
                            distances.data(),
                            lower_bounds.data(),
                            ips.data(),
                            true
                        );
                        ConstBatchDataMap<float>(block, d())
                            .f_error()
                            .copy_to(errors.data(), errors.size());
                        const size_t begin = b * fastscan::kBatchSize;
                        const size_t lanes =
                            std::min(fastscan::kBatchSize, num_points - begin);
                        for (size_t lane = 0; lane < lanes; ++lane) {
                            estimates[begin + lane] = {
                                distances[lane], errors[lane] * terms_.g_error, ips[lane]};
                        }
                    }
                );
            }
            return result;
        });
    }
};

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

// Checksum of a blob with its crc32 field (offset 32) taken as zero, as zlib.crc32
// computes it, without copying the blob. The caller has checked that the blob covers
// the common header. Requires the GIL; zlib.crc32 releases it for large inputs.
uint32_t blob_checksum(const uint8_t* blob, size_t size) {
    constexpr size_t kChecksumOffset = 32;
    constexpr size_t kChecksumEnd = kChecksumOffset + 4;
    const py::object crc32 = py::module_::import("zlib").attr("crc32");
    const auto range = [&](size_t begin, size_t end) {
        return py::memoryview::from_memory(
            blob + begin, static_cast<py::ssize_t>(end - begin)
        );
    };
    py::object crc = crc32(range(0, kChecksumOffset));
    crc = crc32(py::bytes(std::string(kChecksumEnd - kChecksumOffset, '\0')), crc);
    crc = crc32(range(kChecksumEnd, size), crc);
    return crc.cast<uint32_t>();
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
        return checked_integer(
            value,
            kMaxRotatorDim,
            "padded_dim must be an integer multiple of 32 in [dim, " +
                std::to_string(kMaxRotatorDim) + "]"
        );
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
        const auto view = output_rows<float>(out, rows, padded_dim_, {{"x", input}});
        {
            py::gil_scoped_release release;
            check_coordinates(input.data(), rows * dim_, dim_, "x");
            index_detail::parallel_for(
                0, rows, num_threads, [&](size_t i, size_t /*thread*/) {
                    rotator_->rotate(input.data() + (i * dim_), view.row(i));
                }
            );
        }
        return view.owner;
    }

    [[nodiscard]] py::bytes to_bytes() const {
        const size_t body = rotator_->dump_bytes();
        const size_t size = kRotatorHeaderBytes + body;
        // Written in place: a matrix rotator's body is up to 16 GiB, so no staging copy.
        // Not py::bytes(nullptr, size): it turns a failed allocation into RuntimeError
        // while leaving Python's MemoryError set. Raise that MemoryError instead.
        auto result = py::reinterpret_steal<py::bytes>(
            PYBIND11_BYTES_FROM_STRING_AND_SIZE(nullptr, static_cast<py::ssize_t>(size))
        );
        if (!result) {
            throw py::error_already_set();
        }
        auto* blob = reinterpret_cast<uint8_t*>(PYBIND11_BYTES_AS_STRING(result.ptr()));
        std::memset(blob, 0, kRotatorHeaderBytes);
        std::memcpy(blob, kBlobMagic.data(), kBlobMagic.size());
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
        rotator_->save(reinterpret_cast<char*>(blob + kRotatorHeaderBytes));
        put_u32(&blob[32], blob_checksum(blob, size));
        return result;
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
    False. It has no effect at nbits=1, where no scale is searched, and is
    rejected for d above 65536 otherwise.
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
    quant.def(
        "query_terms",
        &query_terms,
        py::arg("rotated_query"),
        py::kw_only(),
        py::arg("metric"),
        py::arg("centroid") = py::none(),
        R"doc(Return the query-side terms of the reference estimator formula.

g_add is |q - c|**2 for L2 and -<q, c> for IP, g_error is |q - c|, and
g_k1xsumq is -0.5 * sum(q). The 1 of the IP distance 1 - <q, x> is folded
into the factors' add, not into g_add.

Parameters
----------
rotated_query : array_like of float32, shape (d,)
    A rotated, padded query; d is a positive multiple of 32.
metric : {'l2', 'ip', 'innerproduct'}
    The metric the factors were computed for.
centroid : array_like of float32, shape (d,), optional
    The working-space centroid the factors were computed with; None is zero.

Returns
-------
terms : numpy.void of QUERY_TERMS_DTYPE)doc"
    );

    quant.def(
        "estimate_from_ip",
        &estimate_from_ip,
        py::arg("ip"),
        py::arg("factors"),
        py::arg("terms"),
        py::kw_only(),
        py::arg("nbits"),
        py::arg("num_threads") = 1,
        py::arg("out") = py::none(),
        R"doc(Apply the reference estimator formula to the caller's inner products.

distance = add + g_add + rescale * (ip + g_k1xsumq * (2**nbits - 1)),
error = error_factor * g_error, and refine_input = ip.

Parameters
----------
ip : array_like of float32, shape (n,)
    Inner products <q, code> with the unsigned nbits-bit codes.
factors : numpy.ndarray of FACTORS_DTYPE, shape (n,)
    The factors of the same codes.
terms : numpy.void of QUERY_TERMS_DTYPE
    The query terms, as query_terms returns them.
nbits : int
    Bit width of the codes, 1 to 9.
num_threads : int, optional
    Worker threads; 1 by default, 0 uses every available CPU.
out : numpy.ndarray of ESTIMATE_DTYPE, shape (n,), optional

Returns
-------
estimates : numpy.ndarray of ESTIMATE_DTYPE, shape (n,))doc"
    );

    quant.def(
        "pack_fastscan_blocks",
        &pack_fastscan_blocks,
        py::arg("bitsets"),
        py::arg("factors"),
        py::kw_only(),
        py::arg("num_threads") = 0,
        py::arg("out") = py::none(),
        R"doc(Pack bitsets and their factors into FastScan blocks (version 1).

Each block holds 32 vectors, the library's fixed FastScan batch; the last block
is zero-padded when n % 32 != 0.

Parameters
----------
bitsets : numpy.ndarray of uint8, shape (n, d // 8)
    pack_codes(prefix, nbits=1) output.
factors : numpy.ndarray of FACTORS_DTYPE, shape (n,)
    The 1-bit (prefix) factors.
num_threads : int, optional
    Worker threads; 0 (the default) uses every available CPU.
out : numpy.ndarray of uint8, optional
    Shape (ceil(n / 32), d * 4 + 384) with contiguous rows.

Returns
-------
blocks : numpy.ndarray of uint8, shape (ceil(n / 32), d * 4 + 384))doc"
    );

    static constexpr const char* kQueryDoc =
        R"doc(A rotated, padded query for scoring codes in the working space.

The query and centroid are copied. set_centroid switches to another
centroid's factors without preparing the query again. Scoring calls may run
concurrently; set_centroid during one raises RuntimeError, so use copy() to
get one query per thread.

Parameters
----------
rotated_query : array_like of float32, shape (d,)
    A rotated, padded query; d is a positive multiple of 32.
metric : {'l2', 'ip', 'innerproduct'}
    The metric the factors were computed for.
centroid : array_like of float32, shape (d,), optional
    The working-space centroid the factors were computed with; None is zero.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import quant
>>> x = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32)
>>> codes = quant.quantize(x, nbits=1)
>>> factors = quant.code_factors(x, codes, nbits=1, metric="l2")
>>> q = quant.Query(x[0], metric="l2")
>>> q.estimate_bitset(quant.pack_codes(codes, nbits=1), factors).shape
(4,))doc";

    static constexpr const char* kIndicesDoc = R"doc(
indices : array_like of int, shape (m,), optional
    Rows to score, read in place; the output lines up with indices. Must be
    integers in [0, n).
num_threads : int, optional
    Worker threads; 1 by default, 0 uses every available CPU.)doc";

    py::class_<QuantQuery>(quant, "Query", kQueryDoc)
        .def(
            py::init(&QuantQuery::create),
            py::arg("rotated_query"),
            py::kw_only(),
            py::arg("metric"),
            py::arg("centroid") = py::none(),
            kQueryDoc
        )
        .def_property_readonly(
            "dim", &QuantQuery::dim, R"doc(Working dimension d of the query.)doc"
        )
        .def_property_readonly(
            "metric", &QuantQuery::metric, R"doc(The metric, 'l2' or 'ip'.)doc"
        )
        .def(
            "set_centroid",
            &QuantQuery::set_centroid,
            py::arg("centroid"),
            R"doc(Switch to the factors of another centroid; None is zero.

Parameters
----------
centroid : array_like of float32, shape (d,) or None
    The working-space centroid the next factors were computed with.)doc"
        )
        .def(
            "estimate_bitset",
            &QuantQuery::estimate_bitset,
            py::arg("bitsets"),
            py::arg("factors"),
            py::kw_only(),
            py::arg("indices") = py::none(),
            py::arg("num_threads") = 1,
            py::arg("out") = py::none(),
            (std::string(
                 R"doc(Estimate distances from packed 1-bit codes with the float query.

Parameters
----------
bitsets : numpy.ndarray of uint8, shape (n, d // 8)
    pack_codes(prefix, nbits=1) output.
factors : numpy.ndarray of FACTORS_DTYPE, shape (n,)
    The 1-bit (prefix) factors.)doc"
             ) +
             kIndicesDoc + R"doc(
out : numpy.ndarray of ESTIMATE_DTYPE, optional

Returns
-------
estimates : numpy.ndarray of ESTIMATE_DTYPE)doc")
                .c_str()
        )
        .def(
            "estimate_codes",
            &QuantQuery::estimate_codes,
            py::arg("codes"),
            py::arg("factors"),
            py::kw_only(),
            py::arg("nbits"),
            py::arg("indices") = py::none(),
            py::arg("num_threads") = 1,
            py::arg("out") = py::none(),
            (std::string(R"doc(Estimate distances from unpacked codes of any width.

Parameters
----------
codes : numpy.ndarray of uint8 (nbits 1 to 8) or uint16 (nbits 9), shape (n, d)
factors : numpy.ndarray of FACTORS_DTYPE, shape (n,)
    The factors of these nbits-bit codes.
nbits : int
    Bit width of the codes, 1 to 9.)doc") +
             kIndicesDoc + R"doc(
out : numpy.ndarray of ESTIMATE_DTYPE, optional

Returns
-------
estimates : numpy.ndarray of ESTIMATE_DTYPE)doc")
                .c_str()
        )
        .def(
            "refine",
            &QuantQuery::refine<false>,
            py::arg("ex"),
            py::arg("factors"),
            py::arg("first"),
            py::kw_only(),
            py::arg("ex_bits"),
            py::arg("prefix_bits") = 1,
            py::arg("indices") = py::none(),
            py::arg("num_threads") = 1,
            py::arg("out") = py::none(),
            (std::string(R"doc(Refine prefix estimates with the unpacked extra bits.

Parameters
----------
ex : numpy.ndarray of uint8, shape (n, d)
    The low ex_bits bits of each code, from split.
factors : numpy.ndarray of FACTORS_DTYPE, shape (n,)
    The full-code factors (prefix_bits + ex_bits bits).
first : numpy.ndarray of ESTIMATE_DTYPE
    The prefix estimates of the scored rows; only refine_input is read.
ex_bits : int
    Width of the extra bits, 1 to 9 - prefix_bits.
prefix_bits : int, optional
    Width of the prefix the first estimates scored, 1 by default.)doc") +
             kIndicesDoc + R"doc(
out : numpy.ndarray of float32, optional

Returns
-------
distances : numpy.ndarray of float32)doc")
                .c_str()
        )
        .def(
            "refine_packed",
            &QuantQuery::refine<true>,
            py::arg("packed"),
            py::arg("factors"),
            py::arg("first"),
            py::kw_only(),
            py::arg("ex_bits"),
            py::arg("prefix_bits") = 1,
            py::arg("indices") = py::none(),
            py::arg("num_threads") = 1,
            py::arg("out") = py::none(),
            (std::string(R"doc(Refine prefix estimates with packed extra bits.

Uses the dispatched SIMD kernel, so it agrees with refine within float32
rounding.

Parameters
----------
packed : numpy.ndarray of uint8, shape (n, d * ex_bits // 8)
    pack_codes(ex, nbits=ex_bits) output.
factors : numpy.ndarray of FACTORS_DTYPE, shape (n,)
    The full-code factors (prefix_bits + ex_bits bits).
first : numpy.ndarray of ESTIMATE_DTYPE
    The prefix estimates of the scored rows; only refine_input is read.
ex_bits : int
    Width of the extra bits, 1 to 9 - prefix_bits.
prefix_bits : int, optional
    Width of the prefix the first estimates scored, 1 by default.)doc") +
             kIndicesDoc + R"doc(
out : numpy.ndarray of float32, optional

Returns
-------
distances : numpy.ndarray of float32)doc")
                .c_str()
        )
        .def(
            "estimate_fastscan",
            &QuantQuery::estimate_fastscan,
            py::arg("blocks"),
            py::kw_only(),
            py::arg("num_points"),
            py::arg("num_threads") = 1,
            py::arg("out") = py::none(),
            R"doc(Estimate distances for FastScan blocks with a 4-bit query table.

The query table is built on the first call. Its quantization makes these
estimates agree with estimate_bitset only within that table's error.

Parameters
----------
blocks : numpy.ndarray of uint8, shape (ceil(num_points / 32), d * 4 + 384)
    pack_fastscan_blocks output.
num_points : int
    Number of vectors in the blocks; padding lanes are not returned.
num_threads : int, optional
    Worker threads; 1 by default, 0 uses every available CPU.
out : numpy.ndarray of ESTIMATE_DTYPE, shape (num_points,), optional

Returns
-------
estimates : numpy.ndarray of ESTIMATE_DTYPE, shape (num_points,))doc"
        )
        .def("copy", &QuantQuery::copy, "Return an independent copy of the query.")
        .def("__copy__", &QuantQuery::copy, "Return an independent copy of the query.")
        .def(
            "__deepcopy__",
            [](QuantQuery& self, const py::object& /*memo*/) { return self.copy(); },
            py::arg("memo"),
            "Return an independent copy of the query."
        )
        .def("__repr__", &QuantQuery::repr, "Return the query's parameters.")
        .def(
            py::pickle(
                [](QuantQuery& self) { return self.state(); },
                [](const py::tuple& state) { return QuantQuery::from_state(state); }
            )
        );
}
