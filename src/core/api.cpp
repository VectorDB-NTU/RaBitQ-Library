#include "rabitqlib/core/api.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/detail/rabitq_impl.hpp"
#include "rabitqlib/core/detail/scratch.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/pack_excode.hpp"
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::quant::v1 {
namespace {
constexpr size_t kWorkspaceAlignment = alignof(std::max_align_t);

void validate(const Parameters& parameters) {
    if (parameters.dim < 2) {
        throw std::invalid_argument("dim must be at least 2");
    }
    if (parameters.dim > static_cast<size_t>(std::numeric_limits<long>::max())) {
        throw std::length_error("dim exceeds the supported index range");
    }
    if (parameters.total_bits < 1 || parameters.total_bits > 9) {
        throw std::invalid_argument("total_bits must be in [1, 9]");
    }
    validate_metric_type(parameters.metric);
    if (!std::isfinite(parameters.t_const) ||
        parameters.t_const > std::numeric_limits<int>::max() / 2.0) {
        throw std::invalid_argument("t_const must be finite and at most INT_MAX / 2");
    }
}

size_t checked_multiply(size_t count, size_t width) {
    if (width == 0) {
        return 0;
    }
    if (count > std::numeric_limits<size_t>::max() / width) {
        throw std::length_error("Quantization buffer size overflows size_t");
    }
    return count * width;
}

[[noreturn]] void invalid_buffer(const char* message) {
    throw std::invalid_argument(message);
}

// All alignments are internal constants. Keep exception construction out of the
// valid-buffer path so callers can inline the checks and remove byte alignment work.
template <size_t Alignment = 1>
void check_buffer(ConstBuffer buffer, size_t required) {
    static_assert(Alignment > 0);
    if (required == 0) {
        return;
    }
    if (buffer.data == nullptr || buffer.bytes < required) {
        invalid_buffer("Buffer is null or smaller than the required size");
    }
    const auto address = reinterpret_cast<std::uintptr_t>(buffer.data);
    if (address % Alignment != 0) {
        invalid_buffer("Buffer does not satisfy the required alignment");
    }
    if (required > std::numeric_limits<std::uintptr_t>::max() - address) {
        invalid_buffer("Buffer address range overflows uintptr_t");
    }
}

void check_disjoint(
    const void* first, size_t first_bytes, const void* second, size_t second_bytes
) {
    if (first_bytes == 0 || second_bytes == 0) {
        return;
    }
    const auto first_address = reinterpret_cast<std::uintptr_t>(first);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second);
    if (first_address < second_address + second_bytes &&
        second_address < first_address + first_bytes) {
        throw std::invalid_argument(
            "Writable buffers must not overlap other buffers or inputs"
        );
    }
}

void check_input(const float* input, size_t dim) {
    check_buffer<alignof(float)>(
        {input, checked_multiply(dim, sizeof(float))}, checked_multiply(dim, sizeof(float))
    );
    if (!all_finite(input, dim)) {
        throw std::invalid_argument("Input vectors must contain only finite values");
    }
}

// Parameters and arithmetic ranges were validated by requirements/prepare_query.
std::pair<size_t, size_t> code_sizes(const Parameters& parameters, CodeLayout layout) {
    switch (layout) {
        case CodeLayout::UnpackedU16:
            return {parameters.dim * sizeof(uint16_t), 0};
        case CodeLayout::SplitPacked:
            if (parameters.dim % 32 != 0) {
                throw std::invalid_argument(
                    "SplitPacked requires dim to be a multiple of 32"
                );
            }
            return {parameters.dim / 8, (parameters.dim / 8) * (parameters.total_bits - 1)};
        default:
            throw std::invalid_argument("Unsupported code layout");
    }
}

void check_code(CodeView code, const Requirements& required) {
    check_buffer(code.codes, required.code_bytes);
    check_buffer(code.extra, required.extra_bytes);
}

CodeView view(CodeOutput code) {
    return {
        code.layout,
        {code.codes.data, code.codes.bytes},
        {code.extra.data, code.extra.bytes}};
}

void check_output(CodeOutput code, const Requirements& required, Buffer workspace) {
    check_code(view(code), required);
    check_buffer<kWorkspaceAlignment>(
        {workspace.data, workspace.bytes}, required.workspace_bytes
    );
    check_disjoint(
        code.codes.data, required.code_bytes, code.extra.data, required.extra_bytes
    );
    check_disjoint(
        code.codes.data, required.code_bytes, workspace.data, required.workspace_bytes
    );
    check_disjoint(
        code.extra.data, required.extra_bytes, workspace.data, required.workspace_bytes
    );
}

void check_input_overlap(
    const float* input,
    size_t dim,
    CodeOutput output,
    const Requirements& required,
    Buffer workspace
) {
    if (input == nullptr) {
        return;
    }
    const size_t bytes = checked_multiply(dim, sizeof(float));
    check_disjoint(input, bytes, output.codes.data, required.code_bytes);
    check_disjoint(input, bytes, output.extra.data, required.extra_bytes);
    check_disjoint(input, bytes, workspace.data, required.workspace_bytes);
}

// One typed allocation list serves both requirements() and actual storage binding.
struct ScratchStorage {
    rabitq_impl::Scratch<float> scratch;
    uint16_t* codes;
    uint8_t* extra;
    float* zero;

    explicit ScratchStorage(rabitq_impl::ScratchAllocator& allocator)
        : scratch{
              allocator.array<float>(),
              allocator.array<float>(),
              allocator.array<float>(),
              allocator.array<int>(),
              allocator.array<int>(),
              allocator.array<double>(),
              allocator.array<double>(),
              allocator.resource()}
        , codes(allocator.array<uint16_t>())
        , extra(allocator.array<uint8_t>())
        , zero(allocator.array<float>()) {}
};

uint16_t read_u16(const uint8_t* bytes) {
    return static_cast<uint16_t>(bytes[0] | (uint16_t{bytes[1]} << 8U));
}

// The i-th UnpackedU16 code, rejected if it does not fit the configured bit width.
uint16_t checked_unpacked_code(const uint8_t* bytes, size_t i, uint16_t maximum) {
    const uint16_t value = read_u16(bytes + 2 * i);
    if (value > maximum) {
        throw std::invalid_argument("Unpacked code exceeds the configured bit width");
    }
    return value;
}

void read_codes(const Parameters& parameters, CodeView input, ScratchStorage& storage) {
    const auto* bytes = static_cast<const uint8_t*>(input.codes.data);
    if (input.layout == CodeLayout::UnpackedU16) {
        const auto maximum = static_cast<uint16_t>((1U << parameters.total_bits) - 1);
        for (size_t i = 0; i < parameters.dim; ++i) {
            storage.codes[i] = checked_unpacked_code(bytes, i, maximum);
        }
        return;
    }
    rabitq_impl::unpack_sign_bits(bytes, parameters.dim, storage.codes);
    const size_t ex_bits = parameters.total_bits - 1;
    if (ex_bits > 0) {
        rabitq_impl::ex_bits::unpacking_rabitqplus_code(
            static_cast<const uint8_t*>(input.extra.data),
            storage.extra,
            parameters.dim,
            ex_bits
        );
        for (size_t i = 0; i < parameters.dim; ++i) {
            storage.codes[i] =
                static_cast<uint16_t>((storage.codes[i] << ex_bits) | storage.extra[i]);
        }
    }
}

void write_codes(const Parameters& parameters, CodeOutput output, ScratchStorage& storage) {
    auto* bytes = static_cast<uint8_t*>(output.codes.data);
    if (output.layout == CodeLayout::UnpackedU16) {
        for (size_t i = 0; i < parameters.dim; ++i) {
            bytes[2 * i] = static_cast<uint8_t>(storage.codes[i]);
            bytes[2 * i + 1] = static_cast<uint8_t>(storage.codes[i] >> 8U);
        }
        return;
    }
    const size_t ex_bits = parameters.total_bits - 1;
    const auto mask = static_cast<uint16_t>((1U << ex_bits) - 1);
    for (size_t i = 0; i < parameters.dim; ++i) {
        storage.scratch.binary[i] = storage.codes[i] >> ex_bits;
        storage.extra[i] = static_cast<uint8_t>(storage.codes[i] & mask);
    }
    pack_binary_to_bytes<uint64_t>(storage.scratch.binary, bytes, parameters.dim);
    if (ex_bits > 0) {
        rabitq_impl::ex_bits::packing_rabitqplus_code(
            storage.extra, static_cast<uint8_t*>(output.extra.data), parameters.dim, ex_bits
        );
    }
}
}  // namespace

Requirements requirements(const Parameters& parameters, CodeLayout layout) {
    validate(parameters);
    // Derive the budget once from the same declarations used by encoding and
    // exact fallback. No array is allocated in this sizing pass.
    static const auto kScratchBudget = [] {
        rabitq_impl::ScratchAllocator sizing;
        [[maybe_unused]] const ScratchStorage arrays(sizing);
        (void)rabitq_impl::event_sweep_buffers(sizing);
        return sizing;
    }();
    Requirements required{0, 0, kScratchBudget.bytes(parameters.dim), kWorkspaceAlignment};
    const auto [code_bytes, extra_bytes] = code_sizes(parameters, layout);
    required.code_bytes = code_bytes;
    required.extra_bytes = extra_bytes;
    return required;
}

Factors encode(
    const Parameters& parameters,
    const float* input,
    const float* centroid,
    CodeOutput output,
    Buffer workspace
) {
    const Requirements required = requirements(parameters, output.layout);
    check_input(input, parameters.dim);
    if (centroid != nullptr) {
        check_input(centroid, parameters.dim);
    }
    check_output(output, required, workspace);
    check_input_overlap(input, parameters.dim, output, required, workspace);
    check_input_overlap(centroid, parameters.dim, output, required, workspace);
    std::pmr::monotonic_buffer_resource resource(
        workspace.data, required.workspace_bytes, std::pmr::null_memory_resource()
    );
    rabitq_impl::ScratchAllocator allocator(&resource, parameters.dim);
    ScratchStorage storage(allocator);
    if (centroid == nullptr) {
        std::fill_n(storage.zero, parameters.dim, 0.0F);
        centroid = storage.zero;
    }
    // Reject overflow before the quantizer's float-to-integer conversions.
    RowMajorArrayMap<float> residual(
        storage.scratch.residual, 1, static_cast<Eigen::Index>(parameters.dim)
    );
    residual =
        ConstRowMajorArrayMap<float>(input, 1, static_cast<Eigen::Index>(parameters.dim)) -
        ConstRowMajorArrayMap<float>(
            centroid, 1, static_cast<Eigen::Index>(parameters.dim)
        );
    const float residual_squared_norm =
        l2norm_sqr(storage.scratch.residual, parameters.dim);
    if (!std::isfinite(residual_squared_norm)) {
        throw std::overflow_error("Residual squared norm is not representable in float32");
    }
    Factors factors;
    if (parameters.total_bits > 1) {
        rabitq_impl::ex_bits::code_with_factor_from_residual<true>(
            storage.scratch.residual,
            residual_squared_norm,
            centroid,
            parameters.dim,
            parameters.total_bits - 1,
            storage.codes,
            factors.add,
            factors.rescale,
            factors.error,
            parameters.metric,
            parameters.t_const,
            &storage.scratch
        );
    } else {
        rabitq_impl::total_bits::rabitq_full_impl_with_scratch(
            input,
            centroid,
            parameters.dim,
            parameters.total_bits,
            storage.codes,
            factors.add,
            factors.rescale,
            factors.error,
            parameters.metric,
            parameters.t_const,
            &storage.scratch
        );
    }
    if (!std::isfinite(factors.add) || !std::isfinite(factors.rescale) ||
        !std::isfinite(factors.error)) {
        throw std::overflow_error("Quantization factors are not representable in float32");
    }
    write_codes(parameters, output, storage);
    return factors;
}

void convert_codes(
    const Parameters& parameters, CodeView source, CodeOutput destination, Buffer workspace
) {
    const Requirements source_required = requirements(parameters, source.layout);
    const Requirements destination_required = requirements(parameters, destination.layout);
    check_code(source, source_required);
    check_output(destination, destination_required, workspace);
    for (const auto& range :
         {std::pair<const void*, size_t>{source.codes.data, source_required.code_bytes},
          {source.extra.data, source_required.extra_bytes}}) {
        check_disjoint(
            range.first,
            range.second,
            destination.codes.data,
            destination_required.code_bytes
        );
        check_disjoint(
            range.first,
            range.second,
            destination.extra.data,
            destination_required.extra_bytes
        );
        check_disjoint(
            range.first, range.second, workspace.data, destination_required.workspace_bytes
        );
    }
    std::pmr::monotonic_buffer_resource resource(
        workspace.data,
        destination_required.workspace_bytes,
        std::pmr::null_memory_resource()
    );
    rabitq_impl::ScratchAllocator allocator(&resource, parameters.dim);
    ScratchStorage storage(allocator);
    read_codes(parameters, source, storage);
    write_codes(parameters, destination, storage);
}

QueryState prepare_query(
    const Parameters& parameters, const float* query, const float* centroid
) {
    (void)requirements(parameters, CodeLayout::UnpackedU16);
    check_input(query, parameters.dim);
    if (centroid != nullptr) {
        check_input(centroid, parameters.dim);
    }
    QueryState state;
    state.query_ = query;
    state.dim_ = parameters.dim;
    state.total_bits_ = parameters.total_bits;
    state.metric_ = parameters.metric;
    state.extra_ip_ = select_excode_ipfunc(parameters.total_bits - 1);
    state.k1xsumq_ = std::accumulate(query, query + parameters.dim, 0.0F) * -0.5F;
    state.g_add_ =
        parameters.metric == METRIC_L2
            ? (centroid == nullptr ? l2norm_sqr(query, parameters.dim)
                                   : euclidean_sqr(query, centroid, parameters.dim))
            : (centroid == nullptr ? 0.0F : -dot_product(query, centroid, parameters.dim));
    if (!std::isfinite(state.g_add_) || !std::isfinite(state.k1xsumq_)) {
        throw std::overflow_error("Query factors are not representable in float32");
    }
    return state;
}

float QueryState::compute_checked(
    const QueryState& query_wrapper, const CodeView& code, const Factors& factors
) {
    if (query_wrapper.query_ == nullptr) {
        throw std::invalid_argument("QueryState has not been prepared");
    }
    const Parameters parameters{
        query_wrapper.dim_, query_wrapper.total_bits_, query_wrapper.metric_};
    const auto [code_bytes, extra_bytes] = code_sizes(parameters, code.layout);
    check_buffer(code.codes, code_bytes);
    check_buffer(code.extra, extra_bytes);
    if (!std::isfinite(factors.add) || !std::isfinite(factors.rescale) ||
        !std::isfinite(factors.error)) {
        throw std::invalid_argument("Quantization factors must be finite");
    }
    float ip = 0;
    if (code.layout == CodeLayout::UnpackedU16) {
        const auto* bytes = static_cast<const uint8_t*>(code.codes.data);
        const auto maximum = static_cast<uint16_t>((1U << parameters.total_bits) - 1);
        for (size_t i = 0; i < parameters.dim; ++i) {
            ip += query_wrapper.query_[i] *
                  static_cast<float>(checked_unpacked_code(bytes, i, maximum));
        }
    } else {
        const size_t ex_bits = parameters.total_bits - 1;
        ip = static_cast<float>(1U << ex_bits) *
             mask_ip_x0_q(
                 query_wrapper.query_,
                 static_cast<const uint8_t*>(code.codes.data),
                 parameters.dim
             );
        if (ex_bits > 0) {
            ip += query_wrapper.extra_ip_(
                query_wrapper.query_,
                static_cast<const uint8_t*>(code.extra.data),
                parameters.dim
            );
        }
    }
    const float distance = quant::detail::full_est_dist_from_ip(
        ip,
        parameters.total_bits,
        factors.add,
        factors.rescale,
        query_wrapper.g_add_,
        query_wrapper.k1xsumq_
    );
    if (!std::isfinite(distance)) {
        throw std::overflow_error("Estimated distance is not representable in float32");
    }
    return distance;
}
}  // namespace rabitqlib::quant::v1
