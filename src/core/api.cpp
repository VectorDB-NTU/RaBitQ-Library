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
#include "rabitqlib/utils/bitops.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::quant::v1 {
namespace {
constexpr size_t kWorkspaceAlignment = alignof(std::max_align_t);

void validate(const Parameters& p) {
    if (p.dim < 2) {
        throw std::invalid_argument("dim must be at least 2");
    }
    if (p.dim > static_cast<size_t>(std::numeric_limits<long>::max())) {
        throw std::length_error("dim exceeds the supported index range");
    }
    if (p.total_bits < 1 || p.total_bits > 9) {
        throw std::invalid_argument("total_bits must be in [1, 9]");
    }
    validate_metric_type(p.metric);
    if (!std::isfinite(p.t_const) || p.t_const > std::numeric_limits<int>::max() / 2.0) {
        throw std::invalid_argument("t_const must be finite and at most INT_MAX / 2");
    }
}

size_t multiply(size_t count, size_t width) {
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

void disjoint(const void* a, size_t a_size, const void* b, size_t b_size) {
    if (a_size == 0 || b_size == 0) {
        return;
    }
    const auto first = reinterpret_cast<std::uintptr_t>(a);
    const auto second = reinterpret_cast<std::uintptr_t>(b);
    if (first < second + b_size && second < first + a_size) {
        throw std::invalid_argument(
            "Writable buffers must not overlap other buffers or inputs"
        );
    }
}

void check_input(const float* input, size_t dim) {
    check_buffer<alignof(float)>(
        {input, multiply(dim, sizeof(float))}, multiply(dim, sizeof(float))
    );
    for (size_t i = 0; i < dim; ++i) {
        if (!std::isfinite(input[i])) {
            throw std::invalid_argument("Input vectors must contain only finite values");
        }
    }
}

// Parameters and arithmetic ranges were validated by requirements/prepare_query.
std::pair<size_t, size_t> code_sizes(const Parameters& p, CodeLayout layout) {
    switch (layout) {
        case CodeLayout::UnpackedU16:
            return {p.dim * sizeof(uint16_t), 0};
        case CodeLayout::SplitPacked:
            if (p.dim % 32 != 0) {
                throw std::invalid_argument(
                    "SplitPacked requires dim to be a multiple of 32"
                );
            }
            return {p.dim / 8, (p.dim / 8) * (p.total_bits - 1)};
        default:
            throw std::invalid_argument("Unsupported code layout");
    }
}

void check_code(CodeView code, const Requirements& r) {
    check_buffer(code.codes, r.code_bytes);
    check_buffer(code.extra, r.extra_bytes);
}

CodeView view(CodeOutput code) {
    return {
        code.layout,
        {code.codes.data, code.codes.bytes},
        {code.extra.data, code.extra.bytes}};
}

void check_output(CodeOutput code, const Requirements& r, Buffer workspace) {
    check_code(view(code), r);
    check_buffer<kWorkspaceAlignment>({workspace.data, workspace.bytes}, r.workspace_bytes);
    disjoint(code.codes.data, r.code_bytes, code.extra.data, r.extra_bytes);
    disjoint(code.codes.data, r.code_bytes, workspace.data, r.workspace_bytes);
    disjoint(code.extra.data, r.extra_bytes, workspace.data, r.workspace_bytes);
}

void check_input_overlap(
    const float* input,
    size_t dim,
    CodeOutput output,
    const Requirements& r,
    Buffer workspace
) {
    if (input == nullptr) {
        return;
    }
    const size_t bytes = multiply(dim, sizeof(float));
    disjoint(input, bytes, output.codes.data, r.code_bytes);
    disjoint(input, bytes, output.extra.data, r.extra_bytes);
    disjoint(input, bytes, workspace.data, r.workspace_bytes);
}

// One typed allocation list serves both requirements() and actual storage binding.
class ScratchStorage {
   public:
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

void read_codes(const Parameters& p, CodeView input, ScratchStorage& s) {
    const auto* bytes = static_cast<const uint8_t*>(input.codes.data);
    if (input.layout == CodeLayout::UnpackedU16) {
        const auto maximum = static_cast<uint16_t>((1U << p.total_bits) - 1);
        for (size_t i = 0; i < p.dim; ++i) {
            s.codes[i] = read_u16(bytes + 2 * i);
            if (s.codes[i] > maximum) {
                throw std::invalid_argument(
                    "Unpacked code exceeds the configured bit width"
                );
            }
        }
        return;
    }
    const size_t ex_bits = p.total_bits - 1;
    if (ex_bits > 0) {
        rabitq_impl::ex_bits::unpacking_rabitqplus_code(
            static_cast<const uint8_t*>(input.extra.data), s.extra, p.dim, ex_bits
        );
    }
    // Sign words retain the existing full-64 / final-32 coordinate convention.
    for (size_t start = 0; start < p.dim; start += 64) {
        const size_t width = std::min(size_t{64}, p.dim - start);
        const uint64_t word = bitops::load_word(bytes + start / 8, width / 8);
        for (size_t i = 0; i < width; ++i) {
            const auto sign = static_cast<uint16_t>((word >> (width - 1 - i)) & 1U);
            s.codes[start + i] = static_cast<uint16_t>(
                (sign << ex_bits) | (ex_bits == 0 ? 0 : s.extra[start + i])
            );
        }
    }
}

void write_codes(const Parameters& p, CodeOutput output, ScratchStorage& s) {
    auto* bytes = static_cast<uint8_t*>(output.codes.data);
    if (output.layout == CodeLayout::UnpackedU16) {
        for (size_t i = 0; i < p.dim; ++i) {
            bytes[2 * i] = static_cast<uint8_t>(s.codes[i]);
            bytes[2 * i + 1] = static_cast<uint8_t>(s.codes[i] >> 8U);
        }
        return;
    }
    const size_t ex_bits = p.total_bits - 1;
    const auto mask = static_cast<uint16_t>((1U << ex_bits) - 1);
    for (size_t i = 0; i < p.dim; ++i) {
        s.scratch.binary[i] = s.codes[i] >> ex_bits;
        s.extra[i] = static_cast<uint8_t>(s.codes[i] & mask);
    }
    pack_binary_to_bytes<uint64_t>(s.scratch.binary, bytes, p.dim);
    if (ex_bits > 0) {
        rabitq_impl::ex_bits::packing_rabitqplus_code(
            s.extra, static_cast<uint8_t*>(output.extra.data), p.dim, ex_bits
        );
    }
}
}  // namespace

Requirements requirements(const Parameters& p, CodeLayout layout) {
    validate(p);
    // Derive the budget once from the same declarations used by encoding and
    // exact fallback. No array is allocated in this sizing pass.
    static const auto budget = [] {
        rabitq_impl::ScratchAllocator sizing;
        [[maybe_unused]] const ScratchStorage arrays(sizing);
        (void)rabitq_impl::event_sweep_buffers(sizing);
        return sizing;
    }();
    Requirements r{0, 0, budget.bytes(p.dim), kWorkspaceAlignment};
    const auto [code_bytes, extra_bytes] = code_sizes(p, layout);
    r.code_bytes = code_bytes;
    r.extra_bytes = extra_bytes;
    return r;
}

Factors encode(
    const Parameters& p,
    const float* input,
    const float* centroid,
    CodeOutput output,
    Buffer workspace
) {
    const Requirements r = requirements(p, output.layout);
    check_input(input, p.dim);
    if (centroid != nullptr) {
        check_input(centroid, p.dim);
    }
    check_output(output, r, workspace);
    check_input_overlap(input, p.dim, output, r, workspace);
    check_input_overlap(centroid, p.dim, output, r, workspace);
    std::pmr::monotonic_buffer_resource resource(
        workspace.data, r.workspace_bytes, std::pmr::null_memory_resource()
    );
    rabitq_impl::ScratchAllocator allocator(&resource, p.dim);
    ScratchStorage s(allocator);
    if (centroid == nullptr) {
        std::fill_n(s.zero, p.dim, 0.0F);
        centroid = s.zero;
    }
    // Reject overflow before the quantizer's float-to-integer conversions.
    RowMajorArrayMap<float> residual(s.scratch.residual, 1, static_cast<long>(p.dim));
    residual = ConstRowMajorArrayMap<float>(input, 1, static_cast<long>(p.dim)) -
               ConstRowMajorArrayMap<float>(centroid, 1, static_cast<long>(p.dim));
    const float residual_squared_norm = l2norm_sqr(s.scratch.residual, p.dim);
    if (!std::isfinite(residual_squared_norm)) {
        throw std::overflow_error("Residual squared norm is not representable in float32");
    }
    Factors factors;
    if (p.total_bits > 1) {
        rabitq_impl::ex_bits::code_with_factor_from_residual<true>(
            s.scratch.residual,
            residual_squared_norm,
            centroid,
            p.dim,
            p.total_bits - 1,
            s.codes,
            factors.add,
            factors.rescale,
            factors.error,
            p.metric,
            p.t_const,
            &s.scratch
        );
    } else {
        rabitq_impl::total_bits::rabitq_full_impl_with_scratch(
            input,
            centroid,
            p.dim,
            p.total_bits,
            s.codes,
            factors.add,
            factors.rescale,
            factors.error,
            p.metric,
            p.t_const,
            &s.scratch
        );
    }
    if (!std::isfinite(factors.add) || !std::isfinite(factors.rescale) ||
        !std::isfinite(factors.error)) {
        throw std::overflow_error("Quantization factors are not representable in float32");
    }
    write_codes(p, output, s);
    return factors;
}

void convert_codes(
    const Parameters& p, CodeView source, CodeOutput destination, Buffer workspace
) {
    const Requirements input = requirements(p, source.layout);
    const Requirements output = requirements(p, destination.layout);
    check_code(source, input);
    check_output(destination, output, workspace);
    for (const auto& range :
         {std::pair<const void*, size_t>{source.codes.data, input.code_bytes},
          {source.extra.data, input.extra_bytes}}) {
        disjoint(range.first, range.second, destination.codes.data, output.code_bytes);
        disjoint(range.first, range.second, destination.extra.data, output.extra_bytes);
        disjoint(range.first, range.second, workspace.data, output.workspace_bytes);
    }
    std::pmr::monotonic_buffer_resource resource(
        workspace.data, output.workspace_bytes, std::pmr::null_memory_resource()
    );
    rabitq_impl::ScratchAllocator allocator(&resource, p.dim);
    ScratchStorage s(allocator);
    read_codes(p, source, s);
    write_codes(p, destination, s);
}

QueryState prepare_query(const Parameters& p, const float* query, const float* centroid) {
    (void)requirements(p, CodeLayout::UnpackedU16);
    check_input(query, p.dim);
    if (centroid != nullptr) {
        check_input(centroid, p.dim);
    }
    QueryState state;
    state.query_ = query;
    state.dim_ = p.dim;
    state.total_bits_ = p.total_bits;
    state.metric_ = p.metric;
    state.extra_ip_ = select_excode_ipfunc(p.total_bits - 1);
    state.k1xsumq_ = std::accumulate(query, query + p.dim, 0.0F) * -0.5F;
    state.g_add_ =
        p.metric == METRIC_L2
            ? (centroid == nullptr ? l2norm_sqr(query, p.dim)
                                   : euclidean_sqr(query, centroid, p.dim))
            : (centroid == nullptr ? 0.0F : -dot_product(query, centroid, p.dim));
    if (!std::isfinite(state.g_add_) || !std::isfinite(state.k1xsumq_)) {
        throw std::overflow_error("Query factors are not representable in float32");
    }
    return state;
}

float QueryState::compute_checked(
    const QueryState& query, const CodeView& code, const Factors& factors
) {
    if (query.query_ == nullptr) {
        throw std::invalid_argument("QueryState has not been prepared");
    }
    const Parameters p{query.dim_, query.total_bits_, query.metric_};
    const auto [code_bytes, extra_bytes] = code_sizes(p, code.layout);
    check_buffer(code.codes, code_bytes);
    check_buffer(code.extra, extra_bytes);
    if (!std::isfinite(factors.add) || !std::isfinite(factors.rescale) ||
        !std::isfinite(factors.error)) {
        throw std::invalid_argument("Quantization factors must be finite");
    }
    float ip = 0;
    if (code.layout == CodeLayout::UnpackedU16) {
        const auto* bytes = static_cast<const uint8_t*>(code.codes.data);
        const auto maximum = static_cast<uint16_t>((1U << p.total_bits) - 1);
        for (size_t i = 0; i < p.dim; ++i) {
            const uint16_t value = read_u16(bytes + 2 * i);
            if (value > maximum) {
                throw std::invalid_argument(
                    "Unpacked code exceeds the configured bit width"
                );
            }
            ip += query.query_[i] * static_cast<float>(value);
        }
    } else {
        const size_t ex_bits = p.total_bits - 1;
        ip =
            static_cast<float>(1U << ex_bits) *
            mask_ip_x0_q(query.query_, static_cast<const uint8_t*>(code.codes.data), p.dim);
        if (ex_bits > 0) {
            ip += query.extra_ip_(
                query.query_, static_cast<const uint8_t*>(code.extra.data), p.dim
            );
        }
    }
    const float distance = quant::detail::full_est_dist_from_ip(
        ip, p.total_bits, factors.add, factors.rescale, query.g_add_, query.k1xsumq_
    );
    if (!std::isfinite(distance)) {
        throw std::overflow_error("Estimated distance is not representable in float32");
    }
    return distance;
}
}  // namespace rabitqlib::quant::v1
