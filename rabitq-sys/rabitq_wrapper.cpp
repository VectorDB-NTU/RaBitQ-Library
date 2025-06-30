#include "rabitq.h"
#include "../rabitqlib/quantization/rabitq.hpp"
#include "../rabitqlib/utils/rotator.hpp"
#include "../rabitqlib/index/estimator.hpp"
#include "../rabitqlib/index/query.hpp"

// 在 C++ 文件中，我们知道 Rotator 是 rabitqlib::Rotator<float> 的别名
// 但这个细节对 C 和 Rust 是隐藏的

extern "C" {

RabitqConfig* rabitq_config_new() {
    return reinterpret_cast<RabitqConfig*>(new rabitqlib::quant::RabitqConfig());
}

void rabitq_config_free(RabitqConfig* config) {
    delete reinterpret_cast<rabitqlib::quant::RabitqConfig*>(config);
}

RabitqConfig* rabitq_faster_config(size_t dim, size_t total_bits) {
    auto config = rabitqlib::quant::faster_config(dim, total_bits);
    return reinterpret_cast<RabitqConfig*>(new rabitqlib::quant::RabitqConfig(config));
}

Rotator* rabitq_rotator_new(size_t dim, size_t padded_dim) {
    auto rotator = new rabitqlib::rotator_impl::FhtKacRotator(dim, padded_dim);
    return reinterpret_cast<Rotator*>(rotator);
}

void rabitq_rotator_free(Rotator* rotator) {
    delete reinterpret_cast<rabitqlib::Rotator<float>*>(rotator);
}

void rabitq_rotator_rotate(const Rotator* rotator, const float* x, float* y) {
    reinterpret_cast<const rabitqlib::Rotator<float>*>(rotator)->rotate(x, y);
}

int rabitq_rotator_load(Rotator* rotator, const char* file_path) {
    std::ifstream input(file_path, std::ios::binary);
    if (!input.is_open()) {
        return -1;
    }
    reinterpret_cast<rabitqlib::Rotator<float>*>(rotator)->load(input);
    return 0;
}

int rabitq_rotator_save(const Rotator* rotator, const char* file_path) {
    std::ofstream output(file_path, std::ios::binary);
    if (!output.is_open()) {
        return -1;
    }
    reinterpret_cast<const rabitqlib::Rotator<float>*>(rotator)->save(output);
    return 0;
}

size_t rabitq_rotator_size(const Rotator* rotator) {
    return reinterpret_cast<const rabitqlib::Rotator<float>*>(rotator)->size();
}

size_t rabitq_rotator_dim(const Rotator* rotator) {
    return reinterpret_cast<const rabitqlib::Rotator<float>*>(rotator)->dim();
}

void rabitq_quantize_full_single(
    const float* data,
    size_t dim,
    size_t total_bits,
    uint8_t* total_code,
    float* f_add,
    float* f_rescale,
    float* f_error,
    MetricType metric_type,
    const RabitqConfig* config
) {
    rabitqlib::quant::quantize_full_single(
        data,
        dim,
        total_bits,
        total_code,
        *f_add,
        *f_rescale,
        *f_error,
        static_cast<rabitqlib::MetricType>(metric_type),
        *reinterpret_cast<const rabitqlib::quant::RabitqConfig*>(config)
    );
}

void rabitq_quantize_split_single(
    const float* data,
    const float* centroid,
    size_t padded_dim,
    size_t ex_bits,
    char* bin_data,
    char* ex_data,
    enum MetricType metric_type,
    const RabitqConfig* config
) {
    rabitqlib::quant::quantize_split_single(
        data,
        centroid,
        padded_dim,
        ex_bits,
        bin_data,
        ex_data,
        static_cast<rabitqlib::MetricType>(metric_type),
        *reinterpret_cast<const rabitqlib::quant::RabitqConfig*>(config)
    );
}

void rabitq_reconstruct_vec(
    const uint8_t* quantized_vec, 
    float delta, 
    float vl, 
    size_t dim, 
    float* results
) {
    rabitqlib::quant::reconstruct_vec(quantized_vec, delta, vl, dim, results);
}

SplitBatchQuery* rabitq_split_batch_query_new(
    const float* rotated_query,
    size_t padded_dim,
    size_t ex_bits,
    MetricType metric_type,
    bool use_hacc) {
    auto q = new rabitqlib::SplitBatchQuery<float>(
        rotated_query, padded_dim, ex_bits, static_cast<rabitqlib::MetricType>(metric_type), use_hacc);
    return reinterpret_cast<SplitBatchQuery*>(q);
}

void rabitq_split_batch_query_free(SplitBatchQuery* q_obj) {
    delete reinterpret_cast<rabitqlib::SplitBatchQuery<float>*>(q_obj);
}

void rabitq_split_batch_query_set_g_add(SplitBatchQuery* q_obj, float norm, float ip) {
    reinterpret_cast<rabitqlib::SplitBatchQuery<float>*>(q_obj)->set_g_add(norm, ip);
}

void rabitq_split_batch_estdist(
    const char* batch_data,
    const SplitBatchQuery* q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc) {
    const auto* cpp_q_obj = reinterpret_cast<const rabitqlib::SplitBatchQuery<float>*>(q_obj);
    rabitqlib::split_batch_estdist(
        batch_data, *cpp_q_obj, padded_dim, est_distance, low_distance, ip_x0_qr, use_hacc);
}

float rabitq_split_distance_boosting_with_batch_query(
    const char* ex_data,
    float (*ip_func_)(const float*, const uint8_t*, size_t),
    const SplitBatchQuery* q_obj,
    size_t padded_dim,
    size_t ex_bits,
    float ip_x0_qr) {
    const auto* cpp_q_obj = reinterpret_cast<const rabitqlib::SplitBatchQuery<float>*>(q_obj);
    return rabitqlib::split_distance_boosting(
        ex_data, ip_func_, *cpp_q_obj, padded_dim, ex_bits, ip_x0_qr);
}

}