#ifndef RABITQ_H
#define RABITQ_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct for RabitqConfig
typedef struct RabitqConfig RabitqConfig;

// Opaque struct for Rotator
typedef struct Rotator Rotator;

enum MetricType {
    METRIC_L2,
    METRIC_IP
};

typedef struct SplitBatchQuery SplitBatchQuery;

RabitqConfig* rabitq_config_new();
void rabitq_config_free(RabitqConfig* config);
RabitqConfig* rabitq_faster_config(size_t dim, size_t total_bits);


Rotator* rabitq_rotator_new(size_t dim, size_t padded_dim);
void rabitq_rotator_free(Rotator* rotator);
void rabitq_rotator_rotate(const Rotator* rotator, const float* x, float* y);

int rabitq_rotator_load(Rotator* rotator, const char* file_path);
int rabitq_rotator_save(const Rotator* rotator, const char* file_path);
size_t rabitq_rotator_size(const Rotator* rotator);
size_t rabitq_rotator_dim(const Rotator* rotator);

void rabitq_quantize_full_single(
    const float* data,
    size_t dim,
    size_t total_bits,
    uint8_t* total_code,
    float* f_add,
    float* f_rescale,
    float* f_error,
    enum MetricType metric_type,
    const RabitqConfig* config
);

void rabitq_reconstruct_vec(
    const uint8_t* quantized_vec, 
    float delta, 
    float vl, 
    size_t dim, 
    float* results
);

void rabitq_quantize_split_single(
    const float* data,
    const float* centroid,
    size_t padded_dim,
    size_t ex_bits,
    char* bin_data,
    char* ex_data,
    enum MetricType metric_type,
    const RabitqConfig* config
);

SplitBatchQuery* rabitq_split_batch_query_new(
    const float* rotated_query,
    size_t padded_dim,
    size_t ex_bits,
    enum MetricType metric_type,
    bool use_hacc
);

void rabitq_split_batch_query_free(SplitBatchQuery* q_obj);

void rabitq_split_batch_query_set_g_add(SplitBatchQuery* q_obj, float norm, float ip);

void rabitq_split_batch_estdist(
    const char* batch_data,
    const SplitBatchQuery* q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc
);

float rabitq_split_distance_boosting_with_batch_query(
    const char* ex_data,
    float (*ip_func_)(const float*, const uint8_t*, size_t),
    const SplitBatchQuery* q_obj,
    size_t padded_dim,
    size_t ex_bits,
    float ip_x0_qr
);

#ifdef __cplusplus
}
#endif

#endif // RABITQ_H