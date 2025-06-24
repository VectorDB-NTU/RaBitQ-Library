#ifndef RABITQ_H
#define RABITQ_H

#include <stddef.h>
#include <stdint.h>

// 使用 typedef 创建一个不透明的结构体类型
// 这样 C 代码就不知道 Rotator 的内部结构了
typedef struct Rotator Rotator;
typedef struct RabitqConfig RabitqConfig;


#ifdef __cplusplus
extern "C" {
#endif

RabitqConfig* rabitq_config_new();
void rabitq_config_free(RabitqConfig* config);

// C API 使用不透明指针 Rotator*
Rotator* rabitq_rotator_new(size_t dim, size_t padded_dim);
void rabitq_rotator_free(Rotator* rotator);
void rabitq_rotator_rotate(const Rotator* rotator, const float* x, float* y);

#ifdef __cplusplus
}
#endif

#endif // RABITQ_H