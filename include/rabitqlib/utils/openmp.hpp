#pragma once

// CMake targets export an explicit choice. Direct header consumers without a
// target definition infer it from the compiler; match the linked library's setting.
#ifndef RABITQ_USE_OPENMP
#ifdef _OPENMP
#define RABITQ_USE_OPENMP 1
#else
#define RABITQ_USE_OPENMP 0
#endif
#endif

#if RABITQ_USE_OPENMP
#include <omp.h>  // IWYU pragma: export
#endif
