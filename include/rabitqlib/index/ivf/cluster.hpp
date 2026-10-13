#pragma once

#include <cstddef>

#include "rabitqlib/index/types.hpp"

namespace rabitqlib::ivf {

// Non-owning view of one IVF cluster's codes, reranking data and point IDs.
class Cluster {
   private:
    size_t num_;                  // Num of vectors in this cluster
    char* batch_data_ = nullptr;  // RaBitQ code and factors
    char* ex_data_ = nullptr;     // Extra-bit codes/factors or original float32 vectors
    PID* ids_ = nullptr;          // PID of vectors

   public:
    explicit Cluster(size_t, char*, char*, PID*);
    Cluster(const Cluster&) = default;
    Cluster(Cluster&&) noexcept = default;
    Cluster& operator=(const Cluster&) = delete;
    Cluster& operator=(Cluster&&) = delete;
    ~Cluster() = default;

    [[nodiscard]] char* batch_data() const { return this->batch_data_; }

    [[nodiscard]] char* ex_data() const { return ex_data_; }

    [[nodiscard]] PID* ids() const { return this->ids_; }

    [[nodiscard]] size_t num() const { return num_; }
};

inline Cluster::Cluster(size_t num, char* batch_data, char* ex_data, PID* ids)
    : num_(num), batch_data_(batch_data), ex_data_(ex_data), ids_(ids) {}

}  // namespace rabitqlib::ivf
