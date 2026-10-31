#pragma once
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

#include "rabitqlib/index/detail/visited_set.hpp"

namespace rabitqlib {
class VisitedSetPool {
    std::deque<std::unique_ptr<VisitedSet>> pool_;
    std::mutex pool_mutex_;
    size_t num_elements_;

   public:
    VisitedSetPool(size_t initial_pool_size, size_t num_elements)
        : num_elements_(num_elements) {
        for (size_t i = 0; i < initial_pool_size; ++i) {
            pool_.push_front(
                std::make_unique<VisitedSet>(num_elements_, num_elements_ / 10)
            );
        }
    }

    std::unique_ptr<VisitedSet> acquire() {
        std::unique_ptr<VisitedSet> visited;
        {
            std::unique_lock<std::mutex> lock(pool_mutex_);
            if (!pool_.empty()) {
                visited = std::move(pool_.front());
                pool_.pop_front();
            } else {
                visited = std::make_unique<VisitedSet>(num_elements_, num_elements_ / 10);
            }
        }
        visited->clear();
        return visited;
    }

    void release(std::unique_ptr<VisitedSet> visited) {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        pool_.push_front(std::move(visited));
    }
};
}  // namespace rabitqlib
