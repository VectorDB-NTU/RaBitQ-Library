#pragma once
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

#include "rabitqlib/index/detail/visited_set.hpp"

namespace rabitqlib {
class VisitedListPool {
    std::deque<std::unique_ptr<VisitedSet>> pool_;
    std::mutex poolguard_;
    size_t numelements_;

   public:
    VisitedListPool(size_t initpoolsize, size_t max_elements) {
        numelements_ = max_elements;
        for (size_t i = 0; i < initpoolsize; i++) {
            pool_.push_front(std::make_unique<VisitedSet>(numelements_, numelements_ / 10));
        }
    }

    std::unique_ptr<VisitedSet> get_free_vislist() {
        std::unique_ptr<VisitedSet> rez;
        {
            std::unique_lock<std::mutex> lock(poolguard_);
            if (pool_.size() > 0) {
                rez = std::move(pool_.front());
                pool_.pop_front();
            } else {
                rez = std::make_unique<VisitedSet>(numelements_, numelements_ / 10);
            }
        }
        rez->clear();
        return rez;
    }

    void release_vis_list(std::unique_ptr<VisitedSet> vl) {
        std::unique_lock<std::mutex> lock(poolguard_);
        pool_.push_front(std::move(vl));
    }

    ~VisitedListPool() = default;
};
}  // namespace rabitqlib
