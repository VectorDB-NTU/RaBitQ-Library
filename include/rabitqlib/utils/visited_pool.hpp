#pragma once
#include <deque>
#include <mutex>

#include "rabitqlib/utils/visited_set.hpp"

namespace rabitqlib {
class VisitedListPool {
    std::deque<VisitedSet*> pool_;
    std::mutex poolguard_;
    size_t numelements_;

   public:
    VisitedListPool(size_t initpoolsize, size_t max_elements) {
        numelements_ = max_elements;
        for (size_t i = 0; i < initpoolsize; i++) {
            pool_.push_front(new VisitedSet(numelements_, numelements_ / 10));
        }
    }

    VisitedSet* get_free_vislist() {
        VisitedSet* rez;
        {
            std::unique_lock<std::mutex> lock(poolguard_);
            if (pool_.size() > 0) {
                rez = pool_.front();
                pool_.pop_front();
            } else {
                rez = new VisitedSet(numelements_, numelements_ / 10);
            }
        }
        rez->clear();
        return rez;
    }

    void release_vis_list(VisitedSet* vl) {
        std::unique_lock<std::mutex> lock(poolguard_);
        pool_.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool_.size() > 0) {
            VisitedSet* rez = pool_.front();
            pool_.pop_front();
            ::delete rez;
        }
    }
};
}  // namespace rabitqlib
