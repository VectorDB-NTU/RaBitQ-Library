#pragma once

#include <atomic>
#include <exception>

namespace rabitqlib::index_detail {

// Shared by one OpenMP team. Capture inside each worksharing iteration and
// rethrow only after the team has joined. Successful iterations never lock.
class ParallelException {
   public:
    bool failed() const noexcept { return failed_.load(std::memory_order_relaxed); }

    void capture() noexcept {
        if (!failed_.exchange(true, std::memory_order_relaxed)) {
            error_ = std::current_exception();
        }
    }

    void rethrow() const {
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

   private:
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
};

}  // namespace rabitqlib::index_detail
