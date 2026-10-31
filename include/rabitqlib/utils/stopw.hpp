#pragma once

#include <chrono>
#include <ratio>

namespace rabitqlib {
class StopW {
    using Clock = std::chrono::steady_clock;
    Clock::time_point time_begin_;

    [[nodiscard]] Clock::duration elapsed() const { return Clock::now() - time_begin_; }

   public:
    StopW() : time_begin_(Clock::now()) {}

    void reset() { time_begin_ = Clock::now(); }

    [[nodiscard]] float get_elapsed_sec() const {
        return std::chrono::duration<float>(elapsed()).count();
    }

    [[nodiscard]] float get_elapsed_mili() const {
        return std::chrono::duration<float, std::milli>(elapsed()).count();
    }

    [[nodiscard]] float get_elapsed_micro() const {
        return std::chrono::duration<float, std::micro>(elapsed()).count();
    }

    [[nodiscard]] float get_elapsed_nano() const {
        return std::chrono::duration<float, std::nano>(elapsed()).count();
    }
};
}  // namespace rabitqlib