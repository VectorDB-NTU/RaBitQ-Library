#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include "rabitqlib/utils/openmp.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::index_detail {
template <class Function>
inline void parallel_for(size_t start, size_t end, size_t num_threads, Function fn) {
    if (start >= end) {
        return;
    }
    num_threads = std::min(resolve_num_threads(num_threads), end - start);

    if (num_threads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
#if defined(_OPENMP) && _OPENMP >= 201307
        if (omp_get_proc_bind() != omp_proc_bind_false) {
            // Native threads inherit only the caller's place. Let OpenMP honor
            // the configured placement of the whole team when binding is enabled.
            std::vector<std::exception_ptr> errors(num_threads);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (size_t id = start; id < end; ++id) {
                const size_t worker = static_cast<size_t>(omp_get_thread_num());
                if (!errors[worker]) {
                    try {
                        fn(id, worker);
                    } catch (...) { errors[worker] = std::current_exception(); }
                }
            }
            for (const auto& error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
            return;
        }
#endif
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr last_exception = nullptr;
        std::mutex last_except_mutex;

        const auto join_threads = [&threads] {
            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        };

        try {
            threads.reserve(num_threads);
            for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
                threads.emplace_back([&, thread_id] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if (id >= end) {
                            break;
                        }

                        try {
                            fn(id, thread_id);
                        } catch (...) {
                            std::unique_lock<std::mutex> last_except_lock(
                                last_except_mutex
                            );
                            last_exception = std::current_exception();
                            /*
                             * This will work even when current is the largest value that
                             * size_t can fit, because fetch_add returns the previous value
                             * before the increment (what will result in overflow
                             * and produce 0 instead of current + 1).
                             */
                            current = end;
                            break;
                        }
                    }
                });
            }
        } catch (...) {
            current = end;
            join_threads();
            throw;
        }
        join_threads();
        if (last_exception) {
            std::rethrow_exception(last_exception);
        }
    }
}

}  // namespace rabitqlib::index_detail
