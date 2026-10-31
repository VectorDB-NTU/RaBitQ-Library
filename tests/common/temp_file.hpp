#ifndef RABITQ_TEST_TEMP_FILE_HPP
#define RABITQ_TEST_TEMP_FILE_HPP

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace rabitq_test {

// Prefix a temporary file name with the process ID so suites running at the same
// time in one temporary directory cannot delete or overwrite each other's files.
inline std::string process_unique(const std::string& name) {
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = static_cast<int>(getpid());
#endif
    return "rabitq-" + std::to_string(pid) + "-" + name;
}

// A file under the GoogleTest temporary directory that is removed, together with an IVF
// HNSW sidecar written next to it, when the test leaves its scope, including after a
// failed assertion. Leftovers from an interrupted earlier run of this process are
// removed up front.
class TempFile {
   public:
    explicit TempFile(const std::string& name)
        : path_(::testing::TempDir() + process_unique(name)) {
        remove();
    }
    ~TempFile() { remove(); }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    [[nodiscard]] const std::string& path() const { return path_; }

   private:
    std::string path_;

    void remove() const noexcept {
        std::error_code ignored;
        std::filesystem::remove(std::filesystem::u8path(path_), ignored);
        std::filesystem::remove(std::filesystem::u8path(path_ + ".hnsw"), ignored);
    }
};

}  // namespace rabitq_test

#endif  // RABITQ_TEST_TEMP_FILE_HPP
