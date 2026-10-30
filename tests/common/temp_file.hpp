#ifndef RABITQ_TEST_TEMP_FILE_HPP
#define RABITQ_TEST_TEMP_FILE_HPP

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <system_error>

namespace rabitq_test {

// A file under the GoogleTest temporary directory that is removed, together with an IVF
// HNSW sidecar written next to it, when the test leaves its scope, including after a
// failed assertion. Leftovers from an interrupted earlier run are removed up front.
class TempFile {
   public:
    explicit TempFile(const std::string& name) : path_(::testing::TempDir() + name) {
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
