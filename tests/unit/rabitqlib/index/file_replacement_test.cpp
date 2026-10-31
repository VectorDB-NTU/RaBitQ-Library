#include "file_replacement.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
namespace fs = std::filesystem;
using rabitqlib::index_detail::save_replacing;

class FileReplacementTest : public ::testing::Test {
   protected:
    fs::path directory_;
    std::string path_;

    void SetUp() override {
        directory_ = fs::path(::testing::TempDir()) /
                     (std::string("rabitq_file_replacement_") +
                      ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(directory_);
        fs::create_directories(directory_);
        path_ = (directory_ / "out.index").string();
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    static void write_text(const std::string& path, const std::string& text) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << text;
        if (!output) {
            throw std::runtime_error("cannot write test file");
        }
    }

    static std::string read_text(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    [[nodiscard]] size_t num_entries() const {
        return static_cast<size_t>(
            std::distance(fs::directory_iterator(directory_), fs::directory_iterator())
        );
    }
};

TEST_F(FileReplacementTest, ReplacesFileAndSidecarWithoutLeftovers) {
    write_text(path_, "old");
    write_text(path_ + ".hnsw", "old sidecar");
    std::string written_path;
    save_replacing(path_, {".hnsw"}, [&](const std::string& path) {
        written_path = path;
        write_text(path, "new");
        write_text(path + ".hnsw", "new sidecar");
    });
    EXPECT_NE(written_path, path_);
    EXPECT_EQ(read_text(path_), "new");
    EXPECT_EQ(read_text(path_ + ".hnsw"), "new sidecar");
    EXPECT_EQ(num_entries(), 2U);
}

TEST_F(FileReplacementTest, KeepsPermissionsOfReplacedFiles) {
#ifdef _WIN32
    GTEST_SKIP() << "Windows exposes only a read-only attribute";
#endif
    constexpr auto kMain =
        fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read;
    constexpr auto kSidecar = fs::perms::owner_read | fs::perms::owner_write;
    write_text(path_, "old");
    write_text(path_ + ".hnsw", "old sidecar");
    fs::permissions(path_, kMain);
    fs::permissions(path_ + ".hnsw", kSidecar);
    save_replacing(path_, {".hnsw"}, [](const std::string& path) {
        write_text(path, "new");
        write_text(path + ".hnsw", "new sidecar");
    });
    EXPECT_EQ(read_text(path_), "new");
    EXPECT_EQ(fs::status(path_).permissions(), kMain);
    EXPECT_EQ(fs::status(path_ + ".hnsw").permissions(), kSidecar);
}

TEST_F(FileReplacementTest, CreatesMissingDestination) {
    save_replacing(path_, {}, [](const std::string& path) { write_text(path, "new"); });
    EXPECT_EQ(read_text(path_), "new");
    EXPECT_EQ(num_entries(), 1U);
}

TEST_F(FileReplacementTest, FailedWriteKeepsFilesAndRemovesTemporaries) {
    write_text(path_, "old");
    write_text(path_ + ".hnsw", "old sidecar");
    try {
        save_replacing(path_, {".hnsw"}, [](const std::string& path) {
            write_text(path, "partial");
            write_text(path + ".hnsw", "partial sidecar");
            throw std::runtime_error("injected write failure");
        });
        FAIL() << "the write failure must propagate";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "injected write failure");
    }
    EXPECT_EQ(read_text(path_), "old");
    EXPECT_EQ(read_text(path_ + ".hnsw"), "old sidecar");
    EXPECT_EQ(num_entries(), 2U);
}

TEST_F(FileReplacementTest, WritesThroughNonRegularDestinationInPlace) {
    // A directory sidecar stands in for devices and FIFOs: nothing may be renamed onto it.
    fs::create_directory(path_ + ".hnsw");
    std::string written_path;
    EXPECT_THROW(
        save_replacing(
            path_,
            {".hnsw"},
            [&](const std::string& path) {
                written_path = path;
                throw std::runtime_error("cannot open sidecar");
            }
        ),
        std::runtime_error
    );
    EXPECT_EQ(written_path, path_);
    EXPECT_TRUE(fs::is_directory(path_ + ".hnsw"));
    EXPECT_EQ(num_entries(), 1U);
}

TEST_F(FileReplacementTest, WritesThroughSymbolicLinkInPlace) {
    const std::string target = (directory_ / "target.index").string();
    write_text(target, "old");
    std::error_code error;
    fs::create_symlink(target, path_, error);
    if (error) {
        GTEST_SKIP() << "symbolic links are unavailable: " << error.message();
    }
    save_replacing(path_, {}, [](const std::string& path) { write_text(path, "new"); });
    EXPECT_TRUE(fs::is_symlink(path_));
    EXPECT_EQ(read_text(target), "new");
    EXPECT_EQ(num_entries(), 2U);
}
}  // namespace
