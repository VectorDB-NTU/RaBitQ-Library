#pragma once

#include <filesystem>
#include <string_view>

namespace rabitqlib::io_impl {

// Windows API strings are UTF-8; POSIX API strings retain their native bytes.
inline std::filesystem::path filesystem_path(std::string_view filename) {
#if defined(_WIN32)
    return std::filesystem::u8path(filename);
#else
    return std::filesystem::path(filename);
#endif
}

}  // namespace rabitqlib::io_impl
