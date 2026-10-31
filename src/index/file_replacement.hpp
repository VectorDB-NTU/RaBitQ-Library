#pragma once

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <random>
#include <string>
#include <system_error>

#include "rabitqlib/utils/path.hpp"

namespace rabitqlib::index_detail {

// Only a missing or regular destination can be replaced by renaming a sibling onto it.
// Devices, FIFOs, directories and symbolic links keep being written through in place.
inline bool replaceable_by_rename(const std::string& filename) {
    std::error_code error;
    const auto type =
        std::filesystem::symlink_status(io_impl::filesystem_path(filename), error).type();
    return type == std::filesystem::file_type::not_found ||
           type == std::filesystem::file_type::regular;
}

// Give a replacement the permission bits of the regular file it replaces. Ownership and
// hard links are not carried over; a file system that cannot change permissions keeps
// the replacement's defaults rather than failing the save.
inline void copy_permissions(const std::string& existing, const std::string& replacement) {
    std::error_code error;
    const auto status = std::filesystem::status(io_impl::filesystem_path(existing), error);
    if (error || status.type() != std::filesystem::file_type::regular) {
        return;
    }
    std::filesystem::permissions(
        io_impl::filesystem_path(replacement),
        status.permissions(),
        std::filesystem::perm_options::replace,
        error
    );
}

// Save through a temporary sibling that replaces `filename` only after `write` succeeds,
// so a failed save leaves any existing file unchanged and removes what it wrote.
// A replaced file keeps its permission bits, but not its owner or other hard links.
//
// `write(path)` writes the main file at `path` and each sidecar at `path + suffix`.
// Sidecars are renamed into place before the main file. If any destination exists but
// is not a regular file, every file is written in place instead, as before.
template <class Write>
void save_replacing(
    const std::string& filename,
    std::initializer_list<const char*> sidecar_suffixes,
    Write write
) {
    bool in_place = !replaceable_by_rename(filename);
    for (const char* suffix : sidecar_suffixes) {
        in_place = in_place || !replaceable_by_rename(filename + suffix);
    }
    if (in_place) {
        write(filename);
        return;
    }

    // A random name keeps concurrent saves to one destination from sharing a temporary.
    std::random_device random;
    const uint64_t token = (static_cast<uint64_t>(random()) << 32U) ^ random();
    const std::string temporary = filename + ".tmp" + std::to_string(token);
    try {
        write(temporary);
        copy_permissions(filename, temporary);
        for (const char* suffix : sidecar_suffixes) {
            copy_permissions(filename + suffix, temporary + suffix);
        }
        for (const char* suffix : sidecar_suffixes) {
            std::filesystem::rename(
                io_impl::filesystem_path(temporary + suffix),
                io_impl::filesystem_path(filename + suffix)
            );
        }
        std::filesystem::rename(
            io_impl::filesystem_path(temporary), io_impl::filesystem_path(filename)
        );
    } catch (...) {
        try {
            std::error_code ignored;
            std::filesystem::remove(io_impl::filesystem_path(temporary), ignored);
            for (const char* suffix : sidecar_suffixes) {
                std::filesystem::remove(
                    io_impl::filesystem_path(temporary + suffix), ignored
                );
            }
        } catch (...) {  // NOLINT(bugprone-empty-catch): report the save failure instead.
        }
        throw;
    }
}

}  // namespace rabitqlib::index_detail
