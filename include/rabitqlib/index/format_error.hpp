#pragma once

#include <stdexcept>

namespace rabitqlib {

/**
 * @brief An index file whose contents are invalid, or which this release cannot read.
 *
 * Loading any index throws it for a damaged, mismatched or unsupported file in every
 * layout; failing to open the file is a plain std::runtime_error. Python raises it as
 * rabitqlib.FormatError, a subclass of both ValueError and RuntimeError.
 */
class FormatError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

}  // namespace rabitqlib
