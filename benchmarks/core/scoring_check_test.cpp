#include <initializer_list>
#include <limits>

#include "scoring_legacy.hpp"

int main() {
    using scoring_benchmark::scores_match;
    if (!scores_match(1, 1, 0) || !scores_match(1, 1.125F, 0.125) ||
        !scores_match(1.125F, 1, 0.125) || scores_match(1, 1.25F, 0.125) ||
        scores_match(1, 1, -1)) {
        return 1;
    }
    for (float invalid :
         {std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()}) {
        if (scores_match(invalid, 1, 1) || scores_match(1, invalid, 1) ||
            scores_match(invalid, invalid, 1) || scores_match(1, 1, invalid)) {
            return 1;
        }
    }
    return 0;
}
