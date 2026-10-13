#include "scoring_legacy.hpp"

#include "rabitqlib/core/api.hpp"

namespace scoring_benchmark {
float unchecked_call(const Query& query, api::CodeView code, const api::Factors& f) {
    return unchecked_inline(query, code, f);
}
float unchecked_ref(const Query& query, const api::CodeView& code, const api::Factors& f) {
    return unchecked_inline(query, code, f);
}
}  // namespace scoring_benchmark
