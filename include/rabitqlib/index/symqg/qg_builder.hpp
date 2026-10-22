#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "rabitqlib/index/detail/visited_set.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::symqg {
constexpr size_t kMaxBsIter = 5;  // max iter for binary search of pruning bar
using CandidateList = std::vector<AnnCandidate<float>>;
enum class QGInitialization { PiPNN, Random };

// Owns temporary state for SymphonyQG initialization and refinement.
class QGBuilder {
    friend struct QGConstructionTestAccess;

   private:
    // Owned by QGBuilder and never persisted.
    struct BuildCache {
        std::vector<float> rotated_vectors;
        std::vector<QuantizedQuery> queries;
    };

    // Reset before scoring. The graph and optional cache must outlive this query.
    // Owns reconstruction/rotation scratch only when no cache can supply it.
    class BuildQuery {
        const QuantizedGraph<float>& graph_;
        const BuildCache* cache_;
        std::vector<float> storage_;
        const float* raw_ = nullptr;
        const float* rotated_ = nullptr;
        std::optional<QuantizedQuery> prepared_;

       public:
        explicit BuildQuery(
            const QuantizedGraph<float>& graph, const BuildCache* cache = nullptr
        );
        // Prepared pointers can borrow storage_; never copy their owner.
        BuildQuery(const BuildQuery&) = delete;
        BuildQuery& operator=(const BuildQuery&) = delete;

        void reset(PID);
        [[nodiscard]] float distance_to(PID) const;
        // Raw scoring needs no rotation. Prepare it lazily for FastScan only.
        [[nodiscard]] const float* rotated_query();
    };

    QuantizedGraph<float>& qg_;
    size_t ef_build_;  // size of search pool for indexing
    size_t num_threads_;
    size_t num_nodes_;
    size_t dim_;
    size_t degree_bound_;  // degree bound for qg, multiple of 32
    bool seeded_ = true;
    static constexpr size_t kMaxCandidatePoolSize = 750;
    static constexpr size_t kMaxPrunedSize = 300;
    std::vector<CandidateList> new_neighbors_;  // new neighbors for current iteration
    std::vector<uint32_t> degrees_;
    uint32_t seed_;  // base seed for graph initialization
    std::optional<BuildCache> cache_;
    std::vector<CandidateList> reverse_buffer_;
    struct BuildWorker {
        VisitedSet visited;
        CandidateList candidates;
        CandidateList pruned;
        CandidateList completion_result;
        CandidateList successful_result;
        std::vector<float> rotated_neighbors;
        std::vector<PID> neighbor_ids;

        BuildWorker(size_t num_nodes, size_t visited_capacity);
    };
    std::vector<BuildWorker> workers_;

    const BuildCache* build_cache() const;
    void find_candidates(PID, CandidateList&, VisitedSet&) const;
    void random_init();
    void search_new_neighbors();
    void heuristic_prune(CandidateList&, CandidateList&, CandidateList* recorded);
    void add_reverse_edges(bool final_pass);
    void add_pruned_edges(
        const CandidateList&, const CandidateList&, CandidateList&, float
    );
    void complete_degree(PID, BuildWorker&);
    void run_pass(bool final_pass);
    void run_passes(size_t);
    void write_row(PID, BuildWorker&);
    void write_row(PID, const CandidateList&, BuildWorker&);

    void initialize_storage(const float* data);

    void initialize_seed(
        const float* data,
        const std::vector<size_t>& offsets,
        const std::vector<PID>& neighbors
    );

    QGBuilder(
        QuantizedGraph<float>& index,
        uint32_t ef_build,
        size_t num_threads,
        uint32_t seed = std::random_device{}(),
        bool cache_vectors = false
    );

   public:
    explicit QGBuilder(
        QuantizedGraph<float>& index,
        uint32_t ef_build,
        const float* data,
        size_t num_threads = std::numeric_limits<size_t>::max(),
        QGInitialization init = QGInitialization::PiPNN,
        uint32_t seed = std::random_device{}(),
        bool cache_vectors = false
    );

    // Preserve the seeded random-initialization overload used before PiPNN.
    QGBuilder(
        QuantizedGraph<float>& index,
        uint32_t ef_build,
        const float* data,
        size_t num_threads,
        uint32_t seed,
        bool cache_vectors = false
    );

    /** Replace vectors and restart the selected initializer, retaining workspace.
     * Cached vectors and queries are owned; input may be released after reset.
     */
    void reset(const float* data);

    // One complete search/prune/reverse-edge/degree-completion iteration. Call
    // before serving a seeded graph: search() requires full final rows.
    void refine();

    // One refinement for PiPNN initialization, three passes for random init.
    void build();

    void build(size_t num_iter);

    [[nodiscard]] bool check_dup() const;

    [[nodiscard]] float avg_degree() const;
};

}  // namespace rabitqlib::symqg
