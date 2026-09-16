#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/ivf/cluster.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::ivf {
namespace detail {
void insert_candidates(buffer::SearchBuffer<float>&, const PID*, const float*, size_t);
}  // namespace detail

class IVF {
   private:
    using ByteStorage =
        std::vector<std::byte, memory::AlignedAllocator<std::byte, 64, true>>;
    using FloatStorage = std::vector<float, memory::AlignedAllocator<float, 64, true>>;
    using IdStorage = std::vector<PID, memory::AlignedAllocator<PID, 64, true>>;

    std::unique_ptr<Initializer> initer_;            // initializer for candidate clusters
    ByteStorage batch_storage_;                      // 1-bit code and factors
    ByteStorage ex_storage_;                         // extra-bit codes and packed factors
    FloatStorage raw_storage_;                       // original vectors for raw reranking
    IdStorage id_storage_;                           // point IDs organized by cluster
    size_t num_ = 0;                                 // num of data points
    size_t dim_ = 0;                                 // dimension of data points
    size_t padded_dim_ = 0;                          // dimension after padding
    size_t num_cluster_ = 0;                         // num of centroids (clusters)
    bool raw_reranking_ = false;                     // raw vectors replace extra-bit codes
    bool ready_ = false;                             // construction or loading completed
    size_t ex_bits_ = 0;                             // total bits = ex_bits_ + 1
    RotatorType type_ = RotatorType::FhtKacRotator;  // type of rotator
    std::unique_ptr<Rotator<float>> rotator_;        // Data Rotator
    std::vector<Cluster> cluster_lst_;               // List of clusters in ivf
    MetricType metric_type_ = rabitqlib::METRIC_L2;  // metric type
    float (*ip_func_)(const float*, const uint8_t*, size_t) = nullptr;

    static constexpr uint64_t kRawFormatMagic = 0x3157415251464252ULL;
    static constexpr uint32_t kRawFormatVersion = 1;

    void
    quantize_cluster(Cluster&, const std::vector<PID>&, const float*, const float*, float*, const quant::RabitqConfig&);

    [[nodiscard]] size_t ids_bytes() const { return sizeof(PID) * num_; }

    // get num of bytes used for 1-bit code and corresponding factors
    [[nodiscard]] size_t batch_data_bytes(const std::vector<size_t>& cluster_sizes) const {
        assert(cluster_sizes.size() == num_cluster_);  // num of clusters
        size_t total_blocks = 0;
        for (auto size : cluster_sizes) {
            total_blocks += div_round_up(size, fastscan::kBatchSize);
        }
        return total_blocks * BatchDataMap<float>::data_bytes(padded_dim_);
    }

    [[nodiscard]] size_t rerank_vector_bytes() const {
        return raw_reranking_ ? sizeof(float) * dim_
                              : ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    }

    [[nodiscard]] size_t ex_data_bytes() const { return rerank_vector_bytes() * num_; }

    [[nodiscard]] char* batch_data() {
        return reinterpret_cast<char*>(batch_storage_.data());
    }

    [[nodiscard]] const char* batch_data() const {
        return reinterpret_cast<const char*>(batch_storage_.data());
    }

    [[nodiscard]] char* ex_data() {
        return raw_reranking_ ? reinterpret_cast<char*>(raw_storage_.data())
                              : reinterpret_cast<char*>(ex_storage_.data());
    }

    [[nodiscard]] const char* ex_data() const {
        return raw_reranking_ ? reinterpret_cast<const char*>(raw_storage_.data())
                              : reinterpret_cast<const char*>(ex_storage_.data());
    }

    [[nodiscard]] PID* ids() { return id_storage_.data(); }

    [[nodiscard]] const PID* ids() const { return id_storage_.data(); }

    void allocate_memory(const std::vector<size_t>&);

    void init_clusters(const std::vector<size_t>&);

    void free_memory() {
        initer_.reset();
        ByteStorage().swap(batch_storage_);
        ByteStorage().swap(ex_storage_);
        FloatStorage().swap(raw_storage_);
        IdStorage().swap(id_storage_);
    }

    void swap(IVF& other) noexcept {
        using std::swap;
        swap(initer_, other.initer_);
        swap(batch_storage_, other.batch_storage_);
        swap(ex_storage_, other.ex_storage_);
        swap(raw_storage_, other.raw_storage_);
        swap(id_storage_, other.id_storage_);
        swap(num_, other.num_);
        swap(dim_, other.dim_);
        swap(padded_dim_, other.padded_dim_);
        swap(num_cluster_, other.num_cluster_);
        swap(raw_reranking_, other.raw_reranking_);
        swap(ready_, other.ready_);
        swap(ex_bits_, other.ex_bits_);
        swap(type_, other.type_);
        swap(rotator_, other.rotator_);
        swap(cluster_lst_, other.cluster_lst_);
        swap(metric_type_, other.metric_type_);
        swap(ip_func_, other.ip_func_);
    }

    void
    search_cluster(const Cluster&, const SplitBatchQuery<float>&, buffer::SearchBuffer<float>&, bool, const float*)
        const;

    void
    scan_one_batch(const char* batch_data, const char* ex_data, const PID* ids, const SplitBatchQuery<float>& q_obj, buffer::SearchBuffer<float>& knns, size_t num_points, bool, const float*)
        const;

   public:
    explicit IVF() = default;
    explicit IVF(
        size_t,
        size_t,
        size_t,
        size_t,
        MetricType metric_type = rabitqlib::METRIC_L2,
        RotatorType type = RotatorType::FhtKacRotator
    );

    ~IVF();

    [[nodiscard]] size_t max_elements() const { return num_; }
    [[nodiscard]] size_t dimension() const { return dim_; }
    [[nodiscard]] size_t nbits() const { return raw_reranking_ ? 32 : ex_bits_ + 1; }
    [[nodiscard]] MetricType metric_type() const { return metric_type_; }
    [[nodiscard]] RotatorType rotator_type() const { return type_; }

    void construct(const float*, const float*, const PID*, bool, size_t);

    void save(const char*) const;

    void load(const char*);

    // Automatically use HACC for 4-9 quantized bits; use standard FastScan otherwise.
    void search(const float*, size_t, size_t, PID*, float* = nullptr) const;

    void search(const float*, size_t, size_t, PID*, bool) const;

    void search(const float*, size_t, size_t, PID*, float*, bool) const;

    [[nodiscard]] size_t padded_dim() const { return this->padded_dim_; }

    [[nodiscard]] size_t num_clusters() const { return this->num_cluster_; }
};

inline IVF::IVF(
    size_t n,
    size_t dim,
    size_t cluster_num,
    size_t bits,
    MetricType metric_type,
    RotatorType type
)
    : num_(n)
    , dim_(dim)
    , padded_dim_(dim)
    , num_cluster_(cluster_num)
    , raw_reranking_(bits == 32)
    , ex_bits_(bits == 32 ? 0 : bits - 1)
    , type_(type)
    , metric_type_(metric_type) {
    validate_metric_type(metric_type);
    if ((bits < 1 || bits > 9) && bits != 32) {
        throw std::invalid_argument("IVF bits must be in [1, 9] or 32 for raw reranking");
    };
    if (n == 0 || n > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF point count exceeds the supported ID range");
    }
    if (cluster_num == 0 || cluster_num > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF cluster count exceeds the supported ID range");
    }
    if (dim == 0 || dim > (std::numeric_limits<size_t>::max() / 32) - 64) {
        throw std::invalid_argument("IVF dimension is invalid or too large");
    }
    padded_dim_ = round_up_to_multiple(dim_, 64);
    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
    const size_t rerank_bytes = rerank_vector_bytes();
    if (n > max_size / sizeof(PID) || n > max_size / sizeof(float) / padded_dim_ ||
        n > max_size / batch_bytes || (rerank_bytes != 0 && n > max_size / rerank_bytes) ||
        cluster_num > max_size / sizeof(float) / padded_dim_ ||
        (type == RotatorType::MatrixRotator && dim > max_size / sizeof(float) / padded_dim_
        )) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }
    rotator_.reset(choose_rotator<float>(dim, type, padded_dim_));
    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);
}

inline IVF::~IVF() { free_memory(); }

/**
 * @brief Construct clusters in IVF
 *
 * @param data Data objects (N*DIM)
 * @param centroids Centroid vectors (K*DIM)
 * @param cluster_ids Cluster ID for each data object
 */
inline void IVF::construct(
    const float* data,
    const float* centroids,
    const PID* cluster_ids,
    bool faster = false,
    size_t num_threads = std::numeric_limits<size_t>::max()
) {
    if (num_ == 0 || dim_ == 0 || num_cluster_ == 0 || rotator_ == nullptr) {
        throw std::logic_error("IVF must be configured before construction");
    }
    if (data == nullptr || centroids == nullptr || cluster_ids == nullptr) {
        throw std::invalid_argument("IVF construction inputs must not be null");
    }

    // get id list for each cluster
    std::vector<size_t> counts(num_cluster_, 0);
    std::vector<std::vector<PID>> id_lists(num_cluster_);
    for (size_t i = 0; i < num_; ++i) {
        PID cid = cluster_ids[i];
        if (cid >= num_cluster_) {
            throw std::invalid_argument("Cluster ID is out of range");
        }
        id_lists[cid].push_back(static_cast<PID>(i));
        counts[cid] += 1;
    }

    allocate_memory(counts);

    // init the cluster list
    init_clusters(counts);

    // all rotated centroids
    std::vector<float> rotated_centroids(num_cluster_ * padded_dim_);

    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    num_threads = std::clamp(num_threads, size_t{1}, rabitqlib::total_threads());
    /* Quantize each cluster */
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t i = 0; i < num_cluster_; ++i) {
        const float* cur_centroid = centroids + (i * dim_);
        float* cur_rotated_c = &rotated_centroids[i * padded_dim_];
        Cluster& cp = cluster_lst_[i];
        quantize_cluster(cp, id_lists[i], data, cur_centroid, cur_rotated_c, config);
    }

    this->initer_->add_vectors(rotated_centroids.data(), num_threads);
    ready_ = true;
}

inline void IVF::allocate_memory(const std::vector<size_t>& cluster_sizes) {
    ready_ = false;
    free_memory();
    cluster_lst_.clear();

    if (num_cluster_ < 20000UL) {
        this->initer_ =
            std::make_unique<FlatInitializer>(padded_dim_, num_cluster_, metric_type_);
    } else {
        this->initer_ = std::make_unique<HNSWInitializer>(padded_dim_, num_cluster_);
    }
    batch_storage_ = ByteStorage(batch_data_bytes(cluster_sizes));
    if (rerank_vector_bytes() > 0) {
        if (raw_reranking_) {
            raw_storage_ = FloatStorage(num_ * dim_);
        } else {
            ex_storage_ = ByteStorage(ex_data_bytes());
        }
    }
    id_storage_ = IdStorage(num_);

    this->ip_func_ = select_excode_ipfunc(ex_bits_);
}

/**
 * @brief Initialize the cluster list by finding each cluster's storage offsets.
 */
inline void IVF::init_clusters(const std::vector<size_t>& cluster_sizes) {
    this->cluster_lst_.reserve(num_cluster_);
    size_t added_vectors = 0;
    size_t added_batches = 0;
    for (size_t i = 0; i < num_cluster_; ++i) {
        // find data location for current cluster
        size_t num = cluster_sizes[i];
        size_t num_batches = div_round_up(num, fastscan::kBatchSize);

        char* current_batch_data =
            batch_data() + (BatchDataMap<float>::data_bytes(padded_dim_) * added_batches);
        char* current_ex_data = rerank_vector_bytes() > 0
                                    ? ex_data() + (added_vectors * rerank_vector_bytes())
                                    : nullptr;
        PID* cluster_ids = ids() + added_vectors;

        Cluster cur_cluster(num, current_batch_data, current_ex_data, cluster_ids);
        this->cluster_lst_.push_back(std::move(cur_cluster));

        added_vectors += num;
        added_batches += num_batches;
    }
}

inline void IVF::quantize_cluster(
    Cluster& cp,
    const std::vector<PID>& IDs,
    const float* data,
    const float* cur_centroid,
    float* rotated_centroid,
    const quant::RabitqConfig& config
) {
    size_t num_points = IDs.size();
    if (cp.num() != num_points) {
        throw std::invalid_argument("Cluster size and ID count differ");
    }

    // copy ids
    std::copy(IDs.begin(), IDs.end(), cp.ids());

    // rotate centroid
    this->rotator_->rotate(cur_centroid, rotated_centroid);

    // rotate vectors for this cluster
    std::vector<float> rotated_data(padded_dim_ * num_points);
    for (size_t i = 0; i < num_points; ++i) {
        const float* vector = data + (IDs[i] * dim_);
        rotator_->rotate(vector, rotated_data.data() + (i * padded_dim_));
        if (raw_reranking_) {
            std::memcpy(
                cp.ex_data() + (i * rerank_vector_bytes()), vector, rerank_vector_bytes()
            );
        }
    }

    char* batch_data = cp.batch_data();
    char* ex_data = cp.ex_data();
    for (size_t i = 0; i < num_points; i += fastscan::kBatchSize) {
        size_t n = std::min(fastscan::kBatchSize, num_points - i);

        quant::quantize_split_batch(
            rotated_data.data() + (i * padded_dim_),
            rotated_centroid,
            n,
            padded_dim_,
            ex_bits_,
            batch_data,
            ex_data,
            metric_type_,
            config
        );

        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        if (ex_bits_ > 0) {
            ex_data += ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * n;
        }
    }
}

inline void IVF::save(const char* filename) const {
    if (!ready_ || initer_ == nullptr || rotator_ == nullptr ||
        cluster_lst_.size() != num_cluster_ || batch_storage_.empty() ||
        id_storage_.size() != num_) {
        throw std::logic_error("Cannot save an unconstructed IVF index");
    }
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("IVF save filename must not be empty");
    }

    std::ofstream output(filename, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    if (raw_reranking_) {
        output.write(
            reinterpret_cast<const char*>(&kRawFormatMagic), sizeof(kRawFormatMagic)
        );
        output.write(
            reinterpret_cast<const char*>(&kRawFormatVersion), sizeof(kRawFormatVersion)
        );
    }

    /* Save meta data */
    output.write(reinterpret_cast<const char*>(&num_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_cluster_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&type_), sizeof(type_));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(metric_type_));

    /* Save number of vectors of each cluster */
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_cluster_);
    for (const auto& cur_cluster : cluster_lst_) {
        cluster_sizes.push_back(cur_cluster.num());
    }
    output.write(
        reinterpret_cast<const char*>(cluster_sizes.data()),
        static_cast<long>(sizeof(size_t) * num_cluster_)
    );

    /* Save rotator */
    this->rotator_->save(output);

    /* Save data */
    this->initer_->save(output, filename);
    output.write(batch_data(), static_cast<long>(batch_data_bytes(cluster_sizes)));
    if (ex_data_bytes() != 0) {
        output.write(ex_data(), static_cast<long>(ex_data_bytes()));
    }
    output.write(reinterpret_cast<const char*>(ids()), static_cast<long>(ids_bytes()));

    output.close();
}

inline void IVF::load(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("IVF load filename must not be empty");
    }
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open IVF index file");
    }

    IVF loaded;
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.seekg(0, std::ios::end);
    const auto file_bytes = static_cast<size_t>(input.tellg());
    input.seekg(0);
    uint64_t magic = 0;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    loaded.raw_reranking_ = magic == kRawFormatMagic;
    if (loaded.raw_reranking_) {
        uint32_t version = 0;
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != kRawFormatVersion) {
            throw std::runtime_error("Unsupported raw IVF index version");
        }
    } else {
        input.seekg(0);
    }

    /* Load meta data */
    input.read(reinterpret_cast<char*>(&loaded.num_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.num_cluster_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.ex_bits_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.type_), sizeof(loaded.type_));
    input.read(reinterpret_cast<char*>(&loaded.metric_type_), sizeof(loaded.metric_type_));
    validate_metric_type(loaded.metric_type_);
    if (loaded.num_ == 0 || loaded.num_ > buffer::kSearchBufferMaxPointCount ||
        loaded.dim_ == 0 || loaded.num_cluster_ == 0 ||
        loaded.num_cluster_ > buffer::kSearchBufferMaxPointCount || loaded.ex_bits_ > 8 ||
        (loaded.raw_reranking_ && loaded.ex_bits_ != 0) ||
        loaded.dim_ > (std::numeric_limits<size_t>::max() / 32) - 64) {
        throw std::runtime_error("Invalid IVF index metadata");
    }
    loaded.padded_dim_ = round_up_to_multiple(loaded.dim_, 64);
    // Bound every payload by the actual file before allocating or multiplying sizes.
    size_t remaining = file_bytes - static_cast<size_t>(input.tellg());
    const auto consume = [&remaining](size_t count, size_t bytes) {
        if (bytes != 0 && count > remaining / bytes) {
            throw std::runtime_error("Invalid or truncated IVF index payload");
        }
        remaining -= count * bytes;
    };
    consume(loaded.num_cluster_, sizeof(size_t));
    consume(loaded.num_, sizeof(PID));
    consume(loaded.num_, loaded.rerank_vector_bytes());
    if (loaded.type_ == RotatorType::MatrixRotator) {
        consume(loaded.dim_, sizeof(float) * loaded.padded_dim_);
    } else if (loaded.type_ == RotatorType::FhtKacRotator) {
        consume(1, loaded.padded_dim_ / 2);
    } else {
        throw std::runtime_error("Invalid IVF rotator type");
    }
    if (loaded.num_cluster_ < 20000UL) {
        consume(loaded.num_cluster_, sizeof(float) * loaded.padded_dim_);
    }

    /* Load number of vectors of each cluster */
    std::vector<size_t> cluster_sizes(loaded.num_cluster_, 0);
    input.read(
        reinterpret_cast<char*>(cluster_sizes.data()),
        static_cast<long>(sizeof(size_t) * loaded.num_cluster_)
    );

    size_t total = 0;
    for (size_t size : cluster_sizes) {
        if (size > loaded.num_ - total) {
            throw std::runtime_error("Invalid cluster counts in IVF index file");
        }
        total += size;
        consume(
            div_round_up(size, fastscan::kBatchSize),
            BatchDataMap<float>::data_bytes(loaded.padded_dim_)
        );
    }
    if (total != loaded.num_) {
        throw std::runtime_error("Invalid cluster counts in IVF index file");
    }
    loaded.rotator_.reset(
        choose_rotator<float>(loaded.dim_, loaded.type_, loaded.padded_dim_)
    );

    /* Load rotator */
    loaded.rotator_->load(input);

    /* Load data */
    loaded.allocate_memory(cluster_sizes);
    loaded.initer_->load(input, filename);
    input.read(
        loaded.batch_data(), static_cast<long>(loaded.batch_data_bytes(cluster_sizes))
    );
    if (loaded.ex_data_bytes() != 0) {
        input.read(loaded.ex_data(), static_cast<long>(loaded.ex_data_bytes()));
    }
    input.read(
        reinterpret_cast<char*>(loaded.ids()), static_cast<long>(loaded.ids_bytes())
    );
    for (size_t i = 0; i < loaded.num_; ++i) {
        if (loaded.ids()[i] >= loaded.num_ ||
            loaded.ids()[i] >= buffer::kSearchBufferCheckedMask) {
            throw std::runtime_error("Invalid point ID in IVF index file");
        }
    }

    /* Init each cluster */
    loaded.init_clusters(cluster_sizes);
    loaded.ready_ = true;

    input.close();
    swap(loaded);
}

inline void IVF::search(
    const float* query, size_t k, size_t nprobe, PID* results, float* dists
) const {
    search(query, k, nprobe, results, dists, !raw_reranking_ && ex_bits_ > 2);
}

inline void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    bool use_hacc
) const {
    this->search(query, k, nprobe, results, nullptr, use_hacc);
}

inline void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    float* __restrict__ dists,
    bool use_hacc
) const {
    if (!ready_ || initer_ == nullptr || rotator_ == nullptr ||
        cluster_lst_.size() != num_cluster_ || batch_storage_.empty() ||
        id_storage_.size() != num_) {
        throw std::logic_error("IVF index must be constructed or loaded before search");
    }
    if (query == nullptr) {
        throw std::invalid_argument("IVF search query must not be null");
    }
    if (results == nullptr) {
        throw std::invalid_argument("IVF search results must not be null");
    }
    if (k == 0 || k > num_ || k > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF search k must be between 1 and the point count");
    }
    if (nprobe == 0) {
        throw std::invalid_argument("IVF search nprobe must be positive");
    }

    nprobe = std::min(nprobe, num_cluster_);  // corner case
    std::vector<float> rotated_query(padded_dim_);
    this->rotator_->rotate(query, rotated_query.data());

    // use initer to get closest nprobe centroids
    std::vector<AnnCandidate<float>> centroid_dist(nprobe);
    this->initer_->centroids_distances(rotated_query.data(), nprobe, centroid_dist);

    buffer::SearchBuffer knns(k);

    SplitBatchQuery<float> q_obj(
        rotated_query.data(), padded_dim_, ex_bits_, metric_type_, use_hacc
    );

    for (size_t i = 0; i < nprobe; ++i) {
        PID cid = centroid_dist[i].id;
        float dist = centroid_dist[i].distance;
        const Cluster& cur_cluster = cluster_lst_[cid];

        if (metric_type_ == METRIC_L2) {
            q_obj.set_g_add(dist);
        } else if (metric_type_ == METRIC_IP) {
            const float residual_norm = std::sqrt(
                euclidean_sqr(rotated_query.data(), initer_->centroid(cid), padded_dim_)
            );
            auto g_add_ip = dot_product<float>(
                rotated_query.data(), initer_->centroid(cid), padded_dim_
            );
            q_obj.set_g_add(residual_norm, g_add_ip);
        } else {
            // unsupported
            throw std::invalid_argument("Quantization only supports L2 and IP metrics");
        }
        // q_obj.set_g_add(dist);
        search_cluster(cur_cluster, q_obj, knns, use_hacc, query);
    }

    const size_t found = knns.size();
    if (dists != nullptr) {
        knns.copy_results(results, dists);
        std::fill(dists + found, dists + k, std::numeric_limits<float>::infinity());
    } else {
        knns.copy_results(results);
    }
    std::fill(results + found, results + k, kPidMax);
}

inline void IVF::search_cluster(
    const Cluster& cur_cluster,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    bool use_hacc,
    const float* query
) const {
    size_t iter = cur_cluster.num() / fastscan::kBatchSize;
    size_t remain = cur_cluster.num() - (iter * fastscan::kBatchSize);

    const char* batch_data = cur_cluster.batch_data();
    const char* ex_data = cur_cluster.ex_data();
    const PID* ids = cur_cluster.ids();

    /* Compute distances block by block */
    for (size_t i = 0; i < iter; ++i) {
        scan_one_batch(
            batch_data, ex_data, ids, q_obj, knns, fastscan::kBatchSize, use_hacc, query
        );

        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        if (rerank_vector_bytes() > 0) {
            ex_data += rerank_vector_bytes() * fastscan::kBatchSize;
        }
        ids += fastscan::kBatchSize;
    }

    if (remain > 0) {
        // scan the last block
        scan_one_batch(batch_data, ex_data, ids, q_obj, knns, remain, use_hacc, query);
    }
}

inline void IVF::scan_one_batch(
    const char* batch_data,
    const char* ex_data,
    const PID* ids,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    size_t num_points,
    bool use_hacc,
    const float* query
) const {
    std::array<float, fastscan::kBatchSize> est_distance;  // estimated distance
    std::array<float, fastscan::kBatchSize> low_distance;  // lower distance
    std::array<float, fastscan::kBatchSize> ip_x0_qr;      // inner product of the 1st bit

    split_batch_estdist(
        batch_data,
        q_obj,
        padded_dim_,
        est_distance.data(),
        low_distance.data(),
        ip_x0_qr.data(),
        use_hacc
    );

    float distk = knns.top_dist();

    // Without reranking data, return the one-bit estimates directly.
    if (ex_bits_ == 0 && !raw_reranking_) {
        detail::insert_candidates(knns, ids, est_distance.data(), num_points);
        return;
    }

    // incremental distance computation - V2
    for (size_t i = 0; i < num_points; ++i) {
        float lower_dist = low_distance[i];
        if (lower_dist < distk) {
            PID id = ids[i];
            float ex_dist;
            if (raw_reranking_) {
                const auto* vector = reinterpret_cast<const float*>(ex_data);
                ex_dist = metric_type_ == METRIC_L2 ? euclidean_sqr(query, vector, dim_)
                                                    : dot_product_dis(query, vector, dim_);
            } else {
                ex_dist = split_distance_boosting(
                    ex_data, ip_func_, q_obj, padded_dim_, ex_bits_, ip_x0_qr[i]
                );
            }
            knns.insert(id, ex_dist);
            distk = knns.top_dist();
        }
        ex_data += rerank_vector_bytes();
    }
}
}  // namespace rabitqlib::ivf
