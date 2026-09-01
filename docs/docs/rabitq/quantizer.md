# Quantizer

RaBitQLib includes two versions of implementations for the RaBitQ algorithm and designs various data formats to ease practical deployment. Specifically, the two implementaions offer different trade-offs as follows. 

1. Optimal accuracy with longer quantization time.
2. Nearly-optimal accuracy with significantly fast quantization.

Various advanced data formats are provided to support the following needs. 

1. Drop-in replacement for uniform scalar quantization.
2. Efficient distance estimation for single vectors.
3. Efficient distance estimation for batched vectors.
4. Incremental distance estimation for splitted single vectors.
5. Incremental distance estimation for splitted batched vectors.

These data formats map raw floating-point vectors into `uint8` or `uint32`
codes. For the packed extended-code layouts, see
`rabitqlib/quantization/pack_excode.hpp`.

RaBitQ quantizer is included in `rabitq_impl.hpp` and `rabitq.hpp`.
```css
.
├── rabitqlib
│   ├── ...
│   └── quantization
│       ├── ...
│       ├── rabitq_impl.hpp
│       └── rabitq.hpp
└── ...
```

## Implementations
Let $B$ be the bit-width for each dimension. Both implementations of RaBitQ include two steps. 

1. Compute a **binary code** by recording the sign of every coordinate. 
2. Compute an **ex-code** of $B-1$ bits (when $B>1$).

The binary code is easily computed by the function `one_bit_code` in `rabitq_impl.hpp`.

The computation of ex-codes includes two versions of implementation. 

In the first implementation, we compute the ex-codes of RaBitQ based on the algorithm described in the RaBitQ [paper](https://arxiv.org/abs/2409.09913) (Section 3.2.2). For a vector, to minimize the quantization error, the algorithm tries many different rescaling factors. For each rescaling factor, it rescales the vector and performs rounding (i.e., scalar quantization) to generate a quantization code. Then it finds out the factor and codes which minimizes the quantization error. Note that in the library, the range of enumeration is approriately shrinked, which brings better efficiency without affecting the accuracy. 

In the second implementation, instead of enumerating different rescaling factors, it directly rounds every vector based on the **expected optimal factor**. Specifically, recall that all data vectors are randomly rotated before quantization. The expected optimal factor is computed as follows. We sample several random vectors which follow uniform distribution on the unit sphere and use the first implementation to quantize them. We record the optimal factor for each and take the average of the optimal factors as the expected optimal factor. This implementation introduces some accuracy decrease while significantly speeds up the quantization.



## Data Format

### Format 1 - Drop-in Replacement of Uniform Scalar Quantization

This format allows RaBitQ to be used as a direct replacement for uniform scalar quantization, which offers higher accuracy under the same bit-width. The improvement of accuracy is significant when the bit-width is small ($B < 6$).
```cpp
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/quantization/rabitq.hpp"

int main() {
    size_t dim = 768;

    std::vector<float> vector(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dist(gen);
    }

    size_t bits = 8;                  // num of bits for total code
    std::vector<uint32_t> code(dim);  // code
    float delta;                      // delta for scalar quantization
    float vl;                         // lower value for scalar quantization

    // scalar quantization
    rabitqlib::quant::quantize_scalar(vector.data(), dim, bits, code.data(), delta, vl);

    // faster version, must init a config struct first
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);
    rabitqlib::quant::quantize_scalar(
        vector.data(), dim, bits, code.data(), delta, vl, config
    );

    // Note that with this interface, the codes are not compactly stored.
    // To compactly store the codes, please refer to
    // `rabitqlib/quantization/pack_excode.hpp`

    return 0;
}
```



### Format 2 - Distance Estimation for a Single Vector


#### Indexing

This format is designed for computing distance metrics between data vectors and query vectors efficiently. It includes precomputed factors to ease similarity calculations.
```cpp
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/quantization/rabitq.hpp"

int main() {
    size_t dim = 768;

    std::vector<float> vector(dim);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dist(gen);
        centroid[i] = dist(gen);
    }

    size_t bits = 8;                  // num of bits for total code
    std::vector<uint32_t> code(dim);  // code
    float f_add;                      // factors for estimating similarity
    float f_rescale;                  // factors for estimating similarity
    float f_error;                    // factors for computing error bounds

    rabitqlib::quant::quantize_full_single(
        vector.data(),
        centroid.data(),
        dim,
        bits,
        code.data(),
        f_add,
        f_rescale,
        f_error,
        rabitqlib::METRIC_L2
    );

    // faster version, must init a config struct first
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);
    rabitqlib::quant::quantize_full_single(
        vector.data(),
        centroid.data(),
        dim,
        bits,
        code.data(),
        f_add,
        f_rescale,
        f_error,
        rabitqlib::METRIC_L2,
        config
    );

    // Note that with this interface, the codes are not compactly stored.
    // To compactly store the codes, please refer to
    // `rabitqlib/quantization/pack_excode.hpp`

    return 0;
}
```

#### Querying
After quantization, we can use the pre-computed quantization codes and factors to get estimated distance for a given query.Here, we separately stored factors and quantization codes, and user may choose to compact them all together to get a improve space locality (e.g., implementation in our index). Also, we did not rotate the vectors in this example for simplicity of illustration.
```cpp
...
    ...... // we omit the quantization code here
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
        query[i] = dist(gen);
    }

    // please refer to estimator.md for defination of these factors
    float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
    float k1xsumq = c_1 * std::accumulate(query.begin(), query.end(), 0.F);
    float g_add = rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim);

    float est_dist = rabitqlib::quant::full_est_dist(
        code.data(),
        query.data(),
        rabitqlib::excode_ipimpl::ip_fxi,
        dim,
        bits,
        f_add,
        f_rescale,
        g_add,
        k1xsumq
    );
    float gt_dist = rabitqlib::euclidean_sqr(query.data(), vector.data(), dim);

    std::cout << "Estimated distance: " << est_dist << '\n';
    std::cout << "GT distance: " << gt_dist << '\n';

```


### Format 3 - Distance Estimation for a Batch of Vectors (for QG)
A variant of this data format is used in QG. 

#### Indexing
This format is designed for scenarios where distances need to be computed between a query vector and multiple data vectors (quantized into 1-bit per dimension) simultaneously, providing significant performance improvements with [FastScan](https://arxiv.org/abs/1704.07355) for batch processing. 
In practical implementation of our SymphonyQG index, the data is compactly stored. Please refer to `QGBatchDataMap` in `rabitqlib/quantization/data_layout.hpp` for detailed information.



```cpp
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/quantization/rabitq_impl.hpp"

int main() {
    size_t dim = 768;
    size_t batch_size = 32;  // a batch for FastScan contains 32 vectors

    std::vector<float> vector(dim * batch_size);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < batch_size * dim; ++i) {
        vector[i] = dist(gen);
    }
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = dist(gen);
    }

    // std::vector<uint32_t> code(dim);  // code
    std::vector<uint8_t> packed_code(batch_size * dim / 8);
    std::vector<float> f_add(batch_size);      // factors for estimating similarity
    std::vector<float> f_rescale(batch_size);  // factors for estimating similarity
    std::vector<float> f_error(batch_size);    // factors for computing error bounds

    rabitqlib::quant::rabitq_impl::one_bit::one_bit_batch_code(
        vector.data(),
        centroid.data(),
        batch_size,
        dim,
        packed_code.data(),
        f_add.data(),
        f_rescale.data(),
        f_error.data(),
        rabitqlib::METRIC_L2
    );

    return 0;
}
```

#### Querying
During querying, a query is pre-processed as follows. Then FastScan can be called to estimate distance batch by batch. The detailed implementation is in `rabitqlib/index/estimator.hpp`. 
Here, we assume the data are compactly stored in the layout of `QGBatchDataMap` in `rabitqlib/quantization/data_layout.hpp`.

```cpp

    size_t dim = 768;  // the dimensionality
    std::vector<float> query(dim);
    std::vector<float> centroid(dim);
    std::vector<char> batch_data(rabitqlib::QGBatchDataMap<float>::data_bytes(dim));

    rabitqlib::BatchQuery<float> processed_query(query.data(), dim);

    // G_add is the squared L2 distance between the query and the centroid.
    // This standalone example omits rotation, so both operands are raw vectors.
    // In an index that applies rotation, rotate both with the same transformation.
    processed_query.set_g_add(
        rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim)
    );

    size_t batch_size = 32;

    std::vector<float> est_distance(batch_size);  // store the estimated distances

    // We suggest users to customize the kernel if some outputs are not needed.
    // Eg., QG does not need error bounds, thus we only get the estimated distance here.
    // If users want to maintain the information of error bound, please refer to our
    // implementation of IVF index.

    rabitqlib::qg_batch_estdist(
        batch_data.data(), processed_query, dim, est_distance.data()
    );

```

### Format 4 - Incremental Distance Estimation for Split Single Vectors (for HNSW)
This format can be used in HNSW. 


#### Indexing
This format supports computing distances incrementally when vectors are split across multiple memory locations or when only partial vector information is available at a time.


```cpp
#include <random>
#include <vector>

#include "rabitqlib/quantization/rabitq.hpp"

int main() {
    size_t dim = 768;

    std::vector<float> vector(dim);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dist(gen);
        centroid[i] = dist(gen);
    }

    size_t bits = 5;  // num of bits for total code
    // `bin_data` includes compact binary codes (dim / 8 bytes) and three factors - f_add,
    // f_rescale and f_error (12 bytes) f_error can be dropped if the error bound is not
    // used in your index (e.g., QG).
    // You can also use rabitqlib::BinDataMap<float>::data_bytes(dim) to get the bin data
    // bytes
    std::vector<char> bin_data((dim / 8) + 12);

    // `ex_data` includes compact binary codes (dim / 8 bytes) and two factors - f_add_ex,
    // f_rescale_ex (8 bytes). Here, we drop f_error_ex since it is not used in this index.
    // You can also use rabitqlib::ExDataMap<float>::data_bytes(dim, bits-1) to get the ex
    // data bytes
    std::vector<char> ex_data((dim * (bits - 1) / 8) + 8);

    rabitqlib::quant::quantize_split_single(
        vector.data(),
        centroid.data(),
        dim,
        bits - 1,
        bin_data.data(),
        ex_data.data(),
        rabitqlib::METRIC_L2
    );

    // use fast implementation for the data format
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);
    rabitqlib::quant::quantize_split_single(
        vector.data(),
        centroid.data(),
        dim,
        bits - 1,
        bin_data.data(),
        ex_data.data(),
        rabitqlib::METRIC_L2,
        config
    );

    return 0;
}
```

#### Querying

```cpp
...
    size_t dim = 768;  // the dimensionality
    size_t bits = 5;   // the bit-width of DATA vectors
    std::vector<float> query(dim);

    // the config of fast quantizer is necessary for preprocessing queries
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);

    rabitqlib::SplitSingleQuery<float> processed_query(
        query.data(), dim, bits - 1, config, rabitqlib::METRIC_L2
    );

    // This standalone example omits rotation, so query and centroid are both raw.
    // If rotation is used, apply the same transformation to both operands.
    float distance_sqr = rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim);
    float distance_norm = std::sqrt(distance_sqr);

    // Compute estimated distances based on binary codes
    float ip_x0_qr;
    float est_dist;
    float low_dist;

    rabitqlib::split_single_estdist(
        bin_data.data(),
        processed_query,
        dim,
        ip_x0_qr,
        est_dist,
        low_dist,
        distance_sqr,
        distance_norm
    );

    // the kernel of computing inner product between compact codes and query vectors
    auto ip_func = rabitqlib::select_excode_ipfunc(bits - 1);

    // Compute more accurate distance based on full codes.
    float est_dist_ex;
    float low_dist_ex;
    float ip_x0_qr_ex;

    rabitqlib::split_single_fulldist(
        bin_data.data(),
        ex_data.data(),
        ip_func,
        processed_query,
        dim,
        bits - 1,
        est_dist_ex,
        low_dist_ex,
        ip_x0_qr_ex,
        distance_sqr,
        distance_norm
    );
```

### Format 5 - Incremental Distance Estimation for Split Batched Vectors (for IVF)
This data format is used in IVF.

#### Indexing
This format combines the benefits of batched processing with incremental computation, allowing efficient distance estimation when dealing with large collections of split vectors.

```cpp
#include <random>
#include <vector>

#include "rabitqlib/quantization/rabitq.hpp"

int main() {
    size_t dim = 768;
    size_t batch_size = 32;

    std::vector<float> vector(dim * batch_size);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = dist(gen);
    }
    for (size_t i = 0; i < dim * batch_size; ++i) {
        vector[i] = dist(gen);
    }

    size_t bits = 5;  // num of bits for full codes
    // `batch_data` includes packed binary codes (dim / 8 bytes) and three factors - f_add,
    // f_rescale and f_error (12 bytes)
    // f_error can be dropped if the error bound is not used in your index (e.g., QG)
    std::vector<char> batch_data((dim / 8 + 12) * batch_size);

    // `ex_data` includes compact binary codes (dim / 8 bytes) and two factors - f_add_ex,
    // f_rescale_ex (8 bytes). Here, we drop f_error_ex since it is not used in this index.
    // You can also use rabitqlib::ExDataMap<float>::data_bytes(dim, bits-1) to get the ex
    // data bytes
    std::vector<char> ex_data((dim * (bits - 1) / 8 + 8) * batch_size);

    rabitqlib::quant::quantize_split_batch(
        vector.data(),
        centroid.data(),
        batch_size,
        dim,
        bits - 1,
        batch_data.data(),
        ex_data.data(),
        rabitqlib::METRIC_L2
    );

    // use fast implementation for the data format
    rabitqlib::quant::RabitqConfig config = rabitqlib::quant::faster_config(dim, bits);
    rabitqlib::quant::quantize_split_batch(
        vector.data(),
        centroid.data(),
        batch_size,
        dim,
        bits - 1,
        batch_data.data(),
        ex_data.data(),
        rabitqlib::METRIC_L2,
        config
    );

    return 0;
}
```

#### Querying

```cpp
    size_t dim = 768;  // the dimensionality
    size_t bits = 5;   // the bit-width of DATA vectors
    std::vector<float> query(dim);

    // The flag use_hacc controls the precision of FastScan.
    // `use_hacc = false` - each number in LUTs is quantized into 8 bits.
    // `use_hacc = true` - each number in LUTs is quantized into 16 bits.
    // By default, `use_hacc = true` as it works for all settings of `bits`,
    // i.e., the bit-width of queries is significantly larger than that for data.
    // When the bit-width of data <= 2, `use_hacc = false` does not harm accuracy.

    rabitqlib::SplitBatchQuery<float> processed_query(
        query.data(), dim, bits - 1, rabitqlib::METRIC_L2, true
    );

    // SplitBatchQuery expects the L2 norm and squares it internally for G_add.
    // This example omits rotation, so query and centroid are both raw vectors.
    float distance_norm =
        std::sqrt(rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim));
    processed_query.set_g_add(distance_norm);

    size_t batch_size = 32;

    std::vector<float> est_distance(batch_size);  // store the estimated distances
    std::vector<float> low_distance(batch_size);  // store the lower bound
    std::vector<float> ip_x0_qr(batch_size
    );  // store the intermediate result of inner product

    // We suggest users to customize the kernel if some outputs are not needed.
    // For example, QG does not need error bounds, see `qg_batch_estdist` in
    // `rabitqlib/index/estimator.hpp` for details.

    rabitqlib::split_batch_estdist(
        batch_data.data(),
        processed_query,
        dim,
        est_distance.data(),
        low_distance.data(),
        ip_x0_qr.data(),
        true
    );

    // the kernel of computing inner product between compact codes and query vectors
    auto ip_func = rabitqlib::select_excode_ipfunc(bits - 1);

    size_t i = 15;
    rabitqlib::split_distance_boosting(
        ex_data.data() + (i * rabitqlib::ExDataMap<float>::data_bytes(dim, bits - 1)),
        ip_func,
        processed_query,
        dim,
        bits - 1,
        ip_x0_qr[i]
    );

```
