# The RaBitQ Library

> **New:** We have recently open-sourced the GPU implementation of RaBitQ for high-dimensional vector search. See [cuvs_rabitq](https://github.com/Stardust-SJF/cuvs_rabitq/tree/cuvs_ivf_rabitq).

The RaBitQ Library provides efficient and lightweight implementations of the RaBitQ quantization algorithm ([1-bit version](https://arxiv.org/abs/2405.12497) and [multi-bit version](https://arxiv.org/abs/2409.09913)) and its applications in high-dimensional vector search. It also provides a [GPU implementation](https://github.com/Stardust-SJF/cuvs_rabitq/tree/cuvs_ivf_rabitq). The core algorithm RaBitQ is based on the research from [VectorDB group](https://vectordb-ntu.github.io/) at Nanyang Technological University, Singapore. 

The library is developped by Yutong Gou, Jianyang Gao, Yuexuan Xu, Jifan Shi and Zhonghao Yang. 

The library provides the following key features:

* **RaBitQ** - a vector quantization algorithm as a drop-in replacement of binary and scalar quantization, offering an optimal theoretical error bound
* **RaBitQ for Vector Search** - a reference implementation of RaBitQ's combination with popular vector search indexes

RaBitQLib supports estimating similarity metrics including Euclidean distance, inner product and cosine similarity.

## RaBitQ 

RaBitQ is a vector quantization algorithm as a drop-in replacement of binary and scalar quantization. The key advantages of RaBitQ include

- **High Accuracy with Tiny Space** - RaBitQ achieves the state-of-the-art accuracy under diverse bit-width for the estimation of similarity metrics. It produces promising accuracy with even **1-bit per dimension**.
- **Fast Distance Estimation** - RaBitQ supports to estimate the similarity metrics with high efficiency based on bitwise operations or [FastScan](https://arxiv.org/abs/1704.07355).
- **Theoretical Error Bound** - RaBitQ provides an asymptotically optimal error bound for the estimation of distances and inner product. The error bound can be used for reliable ordering and reranking.

In this library, we provide simple interfaces to support advanced features of RaBitQ. The details are presented in the documentation.

## RaBitQ for Vector Search
 In the library, RaBitQ is combined with IVF, HNSW and QG to deliever different trade-offs among time, space and accuracy. 

Using RaBitQ with IVF and HNSW targets a balance between memory consumption and query performance. Only the quantization codes produced by RaBitQ are stored and the raw data vectors are not accessed during querying. Thus, these methods consume less memory than the raw dataset. 
Using **4-bit, 5-bit and 7-bit** quantization usually suffices to produce **90%, 95% and 99% recall** respectively without reranking. 

Using RaBitQ with QG targets the best query performance by using more memory. It creates multiple quantization codes for every vector to optimize the data access pattern. Thus, QG usually consumes 2x memory of the raw dataset. 

## RaBitQ in Industry

The RaBitQ algorithm has been implemented in many real-world systems in industry including 

- [Milvus](https://github.com/milvus-io/milvus) - IVF + RaBitQ (C++)
- [Faiss](https://github.com/facebookresearch/faiss) - IVF + RaBitQ (C++)
- [VSAG](https://github.com/antgroup/vsag) - HGraph + RaBitQ (C++)
- [VectorChord](https://github.com/tensorchord/VectorChord) - IVF + RaBitQ (Rust)
- [Volcengine OpenSearch](https://www.volcengine.com/docs/6465/1553583) - DiskANN + RaBitQ
- [CockroachDB](https://github.com/cockroachdb/cockroach) - CSPANN + RaBitQ (Golang)
- [ElasticSearch](https://github.com/elastic/elasticsearch) - HNSW + RaBitQ (Java - the algorithm is adopted with some minor modifications and renamed as "BBQ")
- [Lucene](https://github.com/apache/lucene) - HNSW + RaBitQ (Java - the algorithm is adopted with some minor modifications and renamed as "BBQ")
- [turbopuffer](https://turbopuffer.com/blog/ann-v3#:~:text=ANN%20v3%20employs%20the%20RaBitQ) -  SPFresh + RaBitQ

## Acknowledgement

We acknowledge Alexandr Guzhva, Li Liu, Chao Gao, Silu Huang, Jiabao Jin, Xiaoyao Zhong and Jinjing Zhou for valuable feedbacks. 

## Reference 
Please provide a reference of our paper if it helps in your systems or research projects.

<pre style="white-space: pre-wrap; word-break: break-word; font-family: monospace; background: #f5f5f5; padding: 1em; border-radius: 5px; font-size: 0.85em;">
Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, Raymond Chi-Wing Wong, 
"Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search", 
SIGMOD 2025, available at https://arxiv.org/abs/2409.09913
</pre>
