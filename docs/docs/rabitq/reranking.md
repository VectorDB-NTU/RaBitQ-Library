# Reranking
Reranking is a technique widely adopted for improving recall of vector search. During searching, ANN algorithms usually 

1. shortlist a set of candidates based on an index and their quantization codes (e.g., **in memory**);
2. retrieve the raw vectors (e.g., **from disks**) for those candidates which have the smallest estimated distances;
3. computes the exact distances for the candidates to find the nearest neighbors.

RaBitQ has a unique advantage in reranking due to its theoretical error bound. It can skip reranking a candidate if the lower bound of its estimated distance is larger than the upper bound of the distance of the nearest neighbors. 

Based on error bounds, it is possible to rerank **fewer than $K$ vectors** to achieve nearly perfect recall - it only reranks the vectors on the boundaries of KNNs.

This is an application-level strategy, not a built-in disk-retrieval API.
HNSW stores quantized vectors, so applications must retain raw vectors separately
for exact reranking. [IVF raw mode](../index/ivf.md#index-construction)
(`nbits=32`) instead owns the original vectors and reranks them in memory.

## Algorithm Description
Let $K$ be the number of nearest neighbors we target. After receiving the candidates and their estimated distances from an index, e.g., HNSW RaBitQ, we perform the following strategy of reranking to minimize the number of retrieved raw vectors from disks.

1. Sort the candidates with respect to their estimated distances.
2. Initialize a max-heap (sorted by **upper bounds of distances**) `KNNs` with the top-$K$ candidates which have the smallest estimated distances.
3. Enumerate the remaining candidates. For a new candidate A, 
    1. [Condition 1 - A's lower bound $>$ the maximum upper bound in `KNNs`.] 
        1. Drop the new candidate and move on. 
    2. [Condition 2 -  A's lower bound $\le$ the maximum upper bound in `KNNs`, and the top candidate in `KNNs` has been reranked.]
        1. Rerank the new candidate, update `KNNs` and move on.
    3. [Condition 3 - A's lower bound $\le$ the maximum upper bound in `KNNs`, and the top candidate in `KNNs` has NOT been reranked.]
        1. Rerank the top candidate in `KNNs`, update `KNNs` and repeat the procedure for candidate A. 
