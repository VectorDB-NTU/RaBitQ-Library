# Reranking

Reranking refines shortlisted candidates by computing distances from their
original vectors. Those vectors may be kept in memory or retrieved from disk.

RaBitQ error intervals can guide approximate reranking: a candidate can be
skipped when its estimated lower bound exceeds the relevant upper bound among
the current nearest-neighbor candidates. These are probabilistic intervals, not
deterministic bounds for every query.

The strategy may rerank fewer than $K$ vectors when only boundary candidates
need refinement. It does not guarantee exact top-k membership or exact distances
for results that were never reranked; measure the recall/cost trade-off for the workload.

This is an application-level strategy, not a built-in disk-retrieval API.
HNSW-RaBitQ stores quantized vectors, so applications must retain raw vectors separately
for exact reranking. [IVF-RaBitQ raw mode](../index/ivf.md#index-construction)
(`nbits=32`) instead owns the original vectors and reranks them in memory.

## Algorithm description

Given at least `K` candidates with distance estimates and error intervals, keep
`K` candidates in a max-heap ordered by their upper bounds. The heap root is the
candidate with the **largest upper bound**, not the nearest candidate.
Refining a candidate computes its exact distance and sets both bounds to that value.

```text
Sort candidates by estimated distance.
Initialize heap H with the first K candidates, keyed by upper bound.
For each remaining candidate A:
    Repeat:
        W = candidate with the largest upper bound in H
        If A.lower_bound > W.upper_bound:
            Skip A and continue with the next candidate.
        If W has already been refined:
            Refine A.
            Insert A into H, retaining the K smallest upper bounds.
            Continue with the next candidate.
        Refine W and restore heap order.
```

Candidates remaining in the heap may still have estimated distances. If the
application needs exact distances for every returned candidate, refine those
candidates too. This does not recover candidates excluded earlier by approximate
search or probabilistic pruning.
