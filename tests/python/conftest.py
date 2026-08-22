"""Shared fixtures and helpers for the rabitqlib Python test suite.

All data fixtures use dim=64 (smallest FhtKacRotator-supported dimension) and a
fixed seed so tests are deterministic and fast.
"""

import numpy as np
import pytest

# ── dataset parameters ───────────────────────────────────────────────────────
DIM = 64
N_VECTORS = 500
N_QUERIES = 20
N_CLUSTERS = 5  # used by IVF and HNSW


# ── data fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def base_data() -> np.ndarray:
    """500 random float32 vectors of dimension 64."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_VECTORS, DIM)).astype(np.float32)


@pytest.fixture(scope="session")
def query_data() -> np.ndarray:
    """20 random float32 query vectors of dimension 64."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)


@pytest.fixture(scope="session")
def clusters(base_data: np.ndarray):
    """Round-robin cluster assignment — guaranteed no empty clusters, no faiss needed."""
    cluster_ids = (np.arange(N_VECTORS) % N_CLUSTERS).astype(np.uint32)
    centroids = np.stack(
        [base_data[cluster_ids == c].mean(axis=0) for c in range(N_CLUSTERS)]
    ).astype(np.float32)
    return centroids, cluster_ids


# ── pure-numpy helpers ────────────────────────────────────────────────────────


def brute_force_knn(data: np.ndarray, queries: np.ndarray, k: int):
    """Exact L2 k-NN. Returns (ids, sq_dists) of shape (nq, k).

    Note: distances are *squared* L2. Ordering is identical to L2, so
    recall comparisons against index results (which return true L2) are valid.
    """
    # (nq, n, dim) — stays in RAM comfortably for small test sizes
    diffs = queries[:, np.newaxis, :] - data[np.newaxis, :, :]
    sq_dists = np.einsum("ijk,ijk->ij", diffs, diffs)
    ids = np.argsort(sq_dists, axis=1)[:, :k]
    dists = np.take_along_axis(sq_dists, ids, axis=1)
    return ids, dists


def recall_at_k(approx_ids: np.ndarray, exact_ids: np.ndarray, k: int) -> float:
    """Fraction of exact top-k neighbors found in the approximate top-k results."""
    nq = approx_ids.shape[0]
    hits = sum(len(set(approx_ids[i, :k]) & set(exact_ids[i, :k])) for i in range(nq))
    return hits / (nq * k)
