"""Python boundary coverage for finite quantization factors and pruning."""

import numpy as np
import pytest
from rabitqlib import IvfIndex


@pytest.mark.parametrize("nbits", [4, 9, 32])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_collinear_ivf_query_keeps_finite_distances(nbits, metric):
    data = np.full((1, 512), 0.1, dtype=np.float32)
    centroid = np.zeros_like(data)
    index = IvfIndex(data.shape[1], len(data), 1, nbits=nbits, metric=metric)
    index.build(data, centroid, np.zeros(len(data), dtype=np.uint32), num_threads=1)

    # The C++ regression fixes the rotated residual to trigger negative roundoff.
    # This exercises the normal Python boundary with its randomly sampled rotation.
    # A zero query avoids query-LUT approximation in the distance oracle.
    ids, distances = index.search(centroid, k=1, nprobe=1, num_threads=1)
    np.testing.assert_array_equal(ids, [[0]])
    assert np.all(np.isfinite(distances))
    expected = np.sum(data.astype(np.float64) ** 2) if metric == "l2" else 1.0
    np.testing.assert_allclose(distances, expected, rtol=2e-5, atol=2e-5)
