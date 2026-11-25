"""Check the search benchmark report against small persisted IVF fixtures."""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def check(executable, fixtures, directory):
    query_file = directory / "queries.fvecs"
    truth_file = directory / "truth.ivecs"

    def run(fixture, nprobe, truth, *, success=True):
        reference = (
            (fixtures / fixture).with_suffix(".expected").read_text().splitlines()
        )
        dim = int(reference[0].split()[1])
        query = list(map(float, reference[1].split()[:dim]))
        query_file.write_bytes(struct.pack(f"<i{dim}f", dim, *query))
        truth_file.write_bytes(struct.pack("<11i", 10, *truth))
        return subprocess.run(
            [
                executable,
                (fixtures / fixture).with_suffix(".index"),
                query_file,
                truth_file,
                "1",
                str(nprobe),
                "1",
                "1",
                "1",
            ],
            capture_output=True,
            text=True,
            check=success,
        )

    # The first reference query probes an empty/removed cluster with nprobe=1.
    fixture = "v0.9.0/ivf4_flat_l2"
    truth = [92] + [-1] * 9
    assert "# recall_at_10=0" in run(fixture, 1, truth).stdout.splitlines()
    # With all clusters probed it finds the sole valid reference ID.
    assert "# recall_at_10=1" in run(fixture, 4, truth).stdout.splitlines()
    empty = run(fixture, 1, [-1] * 10, success=False)
    assert empty.returncode and "valid ground-truth ID" in empty.stderr
    for fixture, metadata in [
        ("v0.9.0/ivf7_flatrabitq_cosine", "seed=42 metric=cosine"),
        ("v0.5.2/ivf32_ip", "seed=unknown metric=IP"),
    ]:
        assert metadata in run(fixture, 1, truth).stdout


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check(sys.argv[1], Path(sys.argv[2]), Path(tmp))
