"""Checks that repeated comparisons reuse only identical exhaustive evaluations."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "sample/python/compare_with_faiss.py"


def run_comparison(monkeypatch, capsys, *, force_recompute):
    spec = importlib.util.spec_from_file_location("compare_with_faiss", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    calls = 0
    original = module.exact_objective

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    module.exact_objective = counted
    if force_recompute:

        class NoCacheNumpy:
            def __getattr__(self, name):
                return getattr(np, name)

            @staticmethod
            def array_equal(left, right):
                return False

        module.np = NoCacheNumpy()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--n",
            "1200",
            "--d",
            "64",
            "--k",
            "64",
            "--niter",
            "2",
            "--threads",
            "4",
            "--seed",
            "23",
            "--num-seeds",
            "1",
            "--repeats",
            "2",
        ],
    )
    module.main()
    quality = []
    for line in capsys.readouterr().out.splitlines():
        fields = line.split()
        if (
            fields
            and fields[0] in {"QGKMeans", "RaBitQKMeans", "FAISS"}
            and len(fields) == 5
        ):
            quality.append((fields[0], *fields[2:]))
        elif line.startswith(
            (
                "QGKMeans returned-label objective:",
                "RaBitQKMeans returned-label objective:",
            )
        ):
            quality.append(line)
    return calls, quality


def test_repeated_comparison_reuses_identical_exhaustive_results(monkeypatch, capsys):
    cached_calls, cached_quality = run_comparison(
        monkeypatch, capsys, force_recompute=False
    )
    full_calls, full_quality = run_comparison(monkeypatch, capsys, force_recompute=True)
    assert cached_calls == 3
    assert full_calls == 6
    assert cached_quality == full_quality


def test_default_comparison_trains_once_per_method(monkeypatch):
    spec = importlib.util.spec_from_file_location("compare_with_faiss", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    args = module.parse_args()
    assert args.num_seeds == 1
    assert args.repeats == 1


@pytest.mark.parametrize("spherical", [False, True])
def test_native_runtimes_are_isolated(tmp_path, spherical):
    # This guard is inherited by spawned workers. It catches mixed imports even
    # on machines where the two runtimes happen to coexist without a slowdown.
    log = tmp_path / "imports.jsonl"
    (tmp_path / "sitecustomize.py").write_text(
        """
import json
import os
import sys
from pathlib import Path

class NativeImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"faiss", "rabitqlib"}:
            other = "rabitqlib" if fullname == "faiss" else "faiss"
            assert other not in sys.modules, "mixed native runtimes"
            row = {"pid": os.getpid(), "library": fullname}
            with Path(os.environ["NATIVE_IMPORT_LOG"]).open("a") as stream:
                stream.write(json.dumps(row) + "\\n")
sys.meta_path.insert(0, NativeImportGuard())
"""
    )
    original = np.random.default_rng(23).standard_normal((1200, 64), dtype=np.float32)
    input_path = tmp_path / "input.npy"
    np.save(input_path, original)
    env = dict(
        os.environ,
        PYTHONPATH=str(tmp_path),
        NATIVE_IMPORT_LOG=str(log),
        OMP_PROC_BIND="close",
        OMP_PLACES="threads",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--k",
            "64",
            "--niter",
            "2",
            "--threads",
            "4",
            "--repeats",
            "2",
            *(["--spherical"] if spherical else []),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    imports = [json.loads(line) for line in log.read_text().splitlines()]
    assert {row["library"] for row in imports} == {"faiss", "rabitqlib"}
    assert len({row["pid"] for row in imports}) == 2
    for method in ("QGKMeans", "RaBitQKMeans", "FAISS"):
        rows = [
            line.split()
            for line in result.stdout.splitlines()
            if line.split() and line.split()[0] == method and len(line.split()) == 5
        ]
        assert len(rows) == 2
        assert all(int(row[2]) == 2 for row in rows)
        assert all(np.isfinite(float(row[3])) for row in rows)
        if method != "FAISS":
            assert f"{method} returned-label objective:" in result.stdout
            assert f"{method}/FAISS paired training-time ratio:" in result.stdout
            assert (
                f"{method} exhaustive objective difference vs FAISS (%):"
                in result.stdout
            )
    assert "seeds=1, repeats/seed=2" in result.stdout
    np.testing.assert_array_equal(np.load(input_path), original)
