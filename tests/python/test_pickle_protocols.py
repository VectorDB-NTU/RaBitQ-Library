"""Pickle protocols 0 and 1 raise or round-trip instead of aborting the interpreter.

Those protocols reduce through copyreg, which calls the constructor of pybind11's base
class; that throws through the C API and aborts the process (pybind11 issue 3745)
unless the class defines __reduce_ex__. The calls therefore run in a child process.
"""

import subprocess
import sys
import textwrap

import rabitqlib
from rabitqlib import _rabitqlib

CHILD = textwrap.dedent(
    """
    import pickle
    import sys

    import numpy as np
    import rabitqlib
    from rabitqlib import _rabitqlib, quant

    if rabitqlib.__file__ != sys.argv[1]:
        raise SystemExit(f"child imported {rabitqlib.__file__}, not {sys.argv[1]}")
    x = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)
    flat = rabitqlib.RaBitQKMeans(64, 2, niter=1, num_threads=1, seed=1)
    flat.train(x)
    graph = rabitqlib.QGKMeans(64, 33, niter=1, num_threads=1, seed=1)
    graph.train(x)

    def state(value):
        return pickle.dumps(value, protocol=2)

    picklable = {
        "Rotator": quant.Rotator(64, seed=1),
        "Query": quant.Query(x[0], metric="ip"),
        "Query with centroid": quant.Query(x[0], metric="l2", centroid=x[1]),
        "MetricType": _rabitqlib.MetricType.IP,
        "RotatorType": _rabitqlib.RotatorType.FhtKacRotator,
    }
    unpicklable = {
        "IvfIndex": rabitqlib.IvfIndex(64, 2, seed=1),
        "HnswIndex": rabitqlib.HnswIndex(64, seed=1),
        "SymqgIndex": rabitqlib.SymqgIndex(64, seed=1),
        "QGKMeansParameters": rabitqlib.QGKMeansParameters(),
        "RaBitQKMeansParameters": rabitqlib.RaBitQKMeansParameters(),
        "KMeansIterationStats": flat.iteration_stats[0],
        "_RaBitQKMeans": flat._impl,
        "_QGKMeans": graph._impl,
        "RaBitQKMeans": flat,
    }
    for protocol in (0, 1):
        for name, value in picklable.items():
            copy = pickle.loads(pickle.dumps(value, protocol=protocol))
            if state(copy) != state(value):
                raise SystemExit(f"{name} changed in a protocol {protocol} round trip")
            print(name, protocol, "round-trip", flush=True)
        for name, value in unpicklable.items():
            try:
                pickle.dumps(value, protocol=protocol)
            except TypeError as error:
                if "cannot pickle" not in str(error):
                    raise SystemExit(f"{name}: {error}")
                print(name, protocol, "TypeError", flush=True)
            else:
                raise SystemExit(f"{name} pickled with protocol {protocol}")
    print(len(picklable) + len(unpicklable), "cases", flush=True)
    """
)


def test_old_protocols_round_trip_or_raise_in_a_child_process():
    result = subprocess.run(
        [sys.executable, "-c", CHILD, rabitqlib.__file__],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    cases = int(lines[-1].split()[0])
    assert len(lines) == 1 + 2 * cases, lines
    assert "Query with centroid 0 round-trip" in lines
    assert "_QGKMeans 1 TypeError" in lines


def extension_classes():
    """Every pybind11 class that rabitqlib registered, found through the subclass tree."""
    pending = [
        base
        for base in type(_rabitqlib.MetricType.IP).__mro__
        if base.__name__ == "pybind11_object"
    ]
    found = []
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        # quant.py renames its classes' module to rabitqlib.quant when imported.
        if cls.__module__.split(".")[0] == "rabitqlib":
            found.append(cls)
    return found


def test_every_extension_class_defines_reduce_ex():
    classes = extension_classes()
    names = {cls.__name__ for cls in classes}
    assert {"IvfIndex", "Query", "_QGKMeans", "MetricType"} <= names, names
    assert (
        sorted(cls.__name__ for cls in classes if "__reduce_ex__" not in vars(cls))
        == []
    )
