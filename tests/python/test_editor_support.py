"""Installed-package contracts for editor declarations and runtime help."""

import ast
import doctest
import re
from pathlib import Path

import pytest
import rabitqlib


def installed_stub():
    package = Path(rabitqlib.__file__).parent
    assert (package / "py.typed").is_file()
    stub = package / "__init__.pyi"
    assert stub.is_file()
    return ast.parse(stub.read_text())


def declarations():
    return {
        node.name: node
        for node in installed_stub().body
        if isinstance(node, ast.ClassDef)
    }


def test_installed_public_declarations():
    tree = installed_stub()
    exports = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert ast.literal_eval(exports.value) == rabitqlib.__all__
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    names.update(
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    )
    assert set(rabitqlib.__all__) <= names
    for name, cls in declarations().items():
        runtime = getattr(rabitqlib, name)
        members = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
        for node in cls.body:
            if isinstance(node, ast.AnnAssign):
                assert isinstance(node.target, ast.Name)
                members.add(node.target.id)
        if name == "FinalAssignmentMode":
            assert {"Approximate", "SymphonyQG", "Exact", "name", "value"} <= members
            continue
        public = {key for key in vars(runtime) if not key.startswith("_")}
        assert public <= members, (name, public - members)
        for member in members - {"d", "k", "cp"}:
            assert hasattr(runtime, member), (name, member)


def normalized(text):
    return " ".join(text.split())


def test_editor_docs_match_runtime_help():
    for name, cls in declarations().items():
        runtime = getattr(rabitqlib, name)
        assert normalized(ast.get_docstring(cls)) in normalized(runtime.__doc__)
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            documentation = ast.get_docstring(node)
            if documentation is None:
                continue
            # Python wrappers inherit __init__(d, k, **kwargs); class help documents
            # the supported keyword parameters and the stub expands them for IDEs.
            if name in ("QGKMeans", "RaBitQKMeans") and node.name == "__init__":
                continue
            actual = getattr(runtime, node.name).__doc__
            assert normalized(documentation) in normalized(actual), (name, node.name)


@pytest.mark.parametrize("name", ["IvfIndex", "HnswIndex", "SymqgIndex"])
def test_native_signature_names_and_defaults(name):
    cls = declarations()[name]
    runtime = getattr(rabitqlib, name)
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(
            isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
        ):
            continue
        signature = getattr(runtime, node.name).__doc__.splitlines()[0]
        actual_names = re.findall(r"(?:\(|, )([A-Za-z_]\w*):", signature)
        expected_names = [a.arg for a in node.args.args]
        assert actual_names == expected_names, (name, node.name, signature)
        default_args = node.args.args[len(node.args.args) - len(node.args.defaults) :]
        for arg, value in zip(default_args, node.args.defaults):
            actual = re.search(rf"\b{arg.arg}: [^=]+ = ([^,)]+)", signature)
            assert actual is not None, (name, node.name, arg.arg)
            assert ast.literal_eval(actual[1]) == ast.literal_eval(value)


@pytest.mark.parametrize("name", ["QGKMeans", "RaBitQKMeans"])
def test_clustering_keyword_defaults(name):
    cls = declarations()[name]
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    instance = getattr(rabitqlib, name)(64, 40)
    for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults):
        assert default is not None, (name, arg.arg)
        if isinstance(default, ast.Attribute):
            expected = getattr(rabitqlib.FinalAssignmentMode, default.attr)
        else:
            expected = ast.literal_eval(default)
        assert getattr(instance.cp, arg.arg) == expected
    for name in (
        "centroids",
        "assignments",
        "distances",
        "obj",
        "final_obj",
        "iteration_stats",
    ):
        assert getattr(instance, name) is None


def test_documentation_examples():
    runner = doctest.DocTestRunner()
    parser = doctest.DocTestParser()
    for name in rabitqlib.__all__:
        # Explicitly include native classes, which testmod otherwise skips
        # because they belong to the extension module.
        documentation = getattr(rabitqlib, name).__doc__
        runner.run(parser.get_doctest(documentation, {}, name, None, 0))
    assert runner.failures == 0
    assert runner.tries > 0


@pytest.mark.parametrize("kind", ["ivf", "hnsw", "symqg"])
def test_documented_search_result_types(kind):
    import numpy as np

    data = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    if kind == "symqg":
        index = rabitqlib.SymqgIndex(64, 32)
        index.build(data, 40)
        result = index.search(data[:2], 3, ef=20)
    else:
        centroids = data[:2].copy()
        labels = np.arange(80, dtype=np.uint32) % 2
        if kind == "ivf":
            index = rabitqlib.IvfIndex(64, 80, 2, 4)
        else:
            index = rabitqlib.HnswIndex(64, 80)
        index.build(data, centroids, labels)
        if isinstance(index, rabitqlib.IvfIndex):
            result = index.search(data[:2], 3, nprobe=2)
        else:
            result = index.search(data[:2], 3, ef=20)
    ids, distances = result
    assert ids.shape == distances.shape == (2, 3)
    assert ids.dtype == np.uint32
    assert distances.dtype == np.float32


@pytest.mark.parametrize("name", ["QGKMeans", "RaBitQKMeans"])
def test_documented_training_result_types(name):
    import numpy as np

    data = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    model = getattr(rabitqlib, name)(64, 40, niter=2, num_threads=1)
    assert isinstance(model.train(data), float)
    assert model.centroids.shape == (40, 64)
    assert model.centroids.dtype == np.float32
    assert model.assignments.shape == model.distances.shape == (80,)
    assert model.assignments.dtype == np.uint32
    assert model.distances.dtype == np.float32
    assert model.obj.dtype == np.float64
    assert len(model.obj) == len(model.iteration_stats)
    assert isinstance(model.final_obj, float)


def test_numpy_scalar_inputs_and_enum_members():
    import runpy

    fixture = Path(__file__).parent / "typing" / "public_api.py"
    runpy.run_path(str(fixture))["check_numpy_scalars"]()


def test_valid_string_options_and_integer_ids():
    import runpy

    fixture = Path(__file__).parent / "typing" / "public_api.py"
    runpy.run_path(str(fixture))["check_input_contracts"]()
