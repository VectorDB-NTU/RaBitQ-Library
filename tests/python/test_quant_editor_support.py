"""Installed-package contracts for the rabitqlib.quant stub and runtime help."""

import ast
import doctest
import re
from pathlib import Path

import pytest
import rabitqlib
from rabitqlib import quant


def installed_stub():
    stub = Path(rabitqlib.__file__).parent / "quant.pyi"
    assert stub.is_file()
    return ast.parse(stub.read_text())


def normalized(text):
    return " ".join(text.split())


def stub_functions(body):
    """Each name's first declaration; overloads share their parameters and docs."""
    functions = {}
    for node in body:
        if isinstance(node, ast.FunctionDef):
            functions.setdefault(node.name, node)
    return functions


def stub_parameters(node):
    args = node.args
    positional = [arg.arg for arg in args.posonlyargs + args.args if arg.arg != "self"]
    defaults = {
        arg.arg: ast.literal_eval(value)
        for arg, value in zip(args.kwonlyargs, args.kw_defaults)
        if value is not None
    }
    return positional, [arg.arg for arg in args.kwonlyargs], defaults


def runtime_parameters(member):
    """The same, from pybind11's first doc line."""
    signature = member.__doc__.splitlines()[0]
    names = [
        n for n in re.findall(r"(?:\(|, )([A-Za-z_]\w*):", signature) if n != "self"
    ]
    _, star, keyword_part = signature.partition(", *, ")
    keyword = re.findall(r"(?:^|, )([A-Za-z_]\w*):", keyword_part) if star else []
    defaults = {
        found[1]: ast.literal_eval(found[2])
        for found in re.finditer(r"\b([A-Za-z_]\w*): [^=,]+ = ([^,)]+)", signature)
        if found[1] in keyword
    }
    return [n for n in names if n not in keyword], keyword, defaults


def test_module_docs_and_exports():
    tree = installed_stub()
    assert "PEP 411" in quant.__doc__
    assert normalized(ast.get_docstring(tree)) in normalized(quant.__doc__)
    exports = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    )
    assert ast.literal_eval(exports.value) == quant.__all__
    declared = {
        n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    }
    declared.update(n.target.id for n in tree.body if isinstance(n, ast.AnnAssign))
    assert set(quant.__all__) == declared
    for name in quant.__all__:
        assert hasattr(quant, name), name


@pytest.mark.parametrize(
    "name",
    [
        "padded_dim",
        "quantize",
        "split",
        "code_factors",
        "pack_codes",
        "unpack_codes",
        "query_terms",
        "estimate_from_ip",
    ],
)
def test_function_signatures_and_docs(name):
    node = stub_functions(installed_stub().body)[name]
    runtime = getattr(quant, name)
    assert runtime_parameters(runtime) == stub_parameters(node), name
    assert normalized(ast.get_docstring(node)) in normalized(runtime.__doc__), name


def test_classes_match_runtime():
    classes = {n.name: n for n in installed_stub().body if isinstance(n, ast.ClassDef)}
    assert set(classes) == {"FormatError", "Query", "Rotator"}
    for name, cls in classes.items():
        runtime = getattr(quant, name)
        assert runtime.__module__ == "rabitqlib.quant"
        assert normalized(ast.get_docstring(cls)) in normalized(runtime.__doc__)
    checked = {
        "Rotator": ("rotate", "from_bytes", "__init__"),
        "Query": (
            "__init__",
            "set_centroid",
            "estimate_bitset",
            "estimate_codes",
            "refine",
            "refine_packed",
        ),
    }
    for name, signatures in checked.items():
        cls = getattr(quant, name)
        members = stub_functions(classes[name].body)
        public = {key for key in vars(cls) if not key.startswith("_")}
        assert public <= set(members), public - set(members)
        for member, node in members.items():
            runtime = getattr(cls, member)
            documentation = ast.get_docstring(node)
            if documentation is not None:
                assert normalized(documentation) in normalized(runtime.__doc__), member
            if member in signatures:
                assert runtime_parameters(runtime) == stub_parameters(node), member


def test_documentation_examples():
    runner = doctest.DocTestRunner()
    parser = doctest.DocTestParser()
    for name, documentation in (
        ("quant", quant.__doc__),
        ("Rotator", quant.Rotator.__doc__),
        ("Query", quant.Query.__doc__),
    ):
        runner.run(parser.get_doctest(documentation, {}, name, None, 0))
    assert runner.failures == 0
    assert runner.tries > 0
