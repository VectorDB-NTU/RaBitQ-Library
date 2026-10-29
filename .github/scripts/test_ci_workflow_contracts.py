"""Check lightweight CI coverage and workflow scheduling contracts."""

import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from ci_changes import GROUPS, classify

ROOT = Path(__file__).resolve().parents[2]


class WorkflowContractsTest(unittest.TestCase):
    def test_ctest_invocations_reject_empty_selections(self):
        workflow = (ROOT / ".github/workflows/test.yaml").read_text()
        commands = re.findall(r"\bctest\s+[^\n]+", workflow)
        self.assertTrue(commands)
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(
                    re.findall(r"--no-tests(?:=|\s+)(\w+)", command), ["error"]
                )

        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["ctest", "--test-dir", directory, "--no-tests=error"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No tests were found", result.stdout + result.stderr)

    def test_wheels_preserve_abi_coverage_and_release_gates(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        wheels = config["tool"]["cibuildwheel"]
        # Keep the full installed-wheel suite and platform repair hooks.
        self.assertIn("pytest {project}/tests/python", wheels["test-command"])
        self.assertIn("delocate-wheel", wheels["macos"]["repair-wheel-command"])
        self.assertIn(
            ".github/scripts/repair_windows_wheel.py",
            wheels["windows"]["repair-wheel-command"],
        )
        self.assertIn("scripts/build-macos-openmp.sh", wheels["macos"]["before-all"])
        self.assertEqual(workflow.count("pattern: '*-wheels'"), 2)
        artifacts = set()
        for prefix, tag, check_name in (
            ("linux", "manylinux_x86_64", "Linux x86-64 wheels"),
            (
                "linux-arm64",
                "manylinux_aarch64",
                "Linux ARM64 wheels (CPython 3.11–3.14)",
            ),
            ("windows", "win_amd64", "Windows x86-64 wheels"),
            ("macos", "macosx_arm64", "macOS arm64 wheels (CPython 3.11–3.14)"),
        ):
            with self.subTest(platform=prefix):
                build = workflow.split(f"  {prefix}-wheel-build:\n", 1)[1].split(
                    f"  {prefix}-wheels:\n", 1
                )[0]
                selection = re.search(
                    r"CIBW_BUILD: \$\{\{ github.event_name == 'pull_request' "
                    r"&& '([^']+)' \|\| '([^']+)' \}\}",
                    build,
                )
                self.assertIsNotNone(selection)
                supported = {
                    identifier
                    for identifier in wheels["build"].split()
                    if identifier.endswith(f"-{tag}")
                }
                for patterns, abis in zip(
                    selection.groups(), ((311, 314), (311, 312, 313, 314)), strict=True
                ):
                    selected = {
                        identifier
                        for identifier in supported
                        if any(
                            fnmatch.fnmatchcase(identifier, p) for p in patterns.split()
                        )
                    }
                    self.assertEqual(selected, {f"cp{abi}-{tag}" for abi in abis})
                self.assertNotIn("CIBW_TEST", build)
                self.assertNotIn("continue-on-error", build)
                self.assertNotIn("matrix", build)
                artifact = re.search(r"          name: (\S+)", build).group(1)
                self.assertTrue(fnmatch.fnmatchcase(artifact, "*-wheels"))
                self.assertNotIn(artifact, artifacts)
                artifacts.add(artifact)
                gate = re.search(
                    rf"^  {prefix}-wheels:\n(.*?)(?=^  [\w-]+:|\Z)",
                    workflow,
                    re.MULTILINE | re.DOTALL,
                ).group(1)
                self.assertIn(f"name: {check_name}", gate)
                self.assertIn(f"needs: {prefix}-wheel-build", gate)
                self.assertIn(
                    f"BUILD_RESULT: ${{{{ needs.{prefix}-wheel-build.result }}}}", gate
                )
                command = gate.split("        run: ", 1)[1].strip()
                for result in ("success", "failure", "cancelled", "skipped", ""):
                    with self.subTest(result=result):
                        verdict = subprocess.run(
                            ["bash", "-e", "-c", command],
                            env=dict(os.environ, BUILD_RESULT=result),
                            capture_output=True,
                            check=False,
                        )
                        self.assertEqual(verdict.returncode == 0, result == "success")

    def test_installed_typing_checks_are_required(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        test_dependencies = config["project"]["optional-dependencies"]["test"]
        wheel_dependencies = config["tool"]["cibuildwheel"]["test-requires"]
        checker = [value for value in test_dependencies if value.startswith("mypy==")]
        self.assertEqual(len(checker), 1)
        self.assertFalse(any(value.startswith("mypy") for value in wheel_dependencies))
        wheels = config["tool"]["cibuildwheel"]
        typing_overrides = [
            o for o in wheels["overrides"] if checker[0] in o.get("test-requires", [])
        ]
        self.assertEqual(len(typing_overrides), 1)
        override = typing_overrides[0]
        self.assertEqual(override["inherit"]["test-requires"], "append")
        self.assertEqual(override["inherit"]["test-command"], "append")
        self.assertIn("pytest {project}/tests/test_mypy.py", override["test-command"])
        self.assertEqual(
            {
                identifier
                for identifier in wheels["build"].split()
                if fnmatch.fnmatchcase(identifier, override["select"])
            },
            {f"cp{abi}-manylinux_x86_64" for abi in (311, 312, 313, 314)},
        )
        self.assertIn(
            "--ignore={project}/tests/python/test_benchmark_protocol.py",
            wheels["test-command"],
        )
        self.assertFalse(
            any(value.startswith("mypy") for value in config["project"]["dependencies"])
        )
        workflow = (ROOT / ".github/workflows/python.yml").read_text()
        build = workflow.split("  python-build:\n", 1)[1].split(
            "  python-result:\n", 1
        )[0]
        self.assertIn('pip install ".[test,typing]"', build)
        self.assertIn(
            "python -m pytest tests/python/test_editor_support.py::test_installed_public_declarations",
            build,
        )
        self.assertNotIn("continue-on-error", build)
        self.assertIn(
            "pytest {project}/tests/python",
            config["tool"]["cibuildwheel"]["test-command"],
        )
        self.assertNotIn(
            "--ignore={project}/tests/python/test_editor_support.py",
            config["tool"]["cibuildwheel"]["test-command"],
        )
        for path in (
            "python_bindings/__init__.pyi",
            "tests/python/typing/public_api.py",
        ):
            with self.subTest(path=path):
                self.assertLessEqual(
                    {"python_build", "python_quality", "wheels"}, classify([path])
                )
        self.assertEqual(
            {"python_quality", "python_build", "wheels"},
            classify(["tests/test_mypy.py"]),
        )

    def test_nonrelease_push_keeps_typing_checks_when_wheels_skip(self):
        from test_prepare_release import ReleaseTest

        release = ReleaseTest().run_prepare(tags="v0.3.3", tag_sha="b" * 40)
        self.assertEqual(release["build"], "false")
        workflow = (ROOT / ".github/workflows/python.yml").read_text()
        build = workflow.split("  python-build:\n", 1)[1].split(
            "  python-result:\n", 1
        )[0]
        steps = re.findall(
            r"^      - name: (.*?)(?=^      - |\Z)",
            build,
            re.MULTILINE | re.DOTALL,
        )
        typing_steps = [step for step in steps if "tests/test_mypy.py" in step]
        self.assertEqual(len(typing_steps), 1)
        self.assertIn("if: github.event_name == 'push'", typing_steps[0])
        self.assertIn(
            "run: python -m pytest tests/python/test_editor_support.py tests/test_mypy.py",
            typing_steps[0],
        )
        declaration_steps = [
            step for step in steps if "::test_installed_public_declarations" in step
        ]
        self.assertEqual(len(declaration_steps), 1)
        self.assertIn("if: github.event_name == 'pull_request'", declaration_steps[0])
        for path in (
            "tests/test_mypy.py",
            "tests/python/test_editor_support.py",
            "python_bindings/__init__.pyi",
        ):
            with self.subTest(path=path):
                self.assertIn("python_build", classify([path]))

    def test_pyright_checks_are_required_and_scoped(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            set(config["tool"]["pyright"]["include"]),
            {
                "tests/python/typing/public_api.py",
                "tests/python/test_editor_support.py",
                "tests/test_pyright.py",
            },
        )
        self.assertEqual(config["tool"]["pyright"]["typeCheckingMode"], "standard")
        self.assertTrue(
            any(
                dependency.startswith("pyright==")
                for dependency in config["project"]["optional-dependencies"]["typing"]
            )
        )
        self.assertFalse(
            any(
                dependency.startswith("pyright")
                for dependency in config["project"]["dependencies"]
            )
        )
        self.assertFalse(
            any(
                dependency.startswith("pyright")
                for dependency in config["tool"]["cibuildwheel"]["test-requires"]
            )
        )
        workflow = (ROOT / ".github/workflows/python.yml").read_text()
        build = workflow.split("  python-build:\n", 1)[1].split(
            "  python-result:\n", 1
        )[0]
        self.assertIn("actions/setup-node@v7", build)
        self.assertIn('node-version: "24"', build)
        self.assertIn('pip install ".[test,typing]"', build)
        self.assertIn("python -m pytest tests/test_pyright.py", build)
        self.assertNotIn("continue-on-error", build)
        self.assertEqual(
            classify(["tests/test_pyright.py"]), {"python_quality", "python_build"}
        )

    def test_tidy_rejects_empty_scope_and_propagates_analysis_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            compiler = work / "compiler"
            compiler.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' '#include <...> search starts here:' "
                "' /test/include' 'End of search list.'\n"
            )
            compiler.chmod(0o755)
            runner = work / "runner"
            marker = work / "analyzed"
            runner.write_text(
                f"#!{sys.executable}\n"
                "import os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['TIDY_MARKER']).touch()\n"
                "sys.exit(int(os.environ['ANALYSIS_STATUS']))\n"
            )
            runner.chmod(0o755)
            for entries, status, selected in (
                ([], 0, False),
                ([{"directory": str(work), "file": "foreign.cpp"}], 0, False),
                ([{"directory": str(ROOT), "file": "src/core/api.cpp"}], 0, True),
                ([{"directory": str(ROOT / "src"), "file": "core/api.cpp"}], 0, True),
                ([{"directory": str(ROOT), "file": "src/core/api.cpp"}], 7, True),
            ):
                with self.subTest(entries=entries, status=status):
                    (work / "compile_commands.json").write_text(json.dumps(entries))
                    marker.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["bash", str(ROOT / "scripts/check-tidy.sh"), str(work)],
                        env=dict(
                            os.environ,
                            CXX=str(compiler),
                            RUN_CLANG_TIDY=str(runner),
                            PYTHON=sys.executable,
                            TIDY_MARKER=str(marker),
                            ANALYSIS_STATUS=str(status),
                        ),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode == 0, selected and status == 0)
                    self.assertEqual(marker.exists(), selected)
                    if not selected:
                        self.assertIn(
                            "no matching first-party translation units", result.stderr
                        )
                    if result.returncode:
                        self.assertNotIn("clang-tidy passed", result.stdout)

    def test_format_checker_propagates_file_list_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "bin").mkdir()
            git = root / "bin/git"
            git.write_text(f"#!{sys.executable}\nprint({str(root)!r})\n")
            git.chmod(0o755)
            formatter = root / "bin/formatter"
            log = root / "formatted.json"
            formatter.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print('clang-format version 23.1.4')\n"
                "else:\n"
                "    Path(os.environ['FORMAT_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
                "    sys.exit(int(os.environ.get('FORMAT_STATUS', '0')))\n"
            )
            formatter.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                CLANG_FORMAT=str(formatter),
                FORMAT_LOG=str(log),
            )
            helper = root / "scripts/clang-format-files.sh"
            filename = str(root / "space and\nnewline.cpp")
            for mode, content, status, expected in (
                ("missing", None, "0", False),
                ("failed", "sys.exit(7)\n", "0", False),
                (
                    "partial",
                    f"sys.stdout.write({filename!r} + '\\0')\nsys.exit(7)\n",
                    "0",
                    False,
                ),
                ("empty", "sys.exit(0)\n", "0", True),
                ("valid", f"sys.stdout.write({filename!r} + '\\0')\n", "0", True),
                (
                    "format-error",
                    f"sys.stdout.write({filename!r} + '\\0')\n",
                    "1",
                    False,
                ),
            ):
                with self.subTest(mode=mode):
                    if content is not None:
                        helper.write_text(f"#!{sys.executable}\nimport sys\n" + content)
                        helper.chmod(0o755)
                    log.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["bash", str(ROOT / "scripts/check-format.sh")],
                        env=dict(env, FORMAT_STATUS=status),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode == 0, expected, result.stderr)
                    if mode in ("valid", "format-error"):
                        self.assertEqual(
                            json.loads(log.read_text()),
                            ["--dry-run", "--Werror", filename],
                        )
                    else:
                        self.assertFalse(log.exists())

    def test_benchmark_support_changes_keep_protocol_coverage_without_builds(self):
        for path in sorted((ROOT / "benchmarks").glob("*.py")):
            with self.subTest(path=path.name):
                self.assertEqual(
                    classify([str(path.relative_to(ROOT))]), {"python_quality"}
                )
        self.assertEqual(
            classify(["tests/python/test_benchmark_protocol.py"]), {"python_quality"}
        )
        self.assertEqual(classify(["benchmarks/unknown.py"]), set(GROUPS))
        self.assertIn(
            "cpp",
            classify(["benchmarks/build_indexes.py", "src/core/api.cpp"]),
        )
        workflow = (ROOT / ".github/workflows/python.yml").read_text()
        quality = workflow.split("  python-quality:\n", 1)[1].split(
            "  python-build:\n", 1
        )[0]
        self.assertIn("tests/python/test_benchmark_protocol.py", quality)
        self.assertIn("numpy", quality)
        self.assertIn("pytest", quality)
        self.assertNotIn("pip install .", quality)

    def test_python_checker_covers_every_benchmark_python_file(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "ruff"
            log = Path(directory) / "calls.jsonl"
            helper.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print('ruff 0.16.1')\n"
                "else:\n"
                "    with open(os.environ['RUFF_AUDIT_LOG'], 'a') as stream:\n"
                "        stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            helper.chmod(0o755)
            subprocess.run(
                ["bash", str(ROOT / "scripts/check-python.sh")],
                env=dict(os.environ, RUFF=str(helper), RUFF_AUDIT_LOG=str(log)),
                check=True,
                capture_output=True,
                text=True,
            )
            calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        for call in calls:
            covered = set()
            for argument in call:
                path = ROOT / argument
                if path.is_dir():
                    covered.update(path.rglob("*.py"))
                elif path.suffix == ".py":
                    covered.add(path)
            self.assertLessEqual(set((ROOT / "benchmarks").rglob("*.py")), covered)

    def test_concurrency_groups_distinguish_forks_and_reuse_one_pr(self):
        def group(template, number, sha, workflow):
            context = {
                "github.event.pull_request.number": number,
                "github.head_ref": "main",
                "github.sha": sha,
                "github.workflow": workflow,
            }

            def expand(match):
                return str(
                    next(
                        context[key.strip()]
                        for key in match.group(1).split("||")
                        if context[key.strip()]
                    )
                )

            return re.sub(r"\$\{\{(.*?)\}\}", expand, template)

        for filename in ("test.yaml", "python.yml"):
            text = (ROOT / ".github/workflows" / filename).read_text()
            template = re.search(r"^  group: (.+)$", text, re.MULTILINE).group(1)
            with self.subTest(workflow=filename):
                self.assertNotEqual(
                    group(template, 201, "a", filename),
                    group(template, 202, "b", filename),
                )
                self.assertEqual(
                    group(template, 201, "a", filename),
                    group(template, 201, "b", filename),
                )
                self.assertNotEqual(
                    group(template, 0, "a", filename),
                    group(template, 0, "b", filename),
                )

    def test_windows_consumers_cover_each_component_and_openmp_mode(self):
        workflow = (ROOT / ".github/workflows/test.yaml").read_text()
        block = workflow.split("  core-consumer:\n", 1)[1].split(
            "  core-consumer-result:\n", 1
        )[0]
        variants = re.findall(
            r'- os: windows-2025-vs2026\n\s+openmp: "(ON|OFF)"'
            r"\n\s+component: (core|fastscan)",
            block,
        )
        self.assertCountEqual(
            variants,
            [
                (mode, component)
                for mode in ("ON", "OFF")
                for component in ("core", "fastscan")
            ],
        )
        windows = block.split(
            "      - name: Test external core-only integrations (Windows)\n", 1
        )[1].split("      - name: Report compiler cache statistics", 1)[0]
        self.assertIn("-DRABITQ_COMPONENT=${{ matrix.component }}", windows)
        self.assertIn(
            "-DRABITQ_RUN_REFERENCE_TESTS=${{ matrix.openmp == 'OFF' && "
            "matrix.component != 'core' && 'ON' || 'OFF' }}",
            windows,
        )

    def test_consumer_reference_ownership_preserves_integration_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "calls.bin"
            for tool in ("cmake", "ctest"):
                helper = root / tool
                helper.write_text(
                    r"""#!/usr/bin/env bash
set -euo pipefail
# CMake emits nonempty arguments here; a double NUL separates invocations.
printf '%s\0' "${0##*/}" "$@" >> "$CONSUMER_TRACE"
printf '\0' >> "$CONSUMER_TRACE"
args=("$@")
for ((i = 0; i + 1 < ${#args[@]}; ++i)); do
    case "${args[i]}" in
        --prefix) mkdir -p "${args[i+1]}" ;;
        -B)
            case "${args[i+1]##*/}" in
                invalid-python) printf '%s\n' 'requires RABITQ_BUILD_INDEX=ON' >&2; exit 1 ;;
                invalid-index) printf '%s\n' 'requires RABITQ_BUILD_FASTSCAN=ON' >&2; exit 1 ;;
                invalid-openmp) printf '%s\n' 'ANN indexes require RABITQ_USE_OPENMP=ON' >&2; exit 1 ;;
                missing-index|missing-fastscan) printf '%s\n' 'rabitqlib component not found' >&2; exit 1 ;;
            esac ;;
        --test-dir)
            if [[ "${args[i+1]##*/}" == producer && "${FAIL_REFERENCE:-0}" == 1 ]]; then
                exit 7
            fi ;;
    esac
done
"""
                )
                helper.chmod(0o755)
            for index, (openmp, reference, component, expected, fail) in enumerate(
                [
                    ("ON", "OFF", None, {"core": "OFF", "fastscan": "OFF"}, False),
                    ("OFF", "ON", None, {"core": "OFF", "fastscan": "ON"}, False),
                    ("ON", None, None, {"core": "OFF", "fastscan": "ON"}, False),
                    ("OFF", "OFF", "core", {"core": "OFF"}, False),
                    ("ON", None, "core", {"core": "ON"}, False),
                    ("OFF", "ON", "fastscan", {"fastscan": "ON"}, False),
                    ("OFF", "ON", "fastscan", {"fastscan": "ON"}, True),
                ]
            ):
                with self.subTest(
                    openmp=openmp, reference=reference, component=component, fail=fail
                ):
                    log.unlink(missing_ok=True)
                    command = [
                        "cmake",
                        f"-DRABITQ_SOURCE_DIR={ROOT}",
                        f"-DWORK_DIR={root / str(index)}",
                        f"-DCMAKE_COMMAND={root / 'cmake'}",
                        f"-DCMAKE_CTEST_COMMAND={root / 'ctest'}",
                        f"-DRABITQ_USE_OPENMP={openmp}",
                    ]
                    if reference is not None:
                        command.append(f"-DRABITQ_RUN_REFERENCE_TESTS={reference}")
                    if component is not None:
                        command.append(f"-DRABITQ_COMPONENT={component}")
                    result = subprocess.run(
                        [*command, "-P", str(ROOT / "tests/core_consumer/run.cmake")],
                        env=dict(
                            os.environ,
                            CONSUMER_TRACE=str(log),
                            FAIL_REFERENCE=str(int(fail)),
                        ),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode == 0, not fail, result.stderr)
                    if fail:
                        self.assertIn("Command failed (7)", result.stderr)
                        continue
                    calls = [
                        record.decode().split("\0")
                        for record in log.read_bytes().split(b"\0\0")
                        if record
                    ]
                    producers = {}
                    tests = []
                    for tool, *args in calls:
                        if tool == "cmake" and "-B" in args:
                            build = Path(args[args.index("-B") + 1])
                            if build.name == "producer":
                                producers[build.parent.name] = next(
                                    arg.split("=", 1)[1]
                                    for arg in args
                                    if arg.startswith("-DRABITQ_BUILD_TESTS=")
                                )
                        if tool == "ctest":
                            tests.append(Path(args[args.index("--test-dir") + 1]))
                    self.assertEqual(producers, expected)
                    self.assertEqual(
                        {path.parent.name for path in tests if path.name == "producer"},
                        {name for name, enabled in expected.items() if enabled == "ON"},
                    )
                    for name in expected:
                        self.assertEqual(
                            {
                                path.name
                                for path in tests
                                if path.parent.name == name and path.name != "producer"
                            },
                            {"ADD_SUBDIRECTORY", "FETCHCONTENT", "PACKAGE"},
                        )
                    self.assertEqual(
                        any(path.name == "benchmark" for path in tests),
                        "core" in expected,
                    )

    def test_component_partition_rejects_unknown_or_empty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            for component in ("unknown", ""):
                with self.subTest(component=component):
                    result = subprocess.run(
                        [
                            "cmake",
                            f"-DRABITQ_SOURCE_DIR={ROOT}",
                            f"-DWORK_DIR={directory}",
                            f"-DRABITQ_COMPONENT={component}",
                            "-P",
                            str(ROOT / "tests/core_consumer/run.cmake"),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "RABITQ_COMPONENT must be core or fastscan", result.stderr
                    )
                    self.assertFalse(list(Path(directory).iterdir()))

    def test_documentation_summary_is_available_for_non_doc_prs(self):
        text = (ROOT / ".github/workflows/docs.yml").read_text()
        triggers = text.split("concurrency:", 1)[0]
        self.assertNotIn("paths:", triggers)
        self.assertNotIn("paths-ignore:", triggers)
        self.assertIn("  pull_request:", triggers)
        self.assertIn("  workflow_dispatch:", triggers)
        self.assertIn("uses: ./.github/workflows/changes.yml", text)
        build = text.split("  build:\n", 1)[1].split("  docs-result:\n", 1)[0]
        self.assertIn("needs: changes", build)
        self.assertIn("needs.changes.outputs.docs == 'true'", build)
        self.assertIn("needs.changes.result != 'success'", build)
        self.assertEqual(classify(["docs/docs/quick_start.md"]), {"docs"})


if __name__ == "__main__":
    unittest.main()
