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

import yaml
from ci_changes import GROUPS, classify
from workflow_data import ROOT, job, load_workflow, needs, runs, step, steps, uses


class WorkflowContractsTest(unittest.TestCase):
    def test_ctest_invocations_reject_empty_selections(self):
        workflow = load_workflow("test.yaml")
        commands = [
            command
            for job_data in workflow["jobs"].values()
            for run in runs(job_data)
            for command in re.findall(r"\bctest\s+[^\n]+", run)
        ]
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
        workflow = load_workflow("release.yml")
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
        patterns = [
            entry["with"]["pattern"]
            for job_data in workflow["jobs"].values()
            for entry in uses(job_data, "actions/download-artifact")
        ]
        self.assertEqual(patterns, ["*-wheels*", "*-wheels*"])
        pr_selection = (
            r"github\.event_name == 'pull_request' && '([^']+)' \|\| '([^']+)'"
        )

        def selections(build, event):
            """Return the cibuildwheel identifier patterns and artifact names."""
            cibw = next(
                entry
                for entry in steps(build)
                if "pypa/cibuildwheel" in entry.get("uses", "")
            )["env"]["CIBW_BUILD"]
            artifact = uses(build, "actions/upload-artifact")[0]["with"]["name"]
            matrix = build.get("strategy", {}).get("matrix")
            if matrix is None:
                pr, release = re.search(pr_selection, cibw).groups()
                return [(pr if event == "pull_request" else release, artifact)]
            # One runner per ABI: the matrix itself carries the event-specific list.
            self.assertFalse(build["strategy"]["fail-fast"])
            pr, release = re.search(pr_selection, matrix["abi"]).groups()
            values = json.loads(pr if event == "pull_request" else release)
            return [
                (
                    cibw.replace("${{ matrix.abi }}", abi),
                    artifact.replace("${{ matrix.abi }}", abi),
                )
                for abi in values
            ]

        artifacts = set()
        verdicts = set()
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
                build = job(workflow, f"{prefix}-wheel-build")
                supported = {
                    identifier
                    for identifier in wheels["build"].split()
                    if identifier.endswith(f"-{tag}")
                }
                for event, abis in (
                    ("pull_request", (311, 314)),
                    ("push", (311, 312, 313, 314)),
                ):
                    selected = set()
                    for patterns_text, artifact in selections(build, event):
                        selected |= {
                            identifier
                            for identifier in supported
                            if any(
                                fnmatch.fnmatchcase(identifier, pattern)
                                for pattern in patterns_text.split()
                            )
                        }
                        self.assertTrue(fnmatch.fnmatchcase(artifact, "*-wheels*"))
                        if event == "push":
                            self.assertNotIn(artifact, artifacts)
                            artifacts.add(artifact)
                    self.assertEqual(selected, {f"cp{abi}-{tag}" for abi in abis})
                self.assertNotIn("continue-on-error", build)
                for entry in steps(build):
                    self.assertNotIn("CIBW_TEST", " ".join(entry.get("env", {})))
                    self.assertNotIn("continue-on-error", entry)
                gate = job(workflow, f"{prefix}-wheels")
                self.assertEqual(gate["name"], check_name)
                # The gate always runs so that a build skipped because the
                # eligibility check failed cannot satisfy the required check.
                self.assertEqual(
                    needs(gate), ["prepare", "changes", f"{prefix}-wheel-build"]
                )
                self.assertEqual(gate["if"], "${{ always() }}")
                (verdict_step,) = steps(gate)
                self.assertEqual(
                    verdict_step["env"],
                    {
                        "PREPARE_RESULT": "${{ needs.prepare.result }}",
                        "BUILD_SELECTED": "${{ needs.prepare.outputs.build }}",
                        "CHANGES_RESULT": "${{ needs.changes.result }}",
                        "WHEELS_SELECTED": "${{ needs.changes.outputs.wheels }}",
                        "BUILD_RESULT": f"${{{{ needs.{prefix}-wheel-build.result }}}}",
                    },
                )
                verdicts.add(verdict_step["run"])
        # One shared verdict script, exercised over every input combination.
        (command,) = verdicts
        results = ("success", "failure", "cancelled", "skipped", "")
        for prepare in results:
            for selected in ("true", "false", ""):
                for changes in results:
                    for wheels in ("true", "false", ""):
                        for build in results:
                            if build == "success":
                                expected = True
                            elif build != "skipped":
                                expected = False
                            elif prepare == "skipped":
                                expected = True  # excluded by the workflow_run filter
                            else:
                                expected = prepare == "success" and (
                                    selected == "false"
                                    or (changes == "success" and wheels == "false")
                                )
                            with self.subTest(
                                prepare=prepare,
                                selected=selected,
                                changes=changes,
                                wheels=wheels,
                                build=build,
                            ):
                                verdict = subprocess.run(
                                    ["bash", "-e", "-c", command],
                                    env=dict(
                                        os.environ,
                                        PREPARE_RESULT=prepare,
                                        BUILD_SELECTED=selected,
                                        CHANGES_RESULT=changes,
                                        WHEELS_SELECTED=wheels,
                                        BUILD_RESULT=build,
                                    ),
                                    capture_output=True,
                                    check=False,
                                )
                                self.assertEqual(verdict.returncode == 0, expected)

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
        build = job(load_workflow("python.yml"), "python-build")
        commands = runs(build)
        self.assertTrue(any('pip install ".[test,typing]"' in run for run in commands))
        self.assertTrue(
            any(
                "python -m pytest tests/python/test_editor_support.py"
                "::test_installed_public_declarations" in run
                for run in commands
            )
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
        build = job(load_workflow("python.yml"), "python-build")
        typing_steps = [
            entry
            for entry in steps(build)
            if "tests/test_mypy.py" in entry.get("run", "")
        ]
        self.assertEqual(len(typing_steps), 1)
        self.assertEqual(typing_steps[0]["if"], "github.event_name == 'push'")
        self.assertTrue(
            typing_steps[0]["run"].startswith(
                "python -m pytest tests/python/test_editor_support.py tests/test_mypy.py"
            )
        )
        declaration_steps = [
            entry
            for entry in steps(build)
            if "::test_installed_public_declarations" in entry.get("run", "")
        ]
        self.assertEqual(len(declaration_steps), 1)
        self.assertEqual(
            declaration_steps[0]["if"], "github.event_name == 'pull_request'"
        )
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
        build = job(load_workflow("python.yml"), "python-build")
        (node,) = uses(build, "actions/setup-node")
        self.assertEqual(node["uses"], "actions/setup-node@v7")
        self.assertEqual(node["with"]["node-version"], "24")
        commands = runs(build)
        self.assertTrue(any('pip install ".[test,typing]"' in run for run in commands))
        self.assertTrue(
            any("python -m pytest tests/test_pyright.py" in run for run in commands)
        )
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
        commands = "\n".join(runs(job(load_workflow("python.yml"), "python-quality")))
        self.assertIn("tests/python/test_benchmark_protocol.py", commands)
        self.assertIn("numpy", commands)
        self.assertIn("pytest", commands)
        self.assertNotIn("pip install .", commands)

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
            template = load_workflow(filename)["concurrency"]["group"]
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
        consumers = job(load_workflow("test.yaml"), "core-consumer")
        variants = [
            (entry["openmp"], entry["component"])
            for entry in consumers["strategy"]["matrix"]["include"]
            if entry["os"] == "windows-2025-vs2026"
        ]
        self.assertCountEqual(
            variants,
            [
                (mode, component)
                for mode in ("ON", "OFF")
                for component in ("core", "fastscan")
            ],
        )
        windows = step(consumers, "Test external core-only integrations (Windows)")
        self.assertIn("-DRABITQ_COMPONENT=${{ matrix.component }}", windows["run"])
        self.assertIn(
            "-DRABITQ_RUN_REFERENCE_TESTS=${{ matrix.openmp == 'OFF' && "
            "matrix.component != 'core' && 'ON' || 'OFF' }}",
            windows["run"],
        )

    def test_source_integrations_keep_one_gcc_and_one_msvc_owner(self):
        workflow = load_workflow("test.yaml")
        consumers = job(workflow, "core-consumer")
        selection = (
            "-DRABITQ_INTEGRATIONS=${{ (matrix.os == 'ubuntu-latest' || "
            "matrix.integrations == 'ALL') && 'ALL' || 'PACKAGE' }}"
        )
        for platform in ("Linux", "macOS", "Windows"):
            with self.subTest(platform=platform):
                entry = step(
                    consumers, f"Test external core-only integrations ({platform})"
                )
                self.assertIn(selection, entry["run"])
        matrix = consumers["strategy"]["matrix"]
        owners = [
            entry for entry in matrix["include"] if entry.get("integrations") == "ALL"
        ]
        self.assertEqual(
            [(e["os"], e["openmp"], e["component"]) for e in owners],
            [("windows-2025-vs2026", "ON", "core")],
        )
        self.assertIn("ubuntu-latest", matrix["os"])
        self.assertEqual(matrix["openmp"], ["ON", "OFF"])
        clang = step(job(workflow, "clang"), "Test core and FastScan integrations")
        self.assertIn("-DRABITQ_INTEGRATIONS=PACKAGE", clang["run"])

    def test_compiling_unix_jobs_restore_a_compiler_cache(self):
        workflow = load_workflow("test.yaml")
        (restore,) = [
            entry
            for entry in yaml.safe_load(
                (ROOT / ".github/actions/setup-ccache/action.yml").read_text()
            )["runs"]["steps"]
            if "uses" in entry
        ]
        (save,) = [
            entry
            for entry in yaml.safe_load(
                (ROOT / ".github/actions/save-ccache/action.yml").read_text()
            )["runs"]["steps"]
            if "uses" in entry
        ]
        # Pull requests only restore, so they cannot evict the cache saved from main.
        self.assertTrue(restore["uses"].startswith("actions/cache/restore@"))
        self.assertTrue(save["uses"].startswith("actions/cache/save@"))
        self.assertEqual(
            save["if"], "github.event_name == 'push' && github.ref == 'refs/heads/main'"
        )
        self.assertEqual(
            save["with"],
            {"path": restore["with"]["path"], "key": restore["with"]["key"]},
        )
        self.assertIn("${{ github.sha }}", restore["with"]["key"])
        self.assertTrue(
            restore["with"]["key"].startswith(restore["with"]["restore-keys"])
        )
        keys = []
        for name, job_data in workflow["jobs"].items():
            commands = runs(job_data)
            compiles = any(
                "cmake --build" in run or "tests/core_consumer/run.cmake" in run
                for run in commands
            )
            if not compiles or job_data["runs-on"].startswith("windows"):
                continue
            setups = uses(job_data, "./.github/actions/setup-ccache")
            # run.cmake disables native tuning; direct source configurations must say so.
            portable = all(
                "-DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF" in run
                for run in commands
                if re.search(r"cmake\s+-S\s+\.\s", run)
            )
            with self.subTest(job=name):
                if not portable:
                    # ccache keys -march=native literally; another CPU could reuse it.
                    self.assertEqual(setups, [])
                    continue
                self.assertEqual(len(setups), 1, "a compiling job must set up ccache")
                setup = setups[0]
                position = steps(job_data).index(setup)
                first_build = next(
                    index
                    for index, entry in enumerate(steps(job_data))
                    if "cmake" in entry.get("run", "")
                    and ("-S " in entry["run"] or "run.cmake" in entry["run"])
                )
                self.assertLess(position, first_build)
                # Saving after every build step keeps all of the job's compilations.
                saves = uses(job_data, "./.github/actions/save-ccache")
                self.assertEqual(saves, [steps(job_data)[-1]], "save ccache last")
                self.assertIn("!cancelled()", saves[0]["if"])
                keys.append((job_data["runs-on"], setup["with"]["key"]))
        self.assertTrue(keys)
        self.assertEqual(
            uses(job(workflow, "ubuntu"), "./.github/actions/setup-ccache"), []
        )
        for workflow_path in (ROOT / ".github").rglob("*.y*ml"):
            with self.subTest(file=workflow_path.name):
                self.assertNotIn("actions/cache@", workflow_path.read_text())

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
            all_methods = {"ADD_SUBDIRECTORY", "FETCHCONTENT", "PACKAGE"}
            for index, (
                openmp,
                reference,
                component,
                integrations,
                expected,
                fail,
            ) in enumerate(
                [
                    (
                        "ON",
                        "OFF",
                        None,
                        None,
                        {"core": "OFF", "fastscan": "OFF"},
                        False,
                    ),
                    ("OFF", "ON", None, None, {"core": "OFF", "fastscan": "ON"}, False),
                    ("ON", None, None, None, {"core": "OFF", "fastscan": "ON"}, False),
                    ("OFF", "OFF", "core", None, {"core": "OFF"}, False),
                    ("ON", None, "core", None, {"core": "ON"}, False),
                    ("OFF", "ON", "fastscan", None, {"fastscan": "ON"}, False),
                    ("OFF", "ON", "fastscan", None, {"fastscan": "ON"}, True),
                    (
                        "ON",
                        "OFF",
                        None,
                        "ALL",
                        {"core": "OFF", "fastscan": "OFF"},
                        False,
                    ),
                    (
                        "OFF",
                        "ON",
                        None,
                        "PACKAGE",
                        {"core": "OFF", "fastscan": "ON"},
                        False,
                    ),
                    ("ON", "OFF", "core", "PACKAGE", {"core": "OFF"}, False),
                ]
            ):
                methods = {"PACKAGE"} if integrations == "PACKAGE" else all_methods
                with self.subTest(
                    openmp=openmp,
                    reference=reference,
                    component=component,
                    integrations=integrations,
                    fail=fail,
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
                    if integrations is not None:
                        command.append(f"-DRABITQ_INTEGRATIONS={integrations}")
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
                            methods,
                        )
                    self.assertEqual(
                        any(path.name == "benchmark" for path in tests),
                        "core" in expected,
                    )

    def test_component_partition_rejects_unknown_or_empty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            for option, value, message in (
                (
                    "RABITQ_COMPONENT",
                    "unknown",
                    "RABITQ_COMPONENT must be core or fastscan",
                ),
                ("RABITQ_COMPONENT", "", "RABITQ_COMPONENT must be core or fastscan"),
                (
                    "RABITQ_INTEGRATIONS",
                    "FETCHCONTENT",
                    "RABITQ_INTEGRATIONS must be ALL or PACKAGE",
                ),
                (
                    "RABITQ_INTEGRATIONS",
                    "",
                    "RABITQ_INTEGRATIONS must be ALL or PACKAGE",
                ),
            ):
                with self.subTest(option=option, value=value):
                    result = subprocess.run(
                        [
                            "cmake",
                            f"-DRABITQ_SOURCE_DIR={ROOT}",
                            f"-DWORK_DIR={directory}",
                            f"-D{option}={value}",
                            "-P",
                            str(ROOT / "tests/core_consumer/run.cmake"),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(list(Path(directory).iterdir()))

    def test_documentation_summary_is_available_for_non_doc_prs(self):
        workflow = load_workflow("docs.yml")
        triggers = workflow["on"]
        for event in triggers.values():
            if isinstance(event, dict):
                self.assertNotIn("paths", event)
                self.assertNotIn("paths-ignore", event)
        self.assertIn("pull_request", triggers)
        self.assertIn("workflow_dispatch", triggers)
        self.assertEqual(
            job(workflow, "changes")["uses"], "./.github/workflows/changes.yml"
        )
        build = job(workflow, "build")
        self.assertEqual(needs(build), ["changes"])
        self.assertIn("needs.changes.outputs.docs == 'true'", build["if"])
        self.assertIn("needs.changes.result != 'success'", build["if"])
        self.assertEqual(classify(["docs/docs/quick_start.md"]), {"docs"})


if __name__ == "__main__":
    unittest.main()
