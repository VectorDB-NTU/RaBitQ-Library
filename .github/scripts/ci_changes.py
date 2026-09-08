"""Select CI job groups from the complete push or pull-request diff."""

import json
import os
import subprocess
import tomllib
from pathlib import Path

GROUPS = ("cpp", "python_quality", "python_build", "shell", "wheels")


def classify(paths):
    selected = set()
    for path in paths:
        if path.startswith("docs/") or path.endswith(".md"):
            continue
        if path in (".github/workflows/docs.yml", ".github/release.yml", "LICENSE"):
            continue
        if path in (
            ".github/workflows/changes.yml",
            ".github/scripts/ci_changes.py",
            ".github/scripts/test_ci_changes.py",
            ".editorconfig",
        ):
            return set(GROUPS)
        if path in ("CMakeLists.txt", "pyproject.toml") or path.startswith("cmake/"):
            selected.update(("cpp", "python_quality", "python_build", "wheels"))
        elif path.startswith(("include/", "src/")):
            selected.update(("cpp", "python_build", "wheels"))
        elif path.startswith("python_bindings/"):
            selected.update(("python_quality", "python_build", "wheels"))
            if not path.endswith(".py"):
                selected.add("cpp")
        elif (
            path.startswith(
                (
                    "tests/unit/",
                    "tests/integration/",
                    "tests/common/",
                    "tests/consumer/",
                    "sample/cpp/",
                )
            )
            or path == "tests/CMakeLists.txt"
        ):
            selected.add("cpp")
        elif path.startswith("tests/python/"):
            selected.update(("python_quality", "python_build", "wheels"))
        elif path.startswith(("python/", "sample/python/", ".github/scripts/")):
            selected.add("python_quality")
        elif path == ".github/workflows/test.yaml":
            selected.update(("cpp", "shell"))
        elif path == ".github/workflows/python.yml":
            selected.update(("python_quality", "python_build"))
        elif path == ".github/workflows/release.yml":
            selected.update(("python_quality", "wheels"))
        elif path in (".clang-format", ".clang-tidy"):
            selected.add("cpp")
        elif path.startswith("scripts/") and path.endswith(".sh"):
            selected.add("shell")
            if path == "scripts/check-python.sh":
                selected.add("python_quality")
            else:
                selected.add("cpp")
        else:
            # Unknown inputs may affect builds; never silently skip their checks.
            return set(GROUPS)
    return selected


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip("\n")


def detect(event_name, event, ref_name):
    if event_name not in ("push", "pull_request") or event.get("ref", "").startswith(
        "refs/tags/"
    ):
        return set(GROUPS)  # Always validate manual/tag releases.
    if event_name == "pull_request":
        base = git("merge-base", event["pull_request"]["base"]["sha"], "HEAD")
    else:
        base = event["before"]
        if not base or set(base) == {"0"}:
            return set(GROUPS)
    paths = git("diff", "--name-only", "--no-renames", "-z", base, "HEAD").split("\0")
    selected = classify(path for path in paths if path)
    if event_name == "push" and ref_name == "main":
        version = tomllib.loads(Path("pyproject.toml").read_text())["project"][
            "version"
        ]
        if not git("tag", "--list", f"v{version}"):
            # A docs-only fix after failed release CI must not bypass that CI.
            return set(GROUPS)
    return selected


if __name__ == "__main__":
    selected = detect(
        os.environ["GITHUB_EVENT_NAME"],
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
        os.environ["GITHUB_REF_NAME"],
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for group in GROUPS:
            print(f"{group}={str(group in selected).lower()}", file=output)
    print(
        "Selected checks:", ", ".join(sorted(selected)) or "none (documentation only)"
    )
