"""Decide whether a tested main commit has a new release to publish."""

import json
import os
import re
import subprocess
import tomllib
from pathlib import Path


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def stable_version(value):
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError(
            f"Automatic releases require a stable X.Y.Z version: {value!r}"
        )
    return tuple(map(int, value.split(".")))


def prepare(event_name, event, repository, ref_name):
    sha = command("git", "rev-parse", "HEAD")
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    tag = f"v{version}"
    result = {"sha": sha, "tag": tag, "build": "true", "publish": "false"}
    cmake = Path("CMakeLists.txt").read_text()
    if not re.search(rf"project\(RaBitQLib VERSION {re.escape(version)}\s", cmake):
        raise ValueError("CMake and Python package versions must match")

    if event_name == "push":
        if ref_name != tag:
            raise ValueError(f"Release tag {ref_name!r} must match {tag!r}")
        result["publish"] = "true"
        return result
    if event_name != "workflow_run":
        return result  # PRs and manual dispatch only build wheels.

    result["build"] = "false"
    run = event["workflow_run"]
    if (
        run["event"] != "push"
        or run["head_branch"] != "main"
        or run["head_repository"]["full_name"] != repository
        or run["conclusion"] != "success"
        or run["head_sha"] != sha
    ):
        return result

    for workflow in ("test.yaml", "python.yml"):
        response = json.loads(
            command(
                "gh",
                "api",
                f"repos/{repository}/actions/workflows/{workflow}/runs?event=push&head_sha={sha}&per_page=100",
            )
        )
        runs = [r for r in response["workflow_runs"] if r["head_branch"] == "main"]
        if (
            not runs
            or max(runs, key=lambda r: r["run_number"])["conclusion"] != "success"
        ):
            print(f"Waiting for successful {workflow} on {sha}")
            return result

    current = stable_version(version)
    tags = command("git", "tag", "--list", "v*").splitlines()
    if tag in tags:
        # Allow rerunning a partially completed release, but never retag a commit.
        if command("git", "rev-list", "-n", "1", tag) != sha:
            return result
        releases = json.loads(
            command(
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repository}/releases?per_page=100",
            )
        )
        if any(r["tag_name"] == tag for page in releases for r in page):
            return result
    else:
        previous = [
            stable_version(t[1:]) for t in tags if re.fullmatch(r"v\d+\.\d+\.\d+", t)
        ]
        if previous and current <= max(previous):
            return result

    result.update(build="true", publish="true")
    return result


if __name__ == "__main__":
    outputs = prepare(
        os.environ["GITHUB_EVENT_NAME"],
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_REF_NAME"],
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key, value in outputs.items():
            print(f"{key}={value}", file=output)
    print(outputs)
