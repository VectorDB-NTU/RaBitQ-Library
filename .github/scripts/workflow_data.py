"""Load GitHub Actions workflows as data for CI contract tests."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"


def load_workflow(filename):
    """Parse a workflow, restoring the `on` key that YAML 1.1 reads as true."""
    data = yaml.safe_load((WORKFLOWS / filename).read_text())
    if True in data:
        data["on"] = data.pop(True)
    return data


def job(workflow, name):
    return workflow["jobs"][name]


def needs(job_data):
    value = job_data.get("needs", [])
    return [value] if isinstance(value, str) else list(value)


def steps(job_data):
    return job_data.get("steps", [])


def step(job_data, name):
    """Return the single step with this name."""
    matches = [entry for entry in steps(job_data) if entry.get("name") == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one step named {name!r}, found {len(matches)}")
    return matches[0]


def runs(job_data):
    """Return every shell command of a job, in step order."""
    return [entry["run"] for entry in steps(job_data) if "run" in entry]


def uses(job_data, action):
    """Return the steps that use an action, ignoring its version suffix."""
    return [
        entry
        for entry in steps(job_data)
        if entry.get("uses", "").split("@", 1)[0] == action
    ]
