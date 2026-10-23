"""Execute the metadata-only PR classifier with mocked GitHub API calls."""

import json
import subprocess
import unittest
from pathlib import Path

from ci_changes import classify

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/pr-labels.yml"
SCRIPT = "\n".join(
    line[12:]
    for line in WORKFLOW.read_text().split("          script: |\n", 1)[1].splitlines()
)

HARNESS = r"""
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const calls = [];
const warnings = [];
let lookups = 0;
const issues = {
  listLabelsOnIssue: () => {},
  getLabel: async args => {
    calls.push(['get', args.name]);
    const status = (input.lookupErrors || [])[lookups++];
    if (status) throw { status };
  },
  createLabel: async args => {
    calls.push(['create', args.name]);
    if (input.createError) throw { status: input.createError };
  },
  addLabels: async args => { calls.push(['add', args.labels]); },
  removeLabel: async args => { calls.push(['remove', args.name]); },
};
const github = {
  rest: {
    issues,
    pulls: { get: async () => ({ data: { title: input.title } }) },
  },
  paginate: async () => (input.labels || []).map(name => ({ name })),
};
const context = {
  repo: { owner: 'owner', repo: 'repo' },
  payload: { pull_request: { number: 42, title: 'stale event title' } },
};
const core = { warning: message => warnings.push(message) };
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
new AsyncFunction('github', 'context', 'core', input.script)(github, context, core)
  .then(() => process.stdout.write(JSON.stringify({ calls, warnings })))
  .catch(error => {
    process.stdout.write(JSON.stringify({ calls, warnings, error: error.status }));
  });
"""


def run_classifier(title, **state):
    result = subprocess.run(
        ["node", "-e", HARNESS],
        input=json.dumps(dict(script=SCRIPT, title=title, **state)),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


class PrLabelsTest(unittest.TestCase):
    def test_prefixes_scopes_and_breaking_markers(self):
        categories = {
            "feat": "added",
            "fix": "fixed",
            "perf": "performance",
            "docs": "documentation",
            "ci": "build",
            "build": "build",
            "deps": "dependencies",
            "chore": "maintenance",
            "refactor": "maintenance",
            "test": "maintenance",
            "style": "maintenance",
        }
        release_config = (ROOT / ".github/release.yml").read_text()
        for prefix, category in categories.items():
            for suffix in ("", "(ivf)", "!", "(python)!"):
                with self.subTest(prefix=prefix, suffix=suffix):
                    label = f"release:{category}"
                    result = run_classifier(f"{prefix}{suffix}: meaningful summary")
                    self.assertEqual(result["calls"][-1], ["add", [label]])
                    self.assertFalse(result["warnings"])
                    self.assertIn(f"- {label}\n", release_config)

    def test_edit_adds_new_category_before_removing_old_and_preserves_manual_labels(
        self,
    ):
        result = run_classifier(
            "fix(ivf): append metadata",
            labels=["release:documentation", "bug", "python", "duplicate", "custom"],
        )
        self.assertEqual(
            result["calls"],
            [
                ["get", "release:fixed"],
                ["add", ["release:fixed"]],
                ["remove", "release:documentation"],
            ],
        )

    def test_repeated_event_is_idempotent(self):
        result = run_classifier("docs: guide", labels=["release:documentation"])
        self.assertEqual(result["calls"], [])

    def test_unknown_titles_clear_only_managed_labels_and_warn(self):
        for title in (
            "Improve IVF",
            "fix:",
            "fix:  ",
            "prefix fix: title",
            "fix(): title",
        ):
            with self.subTest(title=title):
                result = run_classifier(
                    title, labels=["release:fixed", "bug", "release:custom"]
                )
                self.assertEqual(result["calls"], [["remove", "release:fixed"]])
                self.assertEqual(len(result["warnings"]), 1)

    def test_title_is_data_and_current_api_title_is_authoritative(self):
        result = run_classifier("docs: `$(command)` ${process.exit(1)} '")
        self.assertEqual(result["calls"][-1], ["add", ["release:documentation"]])

    def test_missing_repository_label_is_created(self):
        result = run_classifier("perf: reduce scratch", lookupErrors=[404])
        self.assertEqual(result["calls"][1], ["create", "release:performance"])
        self.assertEqual(result["calls"][-1][0], "add")

    def test_label_creation_race_requires_successful_recheck(self):
        result = run_classifier("ci: lint", lookupErrors=[404], createError=422)
        self.assertNotIn("error", result)
        self.assertEqual(
            [call[0] for call in result["calls"]], ["get", "create", "get", "add"]
        )
        result = run_classifier("ci: lint", lookupErrors=[404, 404], createError=422)
        self.assertEqual(result["error"], 404)
        self.assertNotIn("add", [call[0] for call in result["calls"]])

    def test_api_failures_are_not_silenced(self):
        for state, status in (
            ({"lookupErrors": [403]}, 403),
            ({"lookupErrors": [404], "createError": 500}, 500),
        ):
            with self.subTest(state=state):
                result = run_classifier(
                    "fix: regression", labels=["release:added"], **state
                )
                self.assertEqual(result["error"], status)
                self.assertNotIn("remove", [call[0] for call in result["calls"]])

    def test_workflow_uses_metadata_only_and_routes_to_regression_checks(self):
        text = WORKFLOW.read_text()
        self.assertIn("pull_request_target:", text)
        self.assertIn("types: [opened, edited, reopened, ready_for_review]", text)
        self.assertNotIn("actions/checkout", text)
        self.assertNotIn("${{", SCRIPT)
        self.assertEqual(
            classify([".github/workflows/pr-labels.yml"]), {"python_quality"}
        )


if __name__ == "__main__":
    unittest.main()
