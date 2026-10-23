"""Exercise LLVM package selection and failure propagation without installing tools."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ClangSetupTest(unittest.TestCase):
    def test_selected_tools_and_install_failures(self):
        action = (ROOT / ".github/actions/setup-clang/action.yml").read_text()
        self.assertIn('default: "false"', action)
        self.assertIn("FORMAT_ONLY: ${{ inputs.format-only }}", action)
        command = textwrap.dedent(action.split("      run: |\n", 1)[1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "commands.jsonl"
            for tool in (
                "sudo",
                "curl",
                "clang-format-23",
                "clang++-23",
                "clang-tidy-23",
                "cmake",
                "ninja",
            ):
                helper = root / tool
                helper.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, sys\n"
                    "from pathlib import Path\n"
                    "name = Path(sys.argv[0]).name\n"
                    "with open(os.environ['SETUP_LOG'], 'a') as stream:\n"
                    "    stream.write(json.dumps([name, *sys.argv[1:]]) + '\\n')\n"
                    "if name == os.environ.get('FAIL_TOOL'):\n"
                    "    sys.exit(7)\n"
                    "if os.environ.get('FAIL_TOOL') == 'install' and "
                    "sys.argv[1:3] == ['apt-get', 'install']:\n"
                    "    sys.exit(7)\n"
                    "if name == 'curl':\n"
                    "    print('test-key')\n"
                    "if name == 'sudo' and sys.argv[1] == 'tee':\n"
                    "    sys.stdin.read()\n"
                )
                helper.chmod(0o755)
            for mode, failure in (
                ("false", ""),
                ("true", ""),
                ("false", "install"),
                ("true", "install"),
                ("true", "clang-format-23"),
                ("false", "clang-tidy-23"),
            ):
                with self.subTest(format_only=mode, failure=failure):
                    log.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["bash", "-c", command],
                        env=dict(
                            os.environ,
                            PATH=str(root) + os.pathsep + os.environ["PATH"],
                            FORMAT_ONLY=mode,
                            SETUP_LOG=str(log),
                            FAIL_TOOL=failure,
                        ),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode == 0, not failure)
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    if failure:
                        self.assertFalse(any(call[0] == "ninja" for call in calls))
                        continue
                    packages = next(
                        call[4:]
                        for call in calls
                        if call[:4] == ["sudo", "apt-get", "install", "-y"]
                    )
                    expected = {"clang-format-23"}
                    if mode == "false":
                        expected.update(("clang-23", "clang-tidy-23", "libomp-23-dev"))
                    self.assertEqual(set(packages), expected)
                    versions = {call[0] for call in calls if call[1:] == ["--version"]}
                    expected = {"clang-format-23"}
                    if mode == "false":
                        expected.update(
                            ("clang++-23", "clang-tidy-23", "cmake", "ninja")
                        )
                    self.assertEqual(versions, expected)


if __name__ == "__main__":
    unittest.main()
