"""Run the Python code blocks from docs/docs/quick_start.md in one shared namespace.

Extracts every ```python fence from the Markdown source and executes them in
order so that any API change that breaks the page also fails CI.
"""

import re
from pathlib import Path

QUICK_START = Path(__file__).resolve().parents[2] / "docs" / "docs" / "quick_start.md"
_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return _BLOCK_RE.findall(text)


def test_quick_start_blocks_run_in_order(monkeypatch, tmp_path):
    """All Python blocks in quick_start.md execute successfully in one namespace."""
    blocks = _extract_blocks(QUICK_START)
    assert len(blocks) == 3, (
        f"Expected 3 python blocks in quick_start.md, found {len(blocks)}. "
        "Update this test if the page gains or loses blocks."
    )

    # Run from a temp dir so a future block writing a relative path can't dirty the repo.
    monkeypatch.chdir(tmp_path)

    namespace: dict = {}
    for i, code in enumerate(blocks, start=1):
        exec(compile(code, f"quick_start.md block {i}", "exec"), namespace)
