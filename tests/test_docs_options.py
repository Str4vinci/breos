"""The generated packaged-options page must match the packaged data."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_options_page_matches_packaged_data():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "generate_option_docs.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
