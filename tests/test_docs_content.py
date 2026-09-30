"""Guard the boundary between public user docs and repository documentation."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"

# Design notes and ADRs live in design/ and the release checklist in
# maintainers/. None of them is a docs/ source, so the places they used to
# occupy under docs/ must stay absent.
INTERNAL_DOC_LOCATIONS = ("architecture", "adr", "release.md")


def test_internal_project_notes_are_not_read_the_docs_sources():
    present = [name for name in INTERNAL_DOC_LOCATIONS if (DOCS_ROOT / name).exists()]
    assert present == []


def test_installation_guide_does_not_pin_yesterdays_release():
    installation = (DOCS_ROOT / "getting-started" / "installation.md").read_text()
    assert re.search(r"github\.com/Str4vinci/breos\.git@v\d+\.\d+\.\d+", installation) is None
