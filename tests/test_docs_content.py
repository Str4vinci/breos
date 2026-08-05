"""Guard the boundary between public user docs and repository documentation."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"

INTERNAL_DOCS = (
    "architecture/third-party-wrapping.md",
    "architecture/0.4x-refactor-plan.md",
    "architecture/string-inverter-sizing.md",
    "architecture/battery-degradation-policy.md",
    "architecture/blast-degradation-engine.md",
    "adr/0001-docs-architecture.md",
    "adr/index.md",
    "release.md",
)


def test_internal_project_notes_are_not_read_the_docs_sources():
    for relative_path in INTERNAL_DOCS:
        assert not (DOCS_ROOT / relative_path).exists(), relative_path

    assert (REPO_ROOT / "design" / "architecture" / "third-party-wrapping.md").is_file()
    assert (REPO_ROOT / "design" / "adr" / "0001-docs-architecture.md").is_file()
    assert (REPO_ROOT / "maintainers" / "release-checklist.md").is_file()


def test_current_user_guides_do_not_describe_the_release_as_0_3_x():
    user_facing_files = [
        DOCS_ROOT / "index.md",
        *sorted((DOCS_ROOT / "getting-started").glob("*.md")),
        *sorted((DOCS_ROOT / "api").glob("*.md")),
        *sorted((REPO_ROOT / "configs" / "examples").glob("*.toml")),
    ]

    stale = [str(path.relative_to(REPO_ROOT)) for path in user_facing_files if "0.3.x" in path.read_text()]
    assert stale == []


def test_installation_guide_does_not_pin_yesterdays_release():
    installation = (DOCS_ROOT / "getting-started" / "installation.md").read_text()
    assert re.search(r"github\.com/Str4vinci/breos\.git@v\d+\.\d+\.\d+", installation) is None
