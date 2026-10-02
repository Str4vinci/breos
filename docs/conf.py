"""Sphinx configuration for the BREOS documentation."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

# The gallery pages import their results loader from docs/_ext.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "BREOS"
author = "Leonardo Rodrigues"
copyright = "2026, Leonardo Rodrigues"
release = pkg_version("breos")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
]

# --- Source files -----------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples/**", "gallery/sg_execution_times.rst"]

# --- Autosummary / autodoc --------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# --- Intersphinx ------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pvlib": ("https://pvlib-python.readthedocs.io/en/stable/", None),
}
if os.environ.get("BREOS_DOCS_OFFLINE"):
    # Keep local/restricted release verification deterministic. External type
    # links are enriched in normal/Read the Docs builds where inventories are
    # reachable, but they are not required to validate BREOS's own sources.
    intersphinx_mapping = {}

# --- MyST -------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
    "substitution",
]
myst_heading_anchors = 3

# --- Example gallery --------------------------------------------------------
#
# Each page in docs/examples/ reports a stored run: it loads results that
# tools/regenerate_gallery_results.py wrote under docs/examples/_results/ and
# draws them. Nothing is simulated or fetched while the docs build.

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "gallery",
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"__init__\.py",
    "subsection_order": [
        "examples/getting_started",
        "examples/pv_design",
        "examples/battery",
        "examples/tariffs",
        "examples/uncertainty",
    ],
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "abort_on_example_error": True,
    "only_warn_on_example_error": False,
    "reset_modules": ("gallery_style.reset_style",),
    "image_scrapers": ("gallery_style.scrape",),
    "image_srcset": ["2x"],
    "capture_repr": ("_repr_html_",),
    "min_reported_time": 3600,
    "write_computation_times": False,
    "show_signature": False,
    "default_thumb_file": str(Path(__file__).parent / "_static" / "BREOS.png"),
    "matplotlib_animations": False,
}

# --- HTML output ------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "BREOS"

html_theme_options = {
    "logo": {
        "image_light": "_static/BREOS_black.svg",
        "image_dark": "_static/BREOS.png",
        "text": "BREOS",
    },
    "github_url": "https://github.com/Str4vinci/breos",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "switcher": {
        "json_url": "https://breos.readthedocs.io/en/latest/_static/switcher.json",
        # Read the Docs sets READTHEDOCS_VERSION to "stable" or "latest", so
        # the selector marks the version being read; local builds use latest.
        "version_match": os.environ.get("READTHEDOCS_VERSION", "latest"),
    },
    "check_switcher": False,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "Str4vinci",
    "github_repo": "breos",
    "github_version": "develop",
    "doc_path": "docs",
}

# --- Misc -------------------------------------------------------------------

# Silence warnings for references that can't be resolved against external
# inventories (e.g. internal type aliases that don't intersphinx).
nitpicky = False
