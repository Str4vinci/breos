"""Sphinx configuration for the BREOS documentation."""

from __future__ import annotations

import os
from importlib.metadata import version as pkg_version

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
]

# --- Source files -----------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
        "version_match": "latest",
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
