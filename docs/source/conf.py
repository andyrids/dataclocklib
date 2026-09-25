"""Configuration file for the Sphinx documentation builder.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

# Path setup
# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make
# it absolute, like shown here.
import pathlib
import sys
from importlib.metadata import version as _version

docs_dir = pathlib.Path(__file__).parent.parent
package_dir = docs_dir.parent / "src" / "dataclocklib"

sys.path.insert(0, package_dir.as_posix())

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dataclocklib"
project_copyright = "2024-2026, Andrew Ridyard"
author = "Andrew Ridyard"

# The full version, including alpha/beta/rc tags
__version__ = _version("dataclocklib")

version = __version__
release = __version__

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # include documentation from docstrings
    "sphinx.ext.napoleon",  # support for Google & NumPy docstrings
    "sphinx.ext.githubpages",  # create .nojekyll file for GitHub Pages
    "sphinx.ext.viewcode",  # add links to highlighted source code
    "sphinx_rtd_theme",  # enable sphinx read the docs theme
    "myst_nb",  # support Jupyter notebooks as source files
    # "sphinx.ext.inheritance_diagram",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# exclude_patterns; "_build", "Thumbs.db", ".DS_Store"
exclude_patterns = []


# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {"version_selector": True}
html_static_path = ["_static"]
