"""Configuration file for the Sphinx documentation builder.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

from importlib.metadata import version as _version

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

# List of patterns, relative to source directory, that match files and
# exclude_patterns; "_build", "Thumbs.db", ".DS_Store"
exclude_patterns: list[str] = []

# theme template overrides (e.g. version in the sidebar)
templates_path = ["_templates"]

# autodoc: types are rendered in the parameter descriptions, not the signature
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# myst-nb: render the outputs stored in the notebooks; never execute them
# (execution would install packages and download data over the network)
nb_execution_mode = "off"


# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
