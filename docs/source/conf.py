"""Configuration file for the Sphinx documentation builder.

Author: Andrew Ridyard.

License: GNU General Public License v3 or later.

Copyright (C): 2024.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import pathlib
import sys
from importlib.metadata import version as _version

docs_dir = pathlib.Path(__file__).parent.parent
package_dir = docs_dir.parent / "src" / "dataclocklib"

sys.path.insert(0, package_dir.as_posix())

print(package_dir)
print(_version("dataclocklib"))

# -- Project information ----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dataclocklib"
copyright = "2025, Andrew Ridyard"
author = "Andrew Ridyard"

# The full version, including alpha/beta/rc tags
__version__ = "0.0.1"

# -- General configuration ----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",     # include documentation from docstrings
    "sphinx.ext.napoleon",    # support for Google & NumPy docstrings
    "sphinx.ext.githubpages", # create .nojekyll file for GitHub Pages
    "sphinx.ext.viewcode",    # add links to highlighted source code
    #"sphinx_gallery.gen_gallery",
    #"sphinx.ext.inheritance_diagram",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
exclude_patterns = []


# -- Options for HTML output ----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme" # sphinx_rtd_theme
html_static_path = ["_static"]
