# SPDX-FileCopyrightText: Copyright DB InfraGO AG
# SPDX-License-Identifier: Apache-2.0


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import alphaghost

# -- Project information -----------------------------------------------------

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
with open("../../pyproject.toml", "rb") as f:
    _metadata = tomllib.load(f)["project"]

project = "alphaghost"
author = _metadata["authors"][0]["name"]
copyright = f"{author} and the {_metadata['name']} contributors"
license = _metadata["license"]["text"]
install_requirements = _metadata["dependencies"]
python_requirement = _metadata["requires-python"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_argparse_cli",
    "sphinx_autodoc_typehints",
]

nitpicky = True

nitpick_ignore_regex = [
    (r"py:.*", r"^pyspiel\..*"),
    (r"py:.*", r"^open_spiel\..*"),
    (r"py:.*", r"^drawsvg\..*"),
]

# -- General information about the project -----------------------------------

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The full version, including alpha/beta/rc tags.
version = alphaghost.__version__
rst_epilog = f"""
.. |Project| replace:: {project}
.. |Version| replace:: {version}
"""

# -- Options for auto-doc ----------------------------------------------------
autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_typehints = "description"
autodoc_mock_imports = ["pyspiel", "open_spiel", "torch"]


# -- Options for napoleon ----------------------------------------------------
napoleon_google_docstring = False
napoleon_include_init_with_doc = True

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = "never"

# -- Options for Intersphinx output ------------------------------------------
intersphinx_mapping = {
    "click": ("https://click.palletsprojects.com/en/latest", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation
# for a list of builtin themes.

html_theme = "sphinx_rtd_theme"

html_short_title = "alphaghost"
html_show_sourcelink = False
html_context = {
    "dependencies": install_requirements,
    "py_req": python_requirement,
}


# -- Skip __new__ methods ----------------------------------------------------
# This skips all __new__ methods. They only appear for NamedTuple
# classes, and they don't show any useful information that's not
# documented on the members already anyways.
def skip_dunder_new(app, what, name, obj, skip, options) -> bool:
    del app, obj, options
    return skip or (what == "class" and name == "__new__")


def setup(app):
    app.connect("autodoc-skip-member", skip_dunder_new)
