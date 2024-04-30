"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options.
For a full list see the documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html.
"""
import os
import shutil
import sys

from PIL import Image

# -- Path setup --------------------------------------------------------------

CONF_FILE_PATH = __file__
SOURCE_FOLDER_FILE_PATH = os.path.abspath(os.path.join(CONF_FILE_PATH, ".."))
DOCS_FOLDER_FILE_PATH = os.path.abspath(os.path.join(SOURCE_FOLDER_FILE_PATH, ".."))
VANGUARD_FOLDER_FILE_PATH = os.path.abspath(os.path.join(DOCS_FOLDER_FILE_PATH, ".."))

sys.path.extend([DOCS_FOLDER_FILE_PATH, SOURCE_FOLDER_FILE_PATH, VANGUARD_FOLDER_FILE_PATH, ".."])

import vanguard

# -- Project information -----------------------------------------------------

project = "Vanguard"
copyright = "UK Crown"
author = "GCHQ"
version = "v" + vanguard.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.bibtex",
]

from refstyle import STYLE_NAME

bibtex_default_style = STYLE_NAME
bibtex_bibfiles = [os.path.join(VANGUARD_FOLDER_FILE_PATH, "references.bib")]

linkcheck_ignore = ["https://www.nationalarchives.gov.uk/", "https://doi.org"]
linkcheck_timeout = 5

coverage_show_missing_items = True
coverage_write_headline = False
coverage_ignore_classes = ["EmptyDataset"]
coverage_ignore_pyobjects = ["vanguard\.warps\.warpfunctions\..*?WarpFunction\.(deriv|forward|inverse)"]

nbsphinx_execute = "never"
nbsphinx_thumbnails = {"examples/*": "_static/logo_circle.png"}


# matplotlib.sphinxext.plot_directive parameters
plot_include_source = False
plot_html_show_formats = False
plot_html_show_source_link = False
plot_rcparams = {
    "figure.facecolor": (0, 0, 0, 0),  # transparent
    "axes.facecolor": (0, 0, 0, 0),
    "legend.framealpha": 0,  # transparent
}

autodoc_mock_imports = ["pandas", "sklearn_extra"]

intersphinx_mapping = {
    "gpytorch": ("https://docs.gpytorch.ai/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python3": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

nitpicky = True
nitpick_ignore = [
    ("py:class", "array_like"),
    ("py:attr", "device"),
    ("py:meth", "_tensor_prediction"),
    ("py:meth", "_tensor_confidence_interval"),
    ("py:mod", "torch"),
    ("py:class", "function"),
    ("py:class", "Any"),
    ("py:class", "gpytorch.mlls._ApproximateMarginalLogLikelihood"),
    ("py:meth", "activate"),
    ("py:class", "gpytorch.models.GP"),
    ("py:class", "torch.Size"),
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "examples/**/README.rst",
    "examples/README.rst",
    "examples/index.rst",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
html_context = {
    "examples_directory": "examples/notebooks/",
}

html_logo = "_static/logo_circle.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["extra.css", "gallery.css"]


def skip(app, what, name, obj, would_skip, options):
    """Ensure that __init__ files are NOT skipped."""
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    """Ensure that our new skip function is called."""
    app.connect("autodoc-skip-member", skip)


# -- FILE PRE-PROCESSING -----------------------------------------------------

import confutils

examples_source = os.path.join(VANGUARD_FOLDER_FILE_PATH, "examples", "notebooks")
examples_dest = os.path.join(SOURCE_FOLDER_FILE_PATH, "examples")

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)

confutils.copy_filtered_files(examples_source, examples_dest, file_types={".ipynb", ".md", ".rst"})

licence_file_path = os.path.join(VANGUARD_FOLDER_FILE_PATH, "LICENSE.md")
copyright_file_path = os.path.join(SOURCE_FOLDER_FILE_PATH, "copyright.rst")

confutils.create_copyright_file(licence_file_path, copyright_file_path)

notebooks_file_paths = [os.path.join(examples_dest, notebook_path) for notebook_path in os.listdir(examples_dest)]
confutils.process_notebooks(notebooks_file_paths)

circle_logo_path = os.path.join(SOURCE_FOLDER_FILE_PATH, "_static", "logo_circle.png")
if not os.path.exists(circle_logo_path):
    logo_path = os.path.join(SOURCE_FOLDER_FILE_PATH, "_static", "logo.png")
    with open(logo_path, "rb") as rf:
        image = Image.open(rf)
        cropped = image.crop((0, 0, image.height, image.height))
        cropped.save(circle_logo_path)
