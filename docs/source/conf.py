"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options.
For a full list see the documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html.
"""
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import gpytorch.constraints
import gpytorch.distributions
import gpytorch.kernels
import gpytorch.likelihoods
import gpytorch.means
import gpytorch.mlls
import gpytorch.models
import gpytorch.module
import gpytorch.variational
import sphinx.config
import torch
import torch.optim
from PIL import Image
from sphinx_autodoc_typehints import format_annotation as default_format_annotation
from typing_extensions import Self, TypeAlias, Unpack

# -- Path setup --------------------------------------------------------------

CONF_FILE_PATH = __file__
SOURCE_FOLDER_FILE_PATH = os.path.abspath(os.path.join(CONF_FILE_PATH, ".."))
DOCS_FOLDER_FILE_PATH = os.path.abspath(os.path.join(SOURCE_FOLDER_FILE_PATH, ".."))
VANGUARD_FOLDER_FILE_PATH = os.path.abspath(os.path.join(DOCS_FOLDER_FILE_PATH, ".."))

sys.path.extend([DOCS_FOLDER_FILE_PATH, SOURCE_FOLDER_FILE_PATH, VANGUARD_FOLDER_FILE_PATH, ".."])

import vanguard
import vanguard.base.basecontroller
from vanguard.hierarchical.collection import ModuleT

# -- Project information -----------------------------------------------------

project = "Vanguard"
copyright = "UK Crown"
author = "GCHQ"
version = "v" + vanguard.__version__

# -- General configuration ---------------------------------------------------

show_warning_types = True
suppress_warnings = ["config.cache"]  # TODO: Remove this if/when Sphinx fix the caching issue

extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]

from refstyle import STYLE_NAME

bibtex_default_style = STYLE_NAME
bibtex_bibfiles = [os.path.join(VANGUARD_FOLDER_FILE_PATH, "references.bib")]

linkcheck_ignore = ["https://doi.org"]
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
    "gpytorch": ('https://docs.gpytorch.ai/en/v1.8.1/', None),  # TODO: Bump this when updating gpytorch
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python3": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ('https://pytorch.org/docs/stable/', None),

}

nitpicky = True
nitpicky_ignore_mapping: Dict[str, List[str]] = {
    "py_attr": [
        "device",
    ],
    "py:class": [
        "gpytorch.models.GP",
        "array_like",
        "function",
        "Any",
        "gpytorch.mlls._ApproximateMarginalLogLikelihood",
        "torch.Size",
        "gpytorch.distributions.multivariate_normal.MultivariateNormal",  # TODO: Remove when bumping gpytorch
        "gpytorch.likelihoods.likelihood.Likelihood",  # TODO: Remove when bumping gpytorch
    ],
    "py:meth": [
        "activate",
        "_tensor_prediction",
        "_tensor_confidence_interval",
    ],
    "py_mod": [
        "torch",
    ]
}
nitpick_ignore = [(type_, target) for type_, targets in nitpicky_ignore_mapping.items() for target in targets]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "examples/**/README.rst",
                    "examples/README.rst", "examples/index.rst"]

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


always_document_param_types = True
autodoc_custom_types: dict[TypeAlias, str] = {
    torch.optim.lr_scheduler.LRScheduler: ":mod:`LRScheduler <torch.optim.lr_scheduler>`",
    ModuleT: ":class:`~gpytorch.Module`",
    Self: ":data:`~typing.Self`",
    gpytorch.mlls.MarginalLogLikelihood: f":mod:`{gpytorch.mlls.MarginalLogLikelihood.__name__} <gpytorch.mlls>`",
    Union[vanguard.base.basecontroller.ttypes, vanguard.base.basecontroller.ttypes_cuda]: str(default_format_annotation(Type[torch.Tensor], sphinx.config.Config())),
}


# TODO: Remove these when gpytorch is sufficiently bumped:
autodoc_custom_types.update({
    gpytorch.means.Mean: ":class:`~gpytorch.means.Mean`",
    gpytorch.kernels.Kernel: ":class:`~gpytorch.kernels.Kernel`",
    gpytorch.likelihoods.Likelihood: ":class:`~gpytorch.likelihoods.Likelihood`",
    gpytorch.likelihoods.GaussianLikelihood: ":class:`~gpytorch.likelihoods.GaussianLikelihood`",
    gpytorch.distributions.Distribution: ":class:`~gpytorch.distributions.Distribution`",
    gpytorch.distributions.MultivariateNormal: ":class:`~gpytorch.distributions.MultivariateNormal`",
    gpytorch.distributions.MultitaskMultivariateNormal: ":class:`~gpytorch.distributions.MultitaskMultivariateNormal`",
    gpytorch.models.ExactGP: ":class:`~gpytorch.models.ExactGP`",
    gpytorch.module.Module: ":class:`~gpytorch.Module",
    gpytorch.constraints.Interval: ":class:`~gpytorch.constraints.Interval`",
    gpytorch.variational._VariationalStrategy: ":class:`~gpytorch.variational._VariationalStrategy`",
    gpytorch.variational._VariationalDistribution: ":class:`~gpytorch.variational._VariationalDistribution`",
})


def typehints_formatter(annotation: Any, config: sphinx.config.Config) -> Optional[str]:
    """
    Properly replace custom type aliases.

    :param annotation: The type annotation to be processed.
    :param config: The current configuration being used.
    :returns: A string of reStructured text (e.g. :class:`something`) or None to fall
        back to the default.

    This function is called on each type annotation that Sphinx processes.
    The following steps occur:

    1. Check whether a specific Sphinx string has been defined in autodoc_custom_types.
        If so, return that.
    2. Check if the annotation is a TypeVar. If so, replace it with its "bound" type
        for clarity in the docs. If not, then replace it with typing.Any.
    3. If not, then return None, which uses thee default formatter.

    See https://github.com/tox-dev/sphinx-autodoc-typehints?tab=readme-ov-file#options
    for specification.
    """
    try:
        return autodoc_custom_types[annotation]
    except KeyError:
        pass

    if isinstance(annotation, TypeVar):
        try:
            if annotation.__bound__ is None:  # when a generic TypeVar has been used.
                return str(default_format_annotation(Any, config))
        except AttributeError:
            if getattr(annotation, "__origin__", None) is Unpack:
                return ":data:`~typing.Unpack`"
            raise
        return str(default_format_annotation(annotation.__bound__, config))  # get the annotation for the bound type

    return None


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

confutils.copy_filtered_files(examples_source, examples_dest, file_types={".ipynb", ".rst"})

notebooks_file_paths = [os.path.join(examples_dest, notebook_path) for notebook_path in os.listdir(examples_dest)]
confutils.process_notebooks(notebooks_file_paths)

circle_logo_path = os.path.join(SOURCE_FOLDER_FILE_PATH, "_static", "logo_circle.png")
if not os.path.exists(circle_logo_path):
    logo_path = os.path.join(SOURCE_FOLDER_FILE_PATH, "_static", "logo.png")
    with open(logo_path, "rb") as rf:
        image = Image.open(rf)
        cropped = image.crop((0, 0, image.height, image.height))
        cropped.save(circle_logo_path)
