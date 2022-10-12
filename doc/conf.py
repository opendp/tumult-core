# pylint: skip-file

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import logging
import os
import sys
import datetime

_logger = logging.getLogger(__name__)

# Project information

project = "Tumult Core"
author = "Tumult Labs"
copyright = "Tumult Labs 2022"

package_name = "tmlt.core"


# Build information

ci_tag = os.getenv("CI_COMMIT_TAG")
ci_branch = os.getenv("CI_COMMIT_BRANCH")

version = ci_tag or ci_branch or "HEAD"
commit_hash = os.getenv("CI_COMMIT_SHORT_SHA") or "unknown version"
build_time = datetime.datetime.utcnow().isoformat(sep=" ", timespec="minutes")

linkcheck_mode_url_prefix = os.getenv("BASE_URL_OVERRIDE")
# Linkcheck fails to check anchors in Github
# See https://github.com/sphinx-doc/sphinx/issues/9016 and also
# https://sphinx-doc.org/en/master/usage/configuration.html

# Sphinx configuration

extensions = [
    "autoapi.extension",
    "scanpydoc.elegant_typehints",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_panels",
]

# Prevent sphinx_panels from loading bootstrap a second time
panels_add_bootstrap_css = False
# Change colors & contrast to inactive tab labels so they pass WCAG AA; all
# other colors are the same as the defaults:
#   https://sphinx-panels.readthedocs.io/en/latest/#tabbed-content
panels_css_variables = {
    "tabs-color-label-active": "hsla(231, 99%, 66%, 1)",
    "tabs-color-label-inactive": "rgba(135, 138, 150, 1)",
    "tabs-color-overline": "rgb(207, 236, 238)",
    "tabs-color-underline": "rgb(207, 236, 238)",
    "tabs-size-label": "1rem",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autoapi settings
autoapi_root = "reference"
autoapi_dirs = ["../tmlt/"]
autoapi_keep_files = False
autoapi_template_dir = "../doc/templates"
autoapi_add_toctree_entry = False
autoapi_python_use_implicit_namespaces = True  # This is important for intersphinx
autoapi_options = [
    "members",
    "show-inheritance",
    "special-members",
    "show-module-summary",
    "imported-members",
    "inherited-members",
]
add_module_names = False


def autoapi_prepare_jinja_env(jinja_env):
    jinja_env.globals["package_name"] = package_name


# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# General settings
master_doc = "index"
exclude_patterns = ["templates"]
# Don't test stand-alone doctest blocks -- this prevents the examples from
# docstrings from being tested by Sphinx (nosetests --with-doctest already
# covers them).
doctest_test_doctest_blocks = ""

# scanpydoc overrides to resolve target
qualname_overrides = {
    "sympy.Expr": "sympy.core.expr.Expr",
    "pyspark.sql.types.Row": "pyspark.sql.Row",
    "pyspark.sql.dataframe.DataFrame": "pyspark.sql.DataFrame",
    "numpy.random._generator.Generator": "numpy.random.Generator",
}

nitpick_ignore = [
    # Expr in __init__ is resolved fine but not in type hint
    ("py:class", "sympy.Expr"),
    ("py:class", "ndarray}"),
    # Type Alias not resolved in type hint
    ("py:class", "ExactNumberInput"),
    # Unable to resolve Base classes
    ("py:class", "Transformation"),
    ("py:class", "ClipType"),
    ("py:class", "Row"),
    ("py:class", "SparkColumnsDescriptor"),
    ("py:class", "Aggregation"),
    ("py:class", "tmlt.core.utils.exact_number.ExactNumberInput"),
    # Numpy dtypes
    ("py:class", "numpy.str_"),
    ("py:class", "numpy.int32"),
    ("py:class", "numpy.int64"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.float64"),
    # Caused by a function in tmlt.core.utils.configuration returning a SparkConf
    ("py:class", "pyspark.conf.SparkConf"),
    # Caused by a function in tmlt.core.utils.configuration taking a RuntimeConf
    # as an argument
    ("py:class", "pyspark.sql.conf.RuntimeConfig"),
    # Caused by pyspark.sql.dataframe.DataFrame in a dataclass (in spark_domains)
    ("py:class", "pyspark.sql.dataframe.DataFrame"),
    # TypeVar support: https://github.com/agronholm/sphinx-autodoc-typehints/issues/39
    ("py:class", "Ellipsis"),
    ("py:class", "T"),
]
nitpick_ignore_regex = [
    # No intersphinx_mapping for typing_extensions
    (r"py:.*", r"typing_extensions.*")
]

# Theme settings
templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
    "navbar_end": ["navbar-icon-links"],
    "footer_items": ["copyright", "build-info", "sphinx-version"],
    "switcher": {
        "json_url": "https://docs.tmlt.dev/core/versions.json",
        "version_match": version,
    },
    "gitlab_url": "https://gitlab.com/tumult-labs/core",
}
html_context = {
    "default_mode": "light",
    "commit_hash": commit_hash,
    "build_time": build_time,
}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/version-banner.js"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = False
html_sidebars = {
    "**": ["package-name", "version-switcher", "search-field", "sidebar-nav-bs"]
}

# Intersphinx mapping

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/1.18/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/1.2.0/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "pyspark": ("https://spark.apache.org/docs/3.0.0/api/python/", None),
}


def skip_members(app, what, name, obj, skip, options):
    """Skip some members."""
    excluded_methods = [
        "__dir__",
        "__format__",
        "__hash__",
        "__post_init__",
        "__reduce__",
        "__reduce_ex__",
        "__repr__",
        "__setattr__",
        "__sizeof__",
        "__str__",
        "__subclasshook__",
    ]
    excluded_attributes = ["__slots__"]
    if what == "method" and name.split(".")[-1] in excluded_methods:
        return True
    if what == "attribute" and name.split(".")[-1] in excluded_attributes:
        return True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_members)
