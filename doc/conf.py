# pylint: skip-file

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

### Project information

project = "Tumult Core"
author = "Tumult Labs"
copyright = f"{datetime.date.today().year} Tumult Labs"
# Note that this is the name of the module provided by the package, not
# necessarily the name of the package as pip understands it.
package_name = "tmlt.core"


### Build information

ci_tag = os.getenv("CI_COMMIT_TAG")
ci_branch = os.getenv("CI_COMMIT_BRANCH")

# For non-prerelease tags, make the version "vX.Y" to match how we show it in
# the version switcher and the docs URLs. Sphinx's nomenclature around versions
# can be a bit confusing -- "version" means sort of the documentation version
# (for us, the minor release), while "release" is the full version number of the
# package on which the docs were built.
if ci_tag and "-" not in ci_tag:
    release = ci_tag
    version = "v" + ".".join(ci_tag.split(".")[:2])
else:
    release = version = ci_tag or ci_branch or "HEAD"

commit_hash = os.getenv("CI_COMMIT_SHORT_SHA") or "unknown version"
build_time = datetime.datetime.utcnow().isoformat(sep=" ", timespec="minutes")

### Sphinx configuration

extensions = [
    "autoapi.extension",
    "sphinxcontrib.images",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    # smart_resolver fixes cases where an object is documented under a name
    # different from its qualname, e.g. due to importing it in an __init__.
    "sphinx_automodapi.smart_resolver",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
]

bibtex_bibfiles = ["ref.bib"]

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
autoapi_dirs = ["../src/tmlt/"]
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
    # Set the package_name variable so it can be used in templates.
    jinja_env.globals["package_name"] = package_name
    # Define a new test for filtering out objects with @nodoc in their
    # docstring; this needs to be defined here because Jinja2 doesn't have a
    # built-in "contains" or "match" test.
    jinja_env.tests["nodoc"] = lambda obj: "@nodoc" in obj.docstring


# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Autodoc typehints settings
always_use_bars_union = True

# General settings
master_doc = "index"
exclude_patterns = ["templates"]
# Don't test stand-alone doctest blocks -- this prevents the examples from
# docstrings from being tested by Sphinx (nosetests --with-doctest already
# covers them).
doctest_test_doctest_blocks = ""

nitpick_ignore = [
    # Expr in __init__ is resolved fine but not in type hint
    ("py:class", "sympy.Expr"),
    # Type aliases are not correctly resolved by Sphinx
    ("py:class", "ExactNumberInput"),
    ("py:class", "PrivacyBudgetInput"),
    ("py:class", "PrivacyBudgetValue"),
    ("py:class", "tmlt.core.measures.PrivacyBudgetInput"),
    ("py:class", "tmlt.core.measures.PrivacyBudgetValue"),
    ("py:class", "SparkColumnsDescriptor"),
    ("py:class", "PandasColumnsDescriptor"),
    ("py:class", "tmlt.core.utils.exact_number.ExactNumberInput"),
    # Numpy dtypes
    ("py:class", "numpy.str_"),
    ("py:class", "numpy.int32"),
    ("py:class", "numpy.int64"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.float64"),
    # TypeVar support: https://github.com/agronholm/sphinx-autodoc-typehints/issues/39
    ("py:class", "Ellipsis"),
    ("py:class", "T"),
]

# Theme settings
templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "header_links_before_dropdown": 6,
    "collapse_navigation": True,
    "navigation_depth": 4,
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright", "build-info"],
    "footer_end": ["sphinx-version", "theme-version"],
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
html_sidebars = {"**": ["package-name", "version-switcher", "sidebar-nav-bs"]}

# Intersphinx mapping

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/1.18/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/1.2.0/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "pyspark": ("https://spark.apache.org/docs/3.5.1/api/python/", None),
}

# The ACM website seems to have some sort of protection that foils the linkchecker.
linkcheck_ignore = [
    r"https://doi.org/10.1145/2382196.2382264",
    r"https://gmplib.org/",
]


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
        "__init_subclass__",
    ]
    excluded_attributes = ["__slots__"]
    if what == "method" and name.split(".")[-1] in excluded_methods:
        return True
    if what == "attribute" and name.split(".")[-1] in excluded_attributes:
        return True
    if "@nodoc" in obj.docstring:
        return True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_members)
    # Write out the version and release numbers (using Sphinx's definitions of
    # them) for use by later automation.
    outdir = Path(sphinx.outdir)
    (outdir / "_version").write_text(version)
    (outdir / "_release").write_text(release)
