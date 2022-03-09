# -*- coding: utf-8 -*-
#
# protopipe documentation build configuration file, created by
# sphinx-quickstart on Mon Nov 12 20:47:02 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from protopipe import __version__

# -- Project information -----------------------------------------------------

# General information about the project.
project = "protopipe"
copyright = "2020, Michele Peresano"
author = "Michele Peresano, Julien Lefaucheur"

# The version info for the project acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_issues",
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

# nbsphinx
nbsphinx_execute = "never"

# sphinx_automodapi: avoid having methods and attributes of classes being shown
# multiple times.
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

autoclass_content = "both"  # include both class docstring and __init__

# autodoc_default_options = {
#     # Make sure that any autodoc declarations show the right members
#     "members": True,
#     "inherited-members": True,
#     "private-members": True,
#     "show-inheritance": True,
# }

autosummary_generate = True  # Make _autosummary files and include them
napoleon_numpy_docstring = False  # Force consistency, leave only Google
napoleon_use_rtype = False  # More legible

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
# html_logo = "_static/CTA_logo.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "logo_link": "https://www.cta-observatory.org/wp-content/themes/ctao/assets/img/logo_red.png",
    "icon_links": [
        # {
        #     "name": "GitHub",
        #     "url": "https://github.com/cta-observatory/protopipe",
        #     "icon": "fab fa-github-square",
        # },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/protopipe/",
            "icon": "fab fa-python",
        },
    ],
    "github_url": "https://github.com/cta-observatory/protopipe",
    "icon_links_label": "Quick Links",
    "external_links": [
        {
            "name": "Performance",
            "url": "https://cta.cloud.xwiki.com/xwiki/wiki/aswg/view/Main/Benchmarks%20and%20Reference%20Analysis/Protopipe%20performance%20and%20pipeline%20comparisons/",
        },
    ],
    # "show_nav_level": 4,
    # "navigation_depth": 4,
    "use_edit_page_button": True,
    "search_bar_text": "Search the docs...",
    # "navbar_align": "left",
}

html_context = {
    "github_user": "https://github.com/cta-observatory",
    "github_repo": "https://github.com/cta-observatory/protopipe",
    "github_version": "master",
    "doc_path": "docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "protopipedoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "protopipe.tex",
        "protopipe Documentation",
        "Michele Peresano, Julien Lefaucheur",
        "manual",
    )
]

# -- Options for sphinx issues -----

# GitHub repo
issues_github_path = "cta-observatory/protopipe"

# equivalent to
issues_uri = "https://github.com/cta-observatory/protopipe/issues/{issue}"
issues_pr_uri = "https://github.com/cta-observatory/protopipe/pull/{pr}"
issues_commit_uri = "https://github.com/cta-observatory/protopipe/commit/{commit}"

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "protopipe", "protopipe Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "protopipe",
        "protopipe Documentation",
        author,
        "protopipe",
        "One line description of project.",
        "Miscellaneous",
    )
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"https://docs.python.org/": None}
