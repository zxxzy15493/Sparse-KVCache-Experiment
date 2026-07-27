
import os
import sys

sys.path.insert(0, os.path.abspath("sphinxext"))

from github_link import make_linkcode_resolve # noqa



#

extensions = [
  "numpydoc",
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "sphinx.ext.doctest",
  "sphinx.ext.intersphinx",
  "sphinx.ext.linkcode",
  "IPython.sphinxext.ipython_console_highlighting",
  "IPython.sphinxext.ipython_directive",
  "breathe",
  "recommonmark",
  "sphinx_markdown_tables",
  "sphinx_copybutton"
]

breathe_default_project = "RAFT"
breathe_projects = {
  "RAFT": "../../cpp/doxygen/_xml/",
}
ipython_mplbackend = "str"

templates_path = ["_templates"]


#
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

master_doc = "index"

project = "raft"
copyright = "2023, NVIDIA Corporation"
author = "NVIDIA Corporation"

#
version = '24.02'
release = '24.02.00'

#
language = "en"

exclude_patterns = []

pygments_style = "sphinx"

todo_include_todos = False


#

html_theme = "pydata_sphinx_theme"


#
html_theme_options = {
  "external_links": [],
  "icon_links": [],
  "github_url": "https://github.com/rapidsai/raft",
  "twitter_url": "https://twitter.com/rapidsai",
  "show_toc_level": 1,
  "navbar_align": "right",
}

html_static_path = ["_static"]

html_js_files = []


htmlhelp_basename = "raftdoc"


latex_elements = {
  #
  #
  #
  #
}

latex_documents = [
  (master_doc, "raft.tex", "RAFT Documentation", "NVIDIA Corporation", "manual"),
]


man_pages = [(master_doc, "raft", "RAFT Documentation", [author], 1)]


texinfo_documents = [
  (
    master_doc,
    "raft",
    "RAFT Documentation",
    author,
    "raft",
    "One line description of project.",
    "Miscellaneous",
  ),
]

intersphinx_mapping = {
  "python": ("https://docs.python.org/", None),
  "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False


def setup(app):
  app.add_css_file("references.css")
  app.add_css_file("https://docs.rapids.ai/assets/css/custom.css")
  app.add_js_file(
    "https://docs.rapids.ai/assets/js/custom.js", loading_method="defer"
  )


linkcode_resolve = make_linkcode_resolve(
  "pylibraft",
  "https://github.com/rapidsai/raft"
  "raft/blob/{revision}/python/pylibraft"
  "{package}/{path}#L{lineno}",
)

default_role = "py:obj"
