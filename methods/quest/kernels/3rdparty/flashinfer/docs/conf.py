import os
import sys

import tlcpack_sphinx_addon
#


project = 'FlashInfer'
author = "FlashInfer Contributors"
footer_copyright = '2023-2024, {}'.format(author)

version = "0.0.1"
release = "0.0.1"


extensions = [
  "sphinx_tabs.tabs",
  "sphinx_toolbox.collapse",
  "sphinxcontrib.httpdomain",
  "sphinx.ext.autodoc",
  "sphinx.ext.napoleon",
  "sphinx_reredirects",
]

source_suffix = [".rst"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

todo_include_todos = False


import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

templates_path = []

html_static_path = []

footer_note = " "

html_theme_options = {
  "logo_only": True,
}

header_links = [
  ("Home", "https://flashinfer.ai"),
  ("Github", "https://github.com/flashinfer-ai/flashinfer"),
  ("Discussions", "https://github.com/orgs/flashinfer-ai/discussions"),
]

html_context = {
  "footer_copyright": footer_copyright,
  "footer_note": footer_note,
  "header_links": header_links,
  "display_github": True,
  "github_user": "flashinfer-ai",
  "github_repo": "flashinfer",
  "github_version": "main/docs/",
  "theme_vcs_pageview_mode": "edit",
}

templates_path += [tlcpack_sphinx_addon.get_templates_path()]
html_static_path += [tlcpack_sphinx_addon.get_static_path()]

