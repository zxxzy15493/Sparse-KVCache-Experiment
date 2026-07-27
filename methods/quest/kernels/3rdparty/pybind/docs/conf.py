#
#
#
#

import os
import re
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent.resolve()




extensions = [
  "breathe",
  "sphinx_copybutton",
  "sphinxcontrib.rsvgconverter",
  "sphinxcontrib.moderncmakedomain",
]

breathe_projects = {"pybind11": ".build/doxygenxml/"}
breathe_default_project = "pybind11"
breathe_domain_by_extension = {"h": "cpp"}

templates_path = [".templates"]

source_suffix = ".rst"


master_doc = "index"

project = "pybind11"
copyright = "2017, Wenzel Jakob"
author = "Wenzel Jakob"


with open("../pybind11/_version.py") as f:
  code = compile(f.read(), "../pybind11/_version.py", "exec")
loc = {}
exec(code, loc)

version = loc["__version__"]

#
language = None


exclude_patterns = [".build", "release.rst"]

default_role = "any"







todo_include_todos = False




html_theme = "furo"







html_static_path = ["_static"]

html_css_files = [
  "css/custom.css",
]

















htmlhelp_basename = "pybind11doc"


latex_engine = "pdflatex"

latex_elements = {
  #
  #
  "classoptions": ",openany,oneside",
  "preamble": r"""
\usepackage{fontawesome}
\usepackage{textgreek}
\DeclareUnicodeCharacter{00A0}{}
\DeclareUnicodeCharacter{2194}{\faArrowsH}
\DeclareUnicodeCharacter{1F382}{\faBirthdayCake}
\DeclareUnicodeCharacter{1F355}{\faAdjust}
\DeclareUnicodeCharacter{0301}{'}
\DeclareUnicodeCharacter{03C0}{\textpi}

""",
}

latex_documents = [
  (master_doc, "pybind11.tex", "pybind11 Documentation", "Wenzel Jakob", "manual"),
]









man_pages = [(master_doc, "pybind11", "pybind11 Documentation", [author], 1)]




texinfo_documents = [
  (
    master_doc,
    "pybind11",
    "pybind11 Documentation",
    author,
    "pybind11",
    "One line description of project.",
    "Miscellaneous",
  ),
]





primary_domain = "cpp"
highlight_language = "cpp"


def generate_doxygen_xml(app):
  build_dir = os.path.join(app.confdir, ".build")
  if not os.path.exists(build_dir):
    os.mkdir(build_dir)

  try:
    subprocess.call(["doxygen", "--version"])
    retcode = subprocess.call(["doxygen"], cwd=app.confdir)
    if retcode < 0:
      sys.stderr.write(f"doxygen error code: {-retcode}\n")
  except OSError as e:
    sys.stderr.write(f"doxygen execution failed: {e}\n")


def prepare(app):
  with open(DIR.parent / "README.rst") as f:
    contents = f.read()

  if app.builder.name == "latex":
    contents = contents[contents.find(r".. start") :]

    contents = re.sub(r"^(.*)\n[-~]{3,}$", r"**\1**", contents, flags=re.MULTILINE)

  with open(DIR / "readme.rst", "w") as f:
    f.write(contents)


def clean_up(app, exception): # noqa: ARG001
  (DIR / "readme.rst").unlink()


def setup(app):
  app.connect("builder-inited", generate_doxygen_xml)

  app.connect("builder-inited", prepare)

  app.connect("build-finished", clean_up)
