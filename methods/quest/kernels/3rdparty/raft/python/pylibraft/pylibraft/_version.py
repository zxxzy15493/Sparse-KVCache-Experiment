#
#
#
#


import importlib.resources

__version__ = (
  importlib.resources.files("pylibraft")
  .joinpath("VERSION")
  .read_text()
  .strip()
)
__git_commit__ = ""
