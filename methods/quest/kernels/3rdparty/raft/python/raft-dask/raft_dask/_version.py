#
#
#
#


import importlib.resources

__version__ = (
  importlib.resources.files("raft_dask")
  .joinpath("VERSION")
  .read_text()
  .strip()
)
__git_commit__ = ""
