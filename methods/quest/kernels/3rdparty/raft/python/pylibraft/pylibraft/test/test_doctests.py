#
#
#
#

import contextlib
import doctest
import inspect
import io

import pytest

import pylibraft.cluster
import pylibraft.distance
import pylibraft.matrix
import pylibraft.neighbors
import pylibraft.random



def _name_in_all(parent, name):
  return name in getattr(parent, "__all__", [])


def _is_public_name(parent, name):
  return not name.startswith("_")


def _find_doctests_in_obj(obj, finder=None, criteria=None):
  """Find all doctests in an object.

  Parameters
  ----------
  obj : module or class
    The object to search for docstring examples.
  finder : doctest.DocTestFinder, optional
    The DocTestFinder object to use. If not provided, a DocTestFinder is
    constructed.
  criteria : callable, optional
    Callable indicating whether to recurse over members of the provided
    object. If not provided, names not defined in the object's ``__all__``
    property are ignored.

  Yields
  ------
  doctest.DocTest
    The next doctest found in the object.
  """
  if finder is None:
    finder = doctest.DocTestFinder()
  if criteria is None:
    criteria = _name_in_all
  for docstring in finder.find(obj):
    if docstring.examples:
      yield docstring
  for name, member in inspect.getmembers(obj):
    if not criteria(obj, name):
      continue
    if inspect.ismodule(member):
      yield from _find_doctests_in_obj(
        member, finder, criteria=_name_in_all
      )
    if inspect.isclass(member):
      yield from _find_doctests_in_obj(
        member, finder, criteria=_is_public_name
      )

    if callable(member) and not inspect.isfunction(member):
      for docstring in finder.find(member):
        if docstring.examples:
          yield docstring


DOC_STRINGS = list(_find_doctests_in_obj(pylibraft.cluster))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.common))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.distance))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.matrix.select_k))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.neighbors))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.neighbors.brute_force))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.neighbors.cagra))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.neighbors.ivf_flat))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.neighbors.ivf_pq))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.neighbors.refine))
DOC_STRINGS.extend(_find_doctests_in_obj(pylibraft.random))


@pytest.mark.parametrize(
  "docstring",
  DOC_STRINGS,
  ids=lambda docstring: docstring.name,
)
def test_docstring(docstring):
  optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
  runner = doctest.DocTestRunner(optionflags=optionflags)

  doctest_stdout = io.StringIO()
  with contextlib.redirect_stdout(doctest_stdout):
    runner.run(docstring)
    results = runner.summarize()
  assert not results.failed, (
    f"{results.failed} of {results.attempted} doctests failed for "
    f"{docstring.name}:\n{doctest_stdout.getvalue()}"
  )
