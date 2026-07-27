from pybind11_tests import docstring_options as m


def test_docstring_options():
  assert not m.test_function1.__doc__

  assert m.test_function2.__doc__ == "A custom docstring"

  assert m.test_overloaded1.__doc__ == "Overload docstring"

  assert m.test_overloaded2.__doc__ == "overload docstring 1\noverload docstring 2"

  assert m.test_overloaded3.__doc__ == "Overload docstr"

  assert m.test_function3.__doc__.startswith("test_function3(a: int, b: int) -> None")

  assert m.test_function4.__doc__.startswith("test_function4(a: int, b: int) -> None")
  assert m.test_function4.__doc__.endswith("A custom docstring\n")

  assert not m.test_function5.__doc__

  assert m.test_function6.__doc__ == "A custom docstring"

  assert m.test_function7.__doc__.startswith("test_function7(a: int, b: int) -> None")
  assert m.test_function7.__doc__.endswith("A custom docstring\n")

  assert m.test_function8.__doc__ is None

  assert not m.DocstringTestFoo.__doc__
  assert not m.DocstringTestFoo.value_prop.__doc__

  assert (
    m.DocstringTestEnum1.__doc__
    == "Enum docstring\n\nMembers:\n\n Member1\n\n Member2"
  )

  assert (
    m.DocstringTestEnum2.__doc__
    == "Enum docstring\n\nMembers:\n\n Member1\n\n Member2"
  )

  assert m.DocstringTestEnum3.__doc__ == "Enum docstring"

  assert m.DocstringTestEnum4.__doc__ == "Members:\n\n Member1\n\n Member2"

  assert m.DocstringTestEnum5.__doc__ is None
