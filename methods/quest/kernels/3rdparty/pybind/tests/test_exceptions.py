import sys

import pytest

import env
import pybind11_cross_module_tests as cm
import pybind11_tests # noqa: F401
from pybind11_tests import exceptions as m


def test_std_exception(msg):
  with pytest.raises(RuntimeError) as excinfo:
    m.throw_std_exception()
  assert msg(excinfo.value) == "This exception was intentionally thrown."


def test_error_already_set(msg):
  with pytest.raises(RuntimeError) as excinfo:
    m.throw_already_set(False)
  assert (
    msg(excinfo.value)
    == "Internal error: pybind11::error_already_set called while Python error indicator not set."
  )

  with pytest.raises(ValueError) as excinfo:
    m.throw_already_set(True)
  assert msg(excinfo.value) == "foo"


def test_raise_from(msg):
  with pytest.raises(ValueError) as excinfo:
    m.raise_from()
  assert msg(excinfo.value) == "outer"
  assert msg(excinfo.value.__cause__) == "inner"


def test_raise_from_already_set(msg):
  with pytest.raises(ValueError) as excinfo:
    m.raise_from_already_set()
  assert msg(excinfo.value) == "outer"
  assert msg(excinfo.value.__cause__) == "inner"


def test_cross_module_exceptions(msg):
  with pytest.raises(RuntimeError) as excinfo:
    cm.raise_runtime_error()
  assert str(excinfo.value) == "My runtime error"

  with pytest.raises(ValueError) as excinfo:
    cm.raise_value_error()
  assert str(excinfo.value) == "My value error"

  with pytest.raises(ValueError) as excinfo:
    cm.throw_pybind_value_error()
  assert str(excinfo.value) == "pybind11 value error"

  with pytest.raises(TypeError) as excinfo:
    cm.throw_pybind_type_error()
  assert str(excinfo.value) == "pybind11 type error"

  with pytest.raises(StopIteration) as excinfo:
    cm.throw_stop_iteration()

  with pytest.raises(cm.LocalSimpleException) as excinfo:
    cm.throw_local_simple_error()
  assert msg(excinfo.value) == "external mod"

  with pytest.raises(KeyError) as excinfo:
    cm.throw_local_error()
  assert str(excinfo.value) == "'just local'"


# TODO: FIXME
@pytest.mark.xfail(
  "env.MACOS and (env.PYPY or pybind11_tests.compiler_info.startswith('Homebrew Clang'))",
  raises=RuntimeError,
  reason="See Issue #2847, PR #2999, PR #4324",
)
def test_cross_module_exception_translator():
  with pytest.raises(KeyError):
    m.throw_should_be_translated_to_key_error()


def test_python_call_in_catch():
  d = {}
  assert m.python_call_in_destructor(d) is True
  assert d["good"] is True


def ignore_pytest_unraisable_warning(f):
  unraisable = "PytestUnraisableExceptionWarning"
  if hasattr(pytest, unraisable): # Python >= 3.8 and pytest >= 6
    dec = pytest.mark.filterwarnings(f"ignore::pytest.{unraisable}")
    return dec(f)
  return f


@pytest.mark.xfail(env.PYPY, reason="Failure on PyPy 3.8 (7.3.7)", strict=False)
@ignore_pytest_unraisable_warning
def test_python_alreadyset_in_destructor(monkeypatch, capsys):
  hooked = False
  triggered = False

  if hasattr(sys, "unraisablehook"): # Python 3.8+
    hooked = True
    default_hook = sys.__unraisablehook__

    def hook(unraisable_hook_args):
      exc_type, exc_value, exc_tb, err_msg, obj = unraisable_hook_args
      if obj == "already_set demo":
        nonlocal triggered
        triggered = True
      default_hook(unraisable_hook_args)
      return

    monkeypatch.setattr(sys, "unraisablehook", hook)

  assert m.python_alreadyset_in_destructor("already_set demo") is True
  if hooked:
    assert triggered is True

  _, captured_stderr = capsys.readouterr()
  assert captured_stderr.startswith("Exception ignored in: 'already_set demo'")
  assert captured_stderr.rstrip().endswith("KeyError: 'bar'")


def test_exception_matches():
  assert m.exception_matches()
  assert m.exception_matches_base()
  assert m.modulenotfound_exception_matches_base()


def test_custom(msg):
  with pytest.raises(m.MyException) as excinfo:
    m.throws1()
  assert msg(excinfo.value) == "this error should go to py::exception<MyException>"

  with pytest.raises(m.MyExceptionUseDeprecatedOperatorCall) as excinfo:
    m.throws1d()
  assert (
    msg(excinfo.value)
    == "this error should go to py::exception<MyExceptionUseDeprecatedOperatorCall>"
  )

  with pytest.raises(RuntimeError) as excinfo:
    m.throws2()
  assert msg(excinfo.value) == "this error should go to a standard Python exception"

  with pytest.raises(RuntimeError) as excinfo:
    m.throws3()
  assert msg(excinfo.value) == "Caught an unknown exception!"

  with pytest.raises(m.MyException) as excinfo:
    m.throws4()
  assert msg(excinfo.value) == "this error is rethrown"

  with pytest.raises(RuntimeError) as excinfo:
    m.throws_logic_error()
  assert (
    msg(excinfo.value) == "this error should fall through to the standard handler"
  )

  with pytest.raises(OverflowError) as excinfo:
    m.throws_overflow_error()

  with pytest.raises(m.MyException5) as excinfo:
    m.throws5()
  assert msg(excinfo.value) == "this is a helper-defined translated exception"

  with pytest.raises(m.MyException5) as excinfo:
    m.throws5_1()
  assert msg(excinfo.value) == "MyException5 subclass"
  assert isinstance(excinfo.value, m.MyException5_1)

  with pytest.raises(m.MyException5_1) as excinfo:
    m.throws5_1()
  assert msg(excinfo.value) == "MyException5 subclass"

  with pytest.raises(m.MyException5) as excinfo: # noqa: PT012
    try:
      m.throws5()
    except m.MyException5_1 as err:
      raise RuntimeError("Exception error: caught child from parent") from err
  assert msg(excinfo.value) == "this is a helper-defined translated exception"


def test_nested_throws(capture):
  """Tests nested (e.g. C++ -> Python -> C++) exception handling"""

  def throw_myex():
    raise m.MyException("nested error")

  def throw_myex5():
    raise m.MyException5("nested error 5")


  with capture:
    m.try_catch(m.MyException5, throw_myex5)
  assert str(capture).startswith("MyException5: nested error 5")

  with pytest.raises(m.MyException) as excinfo:
    m.try_catch(m.MyException5, throw_myex)
  assert str(excinfo.value) == "nested error"

  def pycatch(exctype, f, *args): # noqa: ARG001
    try:
      f(*args)
    except m.MyException as e:
      print(e)

  with capture:
    m.try_catch(
      m.MyException5,
      pycatch,
      m.MyException,
      m.try_catch,
      m.MyException,
      throw_myex5,
    )
  assert str(capture).startswith("MyException5: nested error 5")

  with capture:
    m.try_catch(m.MyException, pycatch, m.MyException5, m.throws4)
  assert capture == "this error is rethrown"

  with pytest.raises(m.MyException5) as excinfo:
    m.try_catch(m.MyException, pycatch, m.MyException, m.throws5)
  assert str(excinfo.value) == "this is a helper-defined translated exception"


def test_throw_nested_exception():
  with pytest.raises(RuntimeError) as excinfo:
    m.throw_nested_exception()
  assert str(excinfo.value) == "Outer Exception"
  assert str(excinfo.value.__cause__) == "Inner Exception"


def test_invalid_repr():
  class MyRepr:
    def __repr__(self):
      raise AttributeError("Example error")

  with pytest.raises(TypeError):
    m.simple_bool_passthrough(MyRepr())


def test_local_translator(msg):
  """Tests that a local translator works and that the local translator from
  the cross module is not applied"""
  with pytest.raises(RuntimeError) as excinfo:
    m.throws6()
  assert msg(excinfo.value) == "MyException6 only handled in this module"

  with pytest.raises(RuntimeError) as excinfo:
    m.throws_local_error()
  assert not isinstance(excinfo.value, KeyError)
  assert msg(excinfo.value) == "never caught"

  with pytest.raises(Exception) as excinfo:
    m.throws_local_simple_error()
  assert not isinstance(excinfo.value, cm.LocalSimpleException)
  assert msg(excinfo.value) == "this mod"


def test_error_already_set_message_with_unicode_surrogate(): # Issue #4288
  assert m.error_already_set_what(RuntimeError, "\ud927") == (
    "RuntimeError: \\ud927",
    False,
  )


def test_error_already_set_message_with_malformed_utf8():
  assert m.error_already_set_what(RuntimeError, b"\x80") == (
    "RuntimeError: b'\\x80'",
    False,
  )


class FlakyException(Exception):
  def __init__(self, failure_point):
    if failure_point == "failure_point_init":
      raise ValueError("triggered_failure_point_init")
    self.failure_point = failure_point

  def __str__(self):
    if self.failure_point == "failure_point_str":
      raise ValueError("triggered_failure_point_str")
    return "FlakyException.__str__"


@pytest.mark.parametrize(
  ("exc_type", "exc_value", "expected_what"),
  [
    (ValueError, "plain_str", "ValueError: plain_str"),
    (ValueError, ("tuple_elem",), "ValueError: tuple_elem"),
    (FlakyException, ("happy",), "FlakyException: FlakyException.__str__"),
  ],
)
def test_error_already_set_what_with_happy_exceptions(
  exc_type, exc_value, expected_what
):
  what, py_err_set_after_what = m.error_already_set_what(exc_type, exc_value)
  assert not py_err_set_after_what
  assert what == expected_what


def _test_flaky_exception_failure_point_init_before_py_3_12():
  with pytest.raises(RuntimeError) as excinfo:
    m.error_already_set_what(FlakyException, ("failure_point_init",))
  lines = str(excinfo.value).splitlines()
  assert lines[:3] == [
    "pybind11::error_already_set: MISMATCH of original and normalized active exception types:"
    " ORIGINAL FlakyException REPLACED BY ValueError: triggered_failure_point_init",
    "",
    "At:",
  ]
  assert "test_exceptions.py(" in lines[3]
  assert lines[3].endswith("): __init__")
  assert lines[4].endswith(
    "): _test_flaky_exception_failure_point_init_before_py_3_12"
  )


def _test_flaky_exception_failure_point_init_py_3_12():
  what, py_err_set_after_what = m.error_already_set_what(
    FlakyException, ("failure_point_init",)
  )
  assert not py_err_set_after_what
  lines = what.splitlines()
  assert lines[0].endswith("ValueError[WITH __notes__]: triggered_failure_point_init")
  assert lines[1] == "__notes__ (len=1):"
  assert "Normalization failed:" in lines[2]
  assert "FlakyException" in lines[2]


@pytest.mark.skipif(
  "env.PYPY and sys.version_info[:2] < (3, 12)",
  reason="PyErr_NormalizeException Segmentation fault",
)
def test_flaky_exception_failure_point_init():
  if sys.version_info[:2] < (3, 12):
    _test_flaky_exception_failure_point_init_before_py_3_12()
  else:
    _test_flaky_exception_failure_point_init_py_3_12()


def test_flaky_exception_failure_point_str():
  what, py_err_set_after_what = m.error_already_set_what(
    FlakyException, ("failure_point_str",)
  )
  assert not py_err_set_after_what
  lines = what.splitlines()
  n = 3 if env.PYPY and len(lines) == 3 else 5
  assert (
    lines[:n]
    == [
      "FlakyException: <MESSAGE UNAVAILABLE DUE TO ANOTHER EXCEPTION>",
      "",
      "MESSAGE UNAVAILABLE DUE TO EXCEPTION: ValueError: triggered_failure_point_str",
      "",
      "At:",
    ][:n]
  )


def test_cross_module_interleaved_error_already_set():
  with pytest.raises(RuntimeError) as excinfo:
    m.test_cross_module_interleaved_error_already_set()
  assert str(excinfo.value) in (
    "2nd error.", # Almost all platforms.
    "RuntimeError: 2nd error.", # Some PyPy builds (seen under macOS).
  )


def test_error_already_set_double_restore():
  m.test_error_already_set_double_restore(True) # dry_run
  with pytest.raises(RuntimeError) as excinfo:
    m.test_error_already_set_double_restore(False)
  assert str(excinfo.value) == (
    "Internal error: pybind11::detail::error_fetch_and_normalize::restore()"
    " called a second time. ORIGINAL ERROR: ValueError: Random error."
  )


def test_pypy_oserror_normalization():
  what = m.test_pypy_oserror_normalization()
  assert "this_filename_must_not_exist" in what


def test_fn_cast_int_exception():
  with pytest.raises(RuntimeError) as excinfo:
    m.test_fn_cast_int(lambda: None)

  assert str(excinfo.value).startswith(
    "Unable to cast Python instance of type <class 'NoneType'> to C++ type"
  )
