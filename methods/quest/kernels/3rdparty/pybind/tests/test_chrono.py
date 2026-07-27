import datetime

import pytest

import env # noqa: F401
from pybind11_tests import chrono as m


def test_chrono_system_clock():
  date0 = datetime.datetime.today()
  date1 = m.test_chrono1()
  date2 = datetime.datetime.today()

  assert isinstance(date1, datetime.datetime)

  diff_python = abs(date2 - date0)
  diff = abs(date1 - date2)

  assert diff.days == 0

  assert diff.seconds <= diff_python.seconds


def test_chrono_system_clock_roundtrip():
  date1 = datetime.datetime.today()

  date2 = m.test_chrono2(date1)

  assert isinstance(date2, datetime.datetime)

  diff = abs(date1 - date2)
  assert diff == datetime.timedelta(0)


def test_chrono_system_clock_roundtrip_date():
  date1 = datetime.date.today()

  datetime2 = m.test_chrono2(date1)
  date2 = datetime2.date()
  time2 = datetime2.time()

  assert isinstance(datetime2, datetime.datetime)
  assert isinstance(date2, datetime.date)
  assert isinstance(time2, datetime.time)

  diff = abs(date1 - date2)
  assert diff.days == 0
  assert diff.seconds == 0
  assert diff.microseconds == 0

  assert date1 == date2

  assert time2.hour == 0
  assert time2.minute == 0
  assert time2.second == 0
  assert time2.microsecond == 0


SKIP_TZ_ENV_ON_WIN = pytest.mark.skipif(
  "env.WIN", reason="TZ environment variable only supported on POSIX"
)


@pytest.mark.parametrize(
  "time1",
  [
    datetime.datetime.today().time(),
    datetime.time(0, 0, 0),
    datetime.time(0, 0, 0, 1),
    datetime.time(0, 28, 45, 109827),
    datetime.time(0, 59, 59, 999999),
    datetime.time(1, 0, 0),
    datetime.time(5, 59, 59, 0),
    datetime.time(5, 59, 59, 1),
  ],
)
@pytest.mark.parametrize(
  "tz",
  [
    None,
    pytest.param("Europe/Brussels", marks=SKIP_TZ_ENV_ON_WIN),
    pytest.param("Asia/Pyongyang", marks=SKIP_TZ_ENV_ON_WIN),
    pytest.param("America/New_York", marks=SKIP_TZ_ENV_ON_WIN),
  ],
)
def test_chrono_system_clock_roundtrip_time(time1, tz, monkeypatch):
  if tz is not None:
    monkeypatch.setenv("TZ", f"/usr/share/zoneinfo/{tz}")

  datetime2 = m.test_chrono2(time1)
  date2 = datetime2.date()
  time2 = datetime2.time()

  assert isinstance(datetime2, datetime.datetime)
  assert isinstance(date2, datetime.date)
  assert isinstance(time2, datetime.time)

  assert time1 == time2

  assert date2.year == 1970
  assert date2.month == 1
  assert date2.day == 1


def test_chrono_duration_roundtrip():
  date1 = datetime.datetime.today()
  date2 = datetime.datetime.today()
  diff = date2 - date1

  assert isinstance(diff, datetime.timedelta)

  cpp_diff = m.test_chrono3(diff)

  assert cpp_diff == diff

  diff = datetime.timedelta(microseconds=-1)
  cpp_diff = m.test_chrono3(diff)

  assert cpp_diff == diff


def test_chrono_duration_subtraction_equivalence():
  date1 = datetime.datetime.today()
  date2 = datetime.datetime.today()

  diff = date2 - date1
  cpp_diff = m.test_chrono4(date2, date1)

  assert cpp_diff == diff


def test_chrono_duration_subtraction_equivalence_date():
  date1 = datetime.date.today()
  date2 = datetime.date.today()

  diff = date2 - date1
  cpp_diff = m.test_chrono4(date2, date1)

  assert cpp_diff == diff


def test_chrono_steady_clock():
  time1 = m.test_chrono5()
  assert isinstance(time1, datetime.timedelta)


def test_chrono_steady_clock_roundtrip():
  time1 = datetime.timedelta(days=10, seconds=10, microseconds=100)
  time2 = m.test_chrono6(time1)

  assert isinstance(time2, datetime.timedelta)

  assert time1 == time2


def test_floating_point_duration():
  time = m.test_chrono7(35.525123)

  assert isinstance(time, datetime.timedelta)

  assert time.seconds == 35
  assert 525122 <= time.microseconds <= 525123

  diff = m.test_chrono_float_diff(43.789012, 1.123456)
  assert diff.seconds == 42
  assert 665556 <= diff.microseconds <= 665557


def test_nano_timepoint():
  time = datetime.datetime.now()
  time1 = m.test_nano_timepoint(time, datetime.timedelta(seconds=60))
  assert time1 == time + datetime.timedelta(seconds=60)


def test_chrono_different_resolutions():
  resolutions = m.different_resolutions()
  time = datetime.datetime.now()
  resolutions.timestamp_h = time
  resolutions.timestamp_m = time
  resolutions.timestamp_s = time
  resolutions.timestamp_ms = time
  resolutions.timestamp_us = time
