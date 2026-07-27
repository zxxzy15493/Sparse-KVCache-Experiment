#
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Unit test utilities for Google C++ Mocking Framework."""

import os

from googletest.test import gtest_test_utils


def GetSourceDir():
 """Returns the absolute path of the directory where the .py files are."""

 return gtest_test_utils.GetSourceDir()


def GetTestExecutablePath(executable_name):
 """Returns the absolute path of the test binary given its name.

 The function will print a message and abort the program if the resulting file
 doesn't exist.

 Args:
  executable_name: name of the test binary that the test script runs.

 Returns:
  The absolute path of the test binary.
 """

 return gtest_test_utils.GetTestExecutablePath(executable_name)


def GetExitStatus(exit_code):
 """Returns the argument to exit(), or -1 if exit() wasn't called.

 Args:
  exit_code: the result value of os.system(command).
 """

 if os.name == 'nt':
  return exit_code
 else:
  if os.WIFEXITED(exit_code):
   return os.WEXITSTATUS(exit_code)
  else:
   return -1


Subprocess = gtest_test_utils.Subprocess
TestCase = gtest_test_utils.TestCase
environ = gtest_test_utils.environ
SetEnvVar = gtest_test_utils.SetEnvVar
PREMATURE_EXIT_FILE_ENV_VAR = gtest_test_utils.PREMATURE_EXIT_FILE_ENV_VAR


def Main():
 """Runs the unit test."""

 gtest_test_utils.Main()
