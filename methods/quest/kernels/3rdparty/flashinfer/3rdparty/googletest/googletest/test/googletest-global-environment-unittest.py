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
"""Unit test for Google Test's global test environment behavior.

A user can specify a global test environment via
testing::AddGlobalTestEnvironment. Failures in the global environment should
result in all unit tests being skipped.

This script tests such functionality by invoking
googletest-global-environment-unittest_ (a program written with Google Test).
"""

import re
from googletest.test import gtest_test_utils


def RunAndReturnOutput(args=None):
 """Runs the test program and returns its output."""

 return gtest_test_utils.Subprocess(
   [
     gtest_test_utils.GetTestExecutablePath(
       'googletest-global-environment-unittest_'
     )
   ]
   + (args or [])
 ).output


class GTestGlobalEnvironmentUnitTest(gtest_test_utils.TestCase):
 """Tests global test environment failures."""

 def testEnvironmentSetUpFails(self):
  """Tests the behavior of not specifying the fail_fast."""

  txt = RunAndReturnOutput()

  self.assertIn('Canned environment setup error', txt)

  self.assertIn('[ SKIPPED ] 1 test', txt)
  self.assertIn('[ PASSED ] 0 tests', txt)

  self.assertNotIn('Unexpected call', txt)

 def testEnvironmentSetUpAndTornDownForEachRepeat(self):
  """Tests the behavior of test environments and gtest_repeat."""

  txt = RunAndReturnOutput([
    '--gtest_repeat=2',
    '--gtest_recreate_environments_when_repeating=true',
  ])

  expected_pattern = (
    '(.|\n)*'
    r'Repeating all tests \(iteration 1\)'
    '(.|\n)*'
    'Global test environment set-up.'
    '(.|\n)*'
    'SomeTest.DoesFoo'
    '(.|\n)*'
    'Global test environment tear-down'
    '(.|\n)*'
    r'Repeating all tests \(iteration 2\)'
    '(.|\n)*'
    'Global test environment set-up.'
    '(.|\n)*'
    'SomeTest.DoesFoo'
    '(.|\n)*'
    'Global test environment tear-down'
    '(.|\n)*'
  )
  self.assertRegex(txt, expected_pattern)

 def testEnvironmentSetUpAndTornDownOnce(self):
  """Tests environment and --gtest_recreate_environments_when_repeating."""

  txt = RunAndReturnOutput(
    [
      '--gtest_repeat=2',
    ]
  )

  expected_pattern = (
    '(.|\n)*'
    r'Repeating all tests \(iteration 1\)'
    '(.|\n)*'
    'Global test environment set-up.'
    '(.|\n)*'
    'SomeTest.DoesFoo'
    '(.|\n)*'
    r'Repeating all tests \(iteration 2\)'
    '(.|\n)*'
    'SomeTest.DoesFoo'
    '(.|\n)*'
    'Global test environment tear-down'
    '(.|\n)*'
  )
  self.assertRegex(txt, expected_pattern)

  self.assertEqual(len(re.findall('Global test environment set-up', txt)), 1)
  self.assertEqual(
    len(re.findall('Global test environment tear-down', txt)), 1
  )


if __name__ == '__main__':
 gtest_test_utils.Main()
