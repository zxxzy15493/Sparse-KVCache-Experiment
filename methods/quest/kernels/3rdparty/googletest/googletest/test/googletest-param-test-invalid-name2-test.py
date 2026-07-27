#
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

"""Verifies that Google Test warns the user when not initialized properly."""

from googletest.test import gtest_test_utils

binary_name = 'googletest-param-test-invalid-name2-test_'
COMMAND = gtest_test_utils.GetTestExecutablePath(binary_name)


def Assert(condition):
 if not condition:
  raise AssertionError


def TestExitCodeAndOutput(command):
 """Runs the given command and verifies its exit code and output."""

 err = "Duplicate parameterized test name 'a'"

 p = gtest_test_utils.Subprocess(command)
 Assert(p.terminated_by_signal)

 Assert(err in p.output)


class GTestParamTestInvalidName2Test(gtest_test_utils.TestCase):

 def testExitCodeAndOutput(self):
  TestExitCodeAndOutput(COMMAND)


if __name__ == '__main__':
 gtest_test_utils.Main()
