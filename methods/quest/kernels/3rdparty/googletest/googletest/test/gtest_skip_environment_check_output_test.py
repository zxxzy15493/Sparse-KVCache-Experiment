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
"""Tests Google Test's gtest skip in environment setup behavior.

This script invokes gtest_skip_in_environment_setup_test_ and verifies its
output.
"""

from googletest.test import gtest_test_utils

EXE_PATH = gtest_test_utils.GetTestExecutablePath(
  'gtest_skip_in_environment_setup_test'
)

OUTPUT = gtest_test_utils.Subprocess([EXE_PATH]).output


class SkipEntireEnvironmentTest(gtest_test_utils.TestCase):

 def testSkipEntireEnvironmentTest(self):
  self.assertIn('Skipping the entire environment', OUTPUT)
  self.assertNotIn('FAILED', OUTPUT)


if __name__ == '__main__':
 gtest_test_utils.Main()
