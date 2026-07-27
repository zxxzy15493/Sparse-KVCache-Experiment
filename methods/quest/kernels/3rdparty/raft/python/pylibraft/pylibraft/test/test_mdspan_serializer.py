#
#
#
#

import numpy as np
import pytest

from pylibraft.common.mdspan import run_roundtrip_test_for_mdspan


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "uint32"])
def test_mdspan_serializer(dtype):
  X = np.random.random_sample((2, 3)).astype(dtype)
  run_roundtrip_test_for_mdspan(X)
