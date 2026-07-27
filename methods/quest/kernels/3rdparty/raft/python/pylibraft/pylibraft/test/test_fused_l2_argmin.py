#
#
#
#

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from pylibraft.common import DeviceResources, device_ndarray
from pylibraft.distance import fused_l2_nn_argmin


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("n_rows", [10, 100])
@pytest.mark.parametrize("n_clusters", [5, 10])
@pytest.mark.parametrize("n_cols", [3, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_l2_nn_minarg(n_rows, n_cols, n_clusters, dtype, inplace):
  input1 = np.random.random_sample((n_rows, n_cols))
  input1 = np.asarray(input1, order="C").astype(dtype)

  input2 = np.random.random_sample((n_clusters, n_cols))
  input2 = np.asarray(input2, order="C").astype(dtype)

  output = np.zeros((n_rows), dtype="int32")
  expected = cdist(input1, input2, metric="euclidean")

  expected = expected.argmin(axis=1)

  input1_device = device_ndarray(input1)
  input2_device = device_ndarray(input2)
  output_device = device_ndarray(output) if inplace else None

  handle = DeviceResources()
  ret_output = fused_l2_nn_argmin(
    input1_device, input2_device, output_device, True, handle=handle
  )
  handle.sync()
  output_device = ret_output if not inplace else output_device
  actual = output_device.copy_to_host()

  assert np.allclose(expected, actual, rtol=1e-4)
