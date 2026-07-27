#
#
#
#

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from pylibraft.common import DeviceResources, Stream, device_ndarray
from pylibraft.distance import pairwise_distance


@pytest.mark.parametrize("n_rows", [50, 100])
@pytest.mark.parametrize("n_cols", [10, 50])
@pytest.mark.parametrize(
  "metric",
  [
    "euclidean",
    "cityblock",
    "chebyshev",
    "canberra",
    "correlation",
    "hamming",
    "jensenshannon",
    "russellrao",
    "cosine",
    "sqeuclidean",
    "inner_product",
  ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance(n_rows, n_cols, inplace, metric, order, dtype):
  input1 = np.random.random_sample((n_rows, n_cols))
  input1 = np.asarray(input1, order=order).astype(dtype)

  if metric == "russellrao":
    input1[input1 < 0.5] = 0
    input1[input1 >= 0.5] = 1

  elif metric == "jensenshannon":
    norm = np.sum(input1, axis=1)
    input1 = (input1.T / norm).T

  output = np.zeros((n_rows, n_rows), dtype=dtype)

  if metric == "inner_product":
    expected = np.matmul(input1, input1.T)
  else:
    expected = cdist(input1, input1, metric)

  input1_device = device_ndarray(input1)
  output_device = device_ndarray(output) if inplace else None

  s2 = Stream()
  handle = DeviceResources(stream=s2)
  ret_output = pairwise_distance(
    input1_device, input1_device, output_device, metric, handle=handle
  )
  handle.sync()

  output_device = ret_output if not inplace else output_device

  actual = output_device.copy_to_host()

  assert np.allclose(expected, actual, atol=1e-3, rtol=1e-3)
