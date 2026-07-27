#
#
#
#

import numpy as np
import pytest

from pylibraft.common import device_ndarray
from pylibraft.matrix import select_k


@pytest.mark.parametrize("n_rows", [32, 100])
@pytest.mark.parametrize("n_cols", [40, 100])
@pytest.mark.parametrize("k", [1, 5, 16, 35])
@pytest.mark.parametrize("inplace", [True, False])
def test_select_k(n_rows, n_cols, k, inplace):
  dataset = np.random.random_sample((n_rows, n_cols)).astype("float32")
  dataset_device = device_ndarray(dataset)

  indices = np.zeros((n_rows, k), dtype="int64")
  distances = np.zeros((n_rows, k), dtype="float32")
  indices_device = device_ndarray(indices)
  distances_device = device_ndarray(distances)

  ret_distances, ret_indices = select_k(
    dataset_device,
    k=k,
    distances=distances_device,
    indices=indices_device,
  )

  distances_device = ret_distances if not inplace else distances_device
  actual_distances = distances_device.copy_to_host()
  argsort = np.argsort(dataset, axis=1)

  for i in range(dataset.shape[0]):
    expected_indices = argsort[i]
    gpu_dists = actual_distances[i]

    cpu_ordered = dataset[i, expected_indices]
    np.testing.assert_allclose(
      cpu_ordered[:k], gpu_dists, atol=1e-4, rtol=1e-4
    )
