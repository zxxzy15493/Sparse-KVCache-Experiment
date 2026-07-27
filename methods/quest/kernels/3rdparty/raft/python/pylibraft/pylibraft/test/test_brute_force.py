#
#
#
#

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from pylibraft.common import DeviceResources, Stream, device_ndarray
from pylibraft.neighbors.brute_force import knn


@pytest.mark.parametrize("n_index_rows", [32, 100])
@pytest.mark.parametrize("n_query_rows", [32, 100])
@pytest.mark.parametrize("n_cols", [40, 100])
@pytest.mark.parametrize("k", [1, 5, 32])
@pytest.mark.parametrize(
  "metric",
  [
    "euclidean",
    "cityblock",
    "chebyshev",
    "canberra",
    "correlation",
    "russellrao",
    "cosine",
    "sqeuclidean",
  ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32])
def test_knn(n_index_rows, n_query_rows, n_cols, k, inplace, metric, dtype):
  index = np.random.random_sample((n_index_rows, n_cols)).astype(dtype)
  queries = np.random.random_sample((n_query_rows, n_cols)).astype(dtype)

  if metric == "russellrao":
    index[index < 0.5] = 0.0
    index[index >= 0.5] = 1.0
    queries[queries < 0.5] = 0.0
    queries[queries >= 0.5] = 1.0

  indices = np.zeros((n_query_rows, k), dtype="int64")
  distances = np.zeros((n_query_rows, k), dtype=dtype)

  index_device = device_ndarray(index)

  queries_device = device_ndarray(queries)
  indices_device = device_ndarray(indices)
  distances_device = device_ndarray(distances)

  s2 = Stream()
  handle = DeviceResources(stream=s2)
  ret_distances, ret_indices = knn(
    index_device,
    queries_device,
    k,
    indices=indices_device,
    distances=distances_device,
    metric=metric,
    handle=handle,
  )
  handle.sync()

  pw_dists = cdist(queries, index, metric=metric)

  distances_device = ret_distances if not inplace else distances_device

  actual_distances = distances_device.copy_to_host()

  actual_distances[actual_distances <= 1e-5] = 0.0
  argsort = np.argsort(pw_dists, axis=1)

  for i in range(pw_dists.shape[0]):
    expected_indices = argsort[i]
    gpu_dists = actual_distances[i]

    cpu_ordered = pw_dists[i, expected_indices]
    np.testing.assert_allclose(
      cpu_ordered[:k], gpu_dists, atol=1e-3, rtol=1e-3
    )


def test_knn_check_col_major_inputs():
  cp = pytest.importorskip("cupy")
  n_index_rows, n_query_rows, n_cols = 128, 16, 32
  index = cp.random.random_sample((n_index_rows, n_cols), dtype="float32")
  queries = cp.random.random_sample((n_query_rows, n_cols), dtype="float32")

  with pytest.raises(ValueError):
    knn(cp.asarray(index, order="F"), queries, k=4)

  with pytest.raises(ValueError):
    knn(index, cp.asarray(queries, order="F"), k=4)

  knn(index, queries, k=4)
