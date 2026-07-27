#
#
#
#

import numpy as np
import pytest

from pylibraft.common import DeviceResources, Stream, device_ndarray
from pylibraft.distance import pairwise_distance

cupy = pytest.importorskip("cupy")


@pytest.mark.parametrize("stream", [cupy.cuda.Stream().ptr, Stream()])
def test_handle_external_stream(stream):

  input1 = np.random.random_sample((50, 3))
  input1 = np.asarray(input1, order="F").astype("float")

  output = np.zeros((50, 50), dtype="float")

  input1_device = device_ndarray(input1)
  output_device = device_ndarray(output)

  handle = DeviceResources(stream)
  pairwise_distance(
    input1_device, input1_device, output_device, "euclidean", handle=handle
  )
  handle.sync()

  with pytest.raises(ValueError):
    handle = DeviceResources(stream=1.0)
