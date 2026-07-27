#
#
#
#


import cupy as cp
import h5py
import os
import tempfile
import time
import urllib

def calc_recall(found_indices, ground_truth):
  found_indices = cp.asarray(found_indices)
  bs, k = found_indices.shape
  if bs != ground_truth.shape[0]:
    raise RuntimeError(
      "Batch sizes do not match {} vs {}".format(
        bs, ground_truth.shape[0]
      )
    )
  if k > ground_truth.shape[1]:
    raise RuntimeError(
      "Not enough indices in the ground truth ({} > {})".format(
        k, ground_truth.shape[1]
      )
    )
  n = 0
  for i in range(bs):
    n += cp.intersect1d(found_indices[i, :k], ground_truth[i, :k]).size
  recall = n / found_indices.size
  return recall


class BenchmarkTimer:
  """Provides a context manager that runs a code block `reps` times
  and records results to the instance variable `timings`. Use like:
  .. code-block:: python
    timer = BenchmarkTimer(rep=5)
    for _ in timer.benchmark_runs():
      ... do something ...
    print(np.min(timer.timings))

    This class is borrowed from the rapids/cuml benchmark suite
  """

  def __init__(self, reps=1, warmup=0):
    self.warmup = warmup
    self.reps = reps
    self.timings = []

  def benchmark_runs(self):
    for r in range(self.reps + self.warmup):
      t0 = time.time()
      yield r
      t1 = time.time()
      self.timings.append(t1 - t0)
      if r >= self.warmup:
        self.timings.append(t1 - t0)


def load_dataset(dataset_url="http://ann-benchmarks.com/sift-128-euclidean.hdf5", work_folder=None):
  """Download dataset from url. It is expected that the dataset contains a hdf5 file in ann-benchmarks format

  Parameters
  ----------
   dataset_url address of hdf5 file
   work_folder name of the local folder to store the dataset

  """
  dataset_filename = dataset_url.split("/")[-1]

  if work_folder is None:
    work_folder = os.path.join(tempfile.gettempdir(), "raft_example")

  if not os.path.exists(work_folder):
    os.makedirs(work_folder)
  print("The index and data will be saved in", work_folder)

  dataset_path = os.path.join(work_folder, dataset_filename)
  if not os.path.exists(dataset_path):
    urllib.request.urlretrieve(dataset_url, dataset_path)

  f = h5py.File(dataset_path, "r")

  return f
