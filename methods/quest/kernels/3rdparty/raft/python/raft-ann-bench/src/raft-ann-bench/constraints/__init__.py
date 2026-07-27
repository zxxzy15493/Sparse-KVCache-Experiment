#
#
#
#

DTYPE_SIZES = {"float": 4, "half": 2, "fp8": 1}


def raft_cagra_build_constraints(params, dims):
  if "graph_degree" in params and "intermediate_graph_degree" in params:
    return params["graph_degree"] <= params["intermediate_graph_degree"]
  return True


def raft_ivf_pq_build_constraints(params, dims):
  if "pq_dim" in params:
    return params["pq_dim"] <= dims
  return True


def raft_ivf_pq_search_constraints(params, build_params, k, batch_size):
  ret = True
  if "internalDistanceDtype" in params and "smemLutDtype" in params:
    ret = (
      DTYPE_SIZES[params["smemLutDtype"]]
      <= DTYPE_SIZES[params["internalDistanceDtype"]]
    )

  if "nlist" in build_params and "nprobe" in params:
    ret = ret and build_params["nlist"] >= params["nprobe"]
  return ret


def raft_cagra_search_constraints(params, build_params, k, batch_size):
  ret = True
  if "itopk" in params:
    ret = ret and params["itopk"] >= k
  return ret


def hnswlib_search_constraints(params, build_params, k, batch_size):
  if "ef" in params:
    return params["ef"] >= k
