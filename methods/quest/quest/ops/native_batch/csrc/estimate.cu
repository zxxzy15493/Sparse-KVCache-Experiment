// Retains Quest's min/max scoring implementation and accumulation behavior.
#include "ops.h"
#include "pytorch_extension_utils.h"
#include "decode/decode_attn.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void native_estimate(torch::Tensor q, torch::Tensor out, torch::Tensor metadata,
                     torch::Tensor indices, torch::Tensor indptr, uint32_t last_len) {
    CHECK_INPUT(q); CHECK_INPUT(out); CHECK_INPUT(metadata); CHECK_DIM(3, q);
    TORCH_CHECK(q.scalar_type() == at::kHalf, "Scoring requires float16");
    c10::cuda::CUDAGuard guard(q.device());
    using Cache = quest_native::paged_kv_t<quest_native::PageStorage::kIndices,
                                         flashinfer::QKVLayout::kNHD, half, int32_t>;
    Cache cache(q.size(1), metadata.size(2), q.size(2), q.size(0), 0, last_len, 0,
                reinterpret_cast<half*>(metadata.data_ptr()), indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>());
    auto status = quest_native::MaxPossibleSampleWithPagedKVCache(
        reinterpret_cast<half*>(q.data_ptr()), cache, reinterpret_cast<half*>(out.data_ptr()),
        q.size(1), flashinfer::RotaryMode::kNone, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
