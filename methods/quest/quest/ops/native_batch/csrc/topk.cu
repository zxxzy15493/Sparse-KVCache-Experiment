// Retains Quest's RAFT radix selection; rows now cover requests and heads.
#include "ops.h"
#include "pytorch_extension_utils.h"
#include "topk/decode_select_k.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void native_topk(torch::Tensor scores, torch::Tensor indices, torch::Tensor values,
                 torch::Tensor selected, torch::Tensor scratch) {
    CHECK_INPUT(scores); CHECK_INPUT(indices); CHECK_INPUT(values); CHECK_INPUT(selected); CHECK_INPUT(scratch);
    TORCH_CHECK(scores.scalar_type() == at::kHalf && indices.scalar_type() == at::kInt &&
                selected.scalar_type() == at::kInt, "Expected float16 scores and int32 page indices");
    c10::cuda::CUDAGuard guard(scores.device());
    const int pages = scores.size(-1), k = values.size(-1);
    if (!k) return;
    const int rows = scores.numel() / pages;
    TORCH_CHECK(k <= pages && indices.numel() == scores.numel(), "Invalid TopK capacity");
    decode_select_k<half, int32_t>(reinterpret_cast<half*>(scores.data_ptr()), indices.data_ptr<int32_t>(),
        reinterpret_cast<char*>(scratch.data_ptr()), pages, k, reinterpret_cast<half*>(values.data_ptr()),
        selected.data_ptr<int32_t>(), rows, at::cuda::getCurrentCUDAStream(), true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
