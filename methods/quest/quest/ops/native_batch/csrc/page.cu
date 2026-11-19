// Adapted from Quest's original page.cu; equal-length requests share positions.
#include "ops.h"
#include "pytorch_extension_utils.h"
#include "decode/decode_page.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

using namespace quest_native;

void native_append(torch::Tensor k, torch::Tensor v, torch::Tensor kv, torch::Tensor metadata,
                   torch::Tensor indices, torch::Tensor metadata_indices, torch::Tensor indptr,
                   torch::Tensor metadata_indptr, uint32_t last_len, uint32_t metadata_last_len) {
    CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(kv); CHECK_INPUT(metadata);
    CHECK_DIM(4, k); CHECK_SHAPE(k, v);
    TORCH_CHECK(k.scalar_type() == at::kHalf && kv.scalar_type() == at::kHalf,
                "Native Quest currently supports float16");
    c10::cuda::CUDAGuard guard(k.device());
    const int batch = k.size(0), length = k.size(1), heads = k.size(2), dim = k.size(3);
    auto stream = at::cuda::getCurrentCUDAStream();
    using Cache = quest_native::paged_kv_t<quest_native::PageStorage::kIndices,
                                         flashinfer::QKVLayout::kNHD, half, int32_t>;
    Cache cache(heads, kv.size(2), dim, batch, 0, last_len, 0,
                reinterpret_cast<half*>(kv.data_ptr()), indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>());
    Cache summaries(heads, metadata.size(2), dim, batch, 0, metadata_last_len, 0,
                    reinterpret_cast<half*>(metadata.data_ptr()), metadata_indices.data_ptr<int32_t>(),
                    metadata_indptr.data_ptr<int32_t>());
    cudaError_t status;
    if (length == 1) {
        status = AppendPagedKVCacheDecode(cache, summaries, reinterpret_cast<half*>(k.data_ptr()),
                                          reinterpret_cast<half*>(v.data_ptr()), stream);
    } else {
        auto append_indptr = torch::arange(batch + 1, indices.options()) * length;
        status = AppendPagedKVCachePrefill(cache, summaries, reinterpret_cast<half*>(k.data_ptr()),
                                           reinterpret_cast<half*>(v.data_ptr()), append_indptr.data_ptr<int32_t>(), stream);
    }
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void native_rope(torch::Tensor q, torch::Tensor k, uint32_t offset, float scale, float theta) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_DIM(4, q); CHECK_DIM(4, k);
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf, "RoPE requires float16");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(3) == k.size(3), "RoPE shapes disagree");
    c10::cuda::CUDAGuard guard(q.device());
    auto status = QKApplyRotaryInPlace(reinterpret_cast<half*>(q.data_ptr()),
                                      reinterpret_cast<half*>(k.data_ptr()), q.size(1), offset,
                                      q.size(2), k.size(2), q.size(3), scale, theta, q.size(0),
                                      at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
