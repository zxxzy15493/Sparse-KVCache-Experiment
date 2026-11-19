// Retains Quest's partitioned sparse attention and explicit final-page chunk.
#include "ops.h"
#include "pytorch_extension_utils.h"
#include "decode/decode_handler.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

struct NativeDecodePlan::Impl {
    quest_native::BatchDecodeHandler handler;
    uint32_t batch, q_heads, kv_heads, dim, page_size, prior_pages;
    bool query_head_selection, closed = false;
    torch::Device device;
    Impl(torch::Tensor ptr, uint32_t q, uint32_t k, uint32_t d, uint32_t p, bool per_head)
      : batch(ptr.numel() - 1), q_heads(q), kv_heads(k), dim(d), page_size(p),
        prior_pages(ptr.data_ptr<int32_t>()[1]), query_head_selection(per_head),
        device(torch::kCUDA, c10::cuda::current_device()) {
        handler.SetCUDAStream(at::cuda::getCurrentCUDAStream());
        if (prior_pages) {
            auto status = handler.BeginForward<quest_native::PageStorage::kIndices, flashinfer::QKVLayout::kNHD,
                                               half, half, int32_t>(ptr.data_ptr<int32_t>(), batch,
                q_heads, per_head ? q_heads : kv_heads, dim, page_size, flashinfer::RotaryMode::kNone);
            TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
        }
    }
};

NativeDecodePlan::NativeDecodePlan(torch::Tensor ptr, uint32_t q, uint32_t k, uint32_t d, uint32_t p, bool per_head) {
    TORCH_CHECK(ptr.device().is_cpu() && ptr.scalar_type() == at::kInt && ptr.is_contiguous(), "Plan indptr must be contiguous int32 on CPU");
    TORCH_CHECK(ptr.numel() >= 2 && q % k == 0, "Invalid batch/head counts");
    impl_ = std::make_unique<Impl>(ptr, q, k, d, p, per_head);
}
NativeDecodePlan::~NativeDecodePlan() { close(); }
void NativeDecodePlan::close() { if (impl_ && !impl_->closed) { c10::cuda::CUDAGuard guard(impl_->device); impl_->handler.EndForward(); impl_->closed = true; } }

torch::Tensor NativeDecodePlan::run(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                                   torch::Tensor indptr, torch::Tensor last_indices, uint32_t last_len) {
    auto& s = *impl_;
    TORCH_CHECK(!s.closed && q.device() == s.device, "Decode plan is closed or on the wrong device");
    CHECK_INPUT(q); CHECK_INPUT(kv); CHECK_INPUT(indices); CHECK_INPUT(indptr); CHECK_INPUT(last_indices);
    TORCH_CHECK(q.scalar_type() == at::kHalf && q.size(0) == s.batch && q.size(1) == s.q_heads &&
                q.size(2) == s.dim, "Query does not match plan");
    c10::cuda::CUDAGuard guard(q.device());
    s.handler.SetCUDAStream(at::cuda::getCurrentCUDAStream());
    using Cache = quest_native::paged_kv_t<quest_native::PageStorage::kIndices,
                                         flashinfer::QKVLayout::kNHD, half, int32_t>;
    Cache cache(s.kv_heads, s.page_size, s.dim, s.batch, s.prior_pages, last_len, 0,
                reinterpret_cast<half*>(kv.data_ptr()), indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>());
    cache.request_last_page_indices = last_indices.data_ptr<int32_t>();
    cache.query_heads = s.q_heads;
    cache.per_query_head = s.query_head_selection;
    auto output = torch::empty_like(q);
    cudaError_t status;
    if (s.prior_pages) {
        status = quest_native::BatchDecodeWithPagedKVCacheWrapper(&s.handler,
            reinterpret_cast<half*>(q.data_ptr()), cache, reinterpret_cast<half*>(output.data_ptr()),
            static_cast<float*>(nullptr), s.q_heads, flashinfer::RotaryMode::kNone, 1.f, 1e4f,
            at::cuda::getCurrentCUDAStream());
    } else {
        quest_native::kv_partition_info_t<int32_t> partition;
        status = quest_native::BatchDecodeWithPagedKVCache(reinterpret_cast<half*>(q.data_ptr()),
            cache, partition, reinterpret_cast<half*>(output.data_ptr()), static_cast<half*>(nullptr),
            static_cast<float*>(nullptr), s.q_heads, flashinfer::RotaryMode::kNone, 1.f, 1e4f,
            at::cuda::getCurrentCUDAStream());
    }
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
