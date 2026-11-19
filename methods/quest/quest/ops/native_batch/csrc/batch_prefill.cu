/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Adapted from Quest's original batch_prefill.cu.
#include "ops.h"
#include "pytorch_extension_utils.h"
#include "prefill/prefill.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Keep the original attention kernel; compile only the modes this API uses.
torch::Tensor native_prefill(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                             torch::Tensor indptr, torch::Tensor query_indptr, uint32_t last_len) {
    CHECK_INPUT(q); CHECK_INPUT(kv); CHECK_INPUT(indices); CHECK_INPUT(indptr); CHECK_INPUT(query_indptr);
    CHECK_DIM(4, q); CHECK_DIM(5, kv);
    TORCH_CHECK(q.scalar_type() == at::kHalf && kv.scalar_type() == at::kHalf, "Prefill requires float16");
    c10::cuda::CUDAGuard guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint32_t batch = q.size(0), length = q.size(1), q_heads = q.size(2), dim = q.size(3);
    const uint32_t kv_heads = kv.size(3), page_size = kv.size(2), group = q_heads / kv_heads;
    TORCH_CHECK(page_size == 1 || page_size == 8 || page_size == 16 || page_size == 32,
                "Page size must be 1, 8, 16 or 32");
    auto output = torch::empty_like(q);
    quest_native::paged_kv_t<quest_native::PageStorage::kIndices, flashinfer::QKVLayout::kNHD, half, int32_t> cache(
        kv_heads, page_size, dim, batch, 0, last_len, 0, reinterpret_cast<half*>(kv.data_ptr()),
        indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>());
    uint32_t num_frags_x, num_qo_tiles;
    std::vector<int32_t> request_h, tile_h;
    std::tie(num_frags_x, num_qo_tiles, request_h, tile_h) = quest_native::split_qo_indptr(
        query_indptr.data_ptr<int32_t>(), batch, group, stream);
    // PyTorch owns these short-lived buffers, including on nondefault streams.
    auto request = torch::empty({int64_t(request_h.size())}, indices.options());
    auto tile = torch::empty({int64_t(tile_h.size())}, indices.options());
    C10_CUDA_CHECK(cudaMemcpyAsync(request.data_ptr(), request_h.data(), request_h.size()*sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(tile.data_ptr(), tile_h.data(), tile_h.size()*sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
    cudaError_t status = cudaSuccess;
    SWITCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, {
        SWITCH_GQA_GROUP_SIZE(group, GROUP_SIZE, {
            SWITCH_HEAD_DIM_PREFILL(dim, HEAD_DIM, {
                SWITCH_PAGE_SIZE(page_size, PAGE_SIZE, {
                    if constexpr (PAGE_SIZE != 0) {
                        status = quest_native::BatchPrefillWithPagedKVCacheDispatched<
                            quest_native::PageStorage::kIndices, flashinfer::QKVLayout::kNHD,
                            NUM_FRAGS_X, PAGE_SIZE, GROUP_SIZE, HEAD_DIM,
                            flashinfer::RotaryMode::kNone, false, true, half, half, int32_t>(
                                reinterpret_cast<half*>(q.data_ptr()), request.data_ptr<int32_t>(),
                                tile.data_ptr<int32_t>(), query_indptr.data_ptr<int32_t>(), cache,
                                reinterpret_cast<half*>(output.data_ptr()), nullptr, nullptr,
                                num_qo_tiles, 1.f, 1e4f, stream);
                    }
                });
            });
        });
    });
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
