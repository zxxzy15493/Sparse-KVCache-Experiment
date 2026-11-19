#pragma once
#include <torch/extension.h>
#include <memory>

void native_append(torch::Tensor k, torch::Tensor v, torch::Tensor kv, torch::Tensor metadata,
                   torch::Tensor indices, torch::Tensor metadata_indices, torch::Tensor indptr,
                   torch::Tensor metadata_indptr, uint32_t last_len, uint32_t metadata_last_len);
void native_rope(torch::Tensor q, torch::Tensor k, uint32_t offset, float scale, float theta);
void rms_norm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor out, float epsilon);
void native_estimate(torch::Tensor q, torch::Tensor out, torch::Tensor metadata,
                     torch::Tensor indices, torch::Tensor indptr, uint32_t last_len);
void native_topk(torch::Tensor scores, torch::Tensor indices, torch::Tensor values,
                 torch::Tensor selected, torch::Tensor scratch);
torch::Tensor native_prefill(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                             torch::Tensor indptr, torch::Tensor query_indptr, uint32_t last_len);

class NativeDecodePlan {
 public:
  NativeDecodePlan(torch::Tensor indptr_cpu, uint32_t q_heads, uint32_t kv_heads,
                   uint32_t dim, uint32_t page_size, bool query_head_selection);
  ~NativeDecodePlan();
  torch::Tensor run(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                     torch::Tensor indptr, torch::Tensor last_indices, uint32_t last_len);
  void close();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
