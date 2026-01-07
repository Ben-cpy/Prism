#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-graph.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "H2O intra-attn test failed: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

llama_hparams make_hparams(uint32_t n_layer, uint32_t n_head, uint32_t n_head_kv, uint32_t n_embd_head) {
    llama_hparams hparams{};
    hparams.n_layer = n_layer;
    hparams.n_embd = n_head * n_embd_head;
    hparams.n_embd_head_k = n_embd_head;
    hparams.n_embd_head_v = n_embd_head;
    hparams.n_rot = n_embd_head;
    hparams.f_norm_eps = 1e-5f;
    hparams.f_norm_rms_eps = 1e-5f;
    hparams.f_attn_logit_softcapping = 50.0f;
    hparams.f_max_alibi_bias = 0.0f;
    hparams.attn_soft_cap = false;
    hparams.causal_attn = true;
    hparams.rope_type = LLAMA_ROPE_TYPE_NONE;
    hparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    hparams.n_head_arr[0] = n_head;
    hparams.n_head_kv_arr[0] = n_head_kv;
    hparams.n_ff_arr[0] = 0;
    return hparams;
}

llama_cparams make_cparams(uint32_t n_ctx) {
    llama_cparams cparams{};
    cparams.n_ctx = n_ctx;
    cparams.n_ctx_seq = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = n_ctx;
    cparams.n_seq_max = 1;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    cparams.rope_freq_base = 10000.0f;
    cparams.rope_freq_scale = 1.0f;
    cparams.n_ctx_orig_yarn = n_ctx;
    cparams.yarn_ext_factor = -1.0f;
    cparams.yarn_attn_factor = 1.0f;
    cparams.yarn_beta_fast = 32.0f;
    cparams.yarn_beta_slow = 1.0f;
    cparams.embeddings = false;
    cparams.causal_attn = true;
    cparams.offload_kqv = true;
    cparams.flash_attn = false;
    cparams.no_perf = true;
    cparams.warmup = false;
    cparams.op_offload = false;
    cparams.kv_unified = false;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.cb_eval = nullptr;
    cparams.cb_eval_user_data = nullptr;
    return cparams;
}

llama_ubatch make_ubatch(uint32_t n_tokens) {
    llama_ubatch ubatch{};
    ubatch.b_equal_seqs = 1;
    ubatch.n_tokens = n_tokens;
    ubatch.n_seq_tokens = n_tokens;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;
    return ubatch;
}

void fill_tensor(ggml_tensor * tensor, const std::vector<float> & data) {
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}
}

int main() {
    llama_backend_init();

    const int32_t il = 0;
    const int64_t n_tokens = 4;
    const int64_t n_kv = n_tokens;
    const int64_t n_head = 2;
    const int64_t n_embd_head = 4;

    llama_hparams hparams = make_hparams(1, static_cast<uint32_t>(n_head), static_cast<uint32_t>(n_head), static_cast<uint32_t>(n_embd_head));
    llama_cparams cparams = make_cparams(static_cast<uint32_t>(n_tokens));
    llama_ubatch ubatch = make_ubatch(static_cast<uint32_t>(n_tokens));

    llm_graph_result graph_result(1024);
    llm_graph_params params{};
    params.arch = LLM_ARCH_UNKNOWN;
    params.hparams = hparams;
    params.cparams = cparams;
    params.ubatch = ubatch;
    params.gtype = LLM_GRAPH_TYPE_DEFAULT;
    params.sched = nullptr;
    params.backend_cpu = nullptr;
    params.cvec = nullptr;
    params.loras = nullptr;
    params.mctx = nullptr;
    params.cross = nullptr;
    params.n_outputs = static_cast<uint32_t>(n_tokens);
    params.cb = nullptr;
    params.res = &graph_result;

    llm_graph_context graph(params);

    ggml_tensor * q = ggml_new_tensor_3d(graph.ctx0, GGML_TYPE_F32, n_embd_head, n_head, n_tokens);
    ggml_tensor * k = ggml_new_tensor_3d(graph.ctx0, GGML_TYPE_F32, n_embd_head, n_head, n_kv);
    ggml_tensor * v = ggml_new_tensor_3d(graph.ctx0, GGML_TYPE_F32, n_embd_head, n_head, n_kv);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(graph.ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1, 1);

    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);
    ggml_set_input(kq_mask);

    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(n_embd_head));

    ggml_tensor * out_std = graph.build_attn_mha(q, k, v, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);

    ggml_tensor * attn_weights = nullptr;
    ggml_tensor * out_h2o = graph.build_attn_mha_h2o(q, k, v, nullptr, kq_mask, nullptr, nullptr, kq_scale, il, &attn_weights);
    require(attn_weights != nullptr, "attn_weights_out not set");

    ggml_tensor * colsum = graph.build_attn_colsum(attn_weights, il);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to init CPU backend");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(graph.ctx0, backend);
    require(buf != nullptr, "failed to allocate backend buffer");

    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> k_data(ggml_nelements(k));
    std::vector<float> v_data(ggml_nelements(v));

    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.01f * static_cast<float>(i % 37);
    }
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = 0.02f * static_cast<float>((i * 7) % 29);
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        v_data[i] = 0.03f * static_cast<float>((i * 5) % 31);
    }

    std::vector<float> mask_data(ggml_nelements(kq_mask));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t q_pos = 0; q_pos < n_tokens; ++q_pos) {
        for (int64_t k_pos = 0; k_pos < n_kv; ++k_pos) {
            const size_t idx = static_cast<size_t>(k_pos + q_pos * n_kv);
            mask_data[idx] = (k_pos > q_pos) ? neg_inf : 0.0f;
        }
    }

    fill_tensor(q, q_data);
    fill_tensor(k, k_data);
    fill_tensor(v, v_data);
    fill_tensor(kq_mask, mask_data);

    ggml_status status = ggml_backend_graph_compute(backend, graph.gf);
    require(status == GGML_STATUS_SUCCESS, "graph compute failed");

    std::vector<float> out_std_data(ggml_nelements(out_std));
    std::vector<float> out_h2o_data(ggml_nelements(out_h2o));

    ggml_backend_tensor_get(out_std, out_std_data.data(), 0, out_std_data.size() * sizeof(float));
    ggml_backend_tensor_get(out_h2o, out_h2o_data.data(), 0, out_h2o_data.size() * sizeof(float));

    float max_diff = 0.0f;
    for (size_t i = 0; i < out_std_data.size(); ++i) {
        float diff = std::fabs(out_std_data[i] - out_h2o_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    require(max_diff < 1e-5f, "H2O output does not match standard attention");

    std::vector<float> attn_data(ggml_nelements(attn_weights));
    ggml_backend_tensor_get(attn_weights, attn_data.data(), 0, attn_data.size() * sizeof(float));

    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t q_pos = 0; q_pos < n_tokens; ++q_pos) {
            float row_sum = 0.0f;
            for (int64_t k_pos = 0; k_pos < n_kv; ++k_pos) {
                const size_t idx = static_cast<size_t>(k_pos + q_pos * n_kv + h * n_kv * n_tokens);
                const float w = attn_data[idx];
                row_sum += w;
                if (k_pos > q_pos) {
                    require(std::fabs(w) < 1e-6f, "causal mask leaked future attention");
                }
            }
            require(std::fabs(row_sum - 1.0f) < 1e-5f, "row sum not normalized");
        }
    }

    std::vector<float> colsum_data(ggml_nelements(colsum));
    ggml_backend_tensor_get(colsum, colsum_data.data(), 0, colsum_data.size() * sizeof(float));

    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t k_pos = 0; k_pos < n_kv; ++k_pos) {
            float expected = 0.0f;
            for (int64_t q_pos = 0; q_pos < n_tokens; ++q_pos) {
                const size_t idx = static_cast<size_t>(k_pos + q_pos * n_kv + h * n_kv * n_tokens);
                expected += attn_data[idx];
            }
            const size_t col_idx = static_cast<size_t>(k_pos + h * n_kv);
            require(std::fabs(expected - colsum_data[col_idx]) < 1e-4f, "column sum mismatch");
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);

    llama_backend_free();

    fprintf(stdout, "H2O intra-attn checks passed\n");
    return EXIT_SUCCESS;
}
