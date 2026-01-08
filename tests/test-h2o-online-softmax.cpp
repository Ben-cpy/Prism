#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "H2O online softmax test failed: %s\n", message);
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
    const int64_t n_tokens = 2;
    const int64_t n_head = 1;
    const int64_t n_stream = 1;
    const int64_t n_embd_head_v = 4;
    const int64_t n_kv_inter = 2;
    const int64_t n_kv_intra = 2;

    llama_hparams hparams = make_hparams(1, static_cast<uint32_t>(n_head), static_cast<uint32_t>(n_head), static_cast<uint32_t>(n_embd_head_v));
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

    ggml_tensor * inter_logits = ggml_new_tensor_4d(graph.ctx0, GGML_TYPE_F32, n_kv_inter, n_tokens, n_head, n_stream);
    ggml_tensor * intra_logits = ggml_new_tensor_4d(graph.ctx0, GGML_TYPE_F32, n_kv_intra, n_tokens, n_head, n_stream);
    ggml_tensor * v_inter = ggml_new_tensor_4d(graph.ctx0, GGML_TYPE_F32, n_embd_head_v, n_kv_inter, n_head, n_stream);
    ggml_tensor * v_intra = ggml_new_tensor_4d(graph.ctx0, GGML_TYPE_F32, n_embd_head_v, n_kv_intra, n_head, n_stream);

    ggml_set_input(inter_logits);
    ggml_set_input(intra_logits);
    ggml_set_input(v_inter);
    ggml_set_input(v_intra);

    llm_graph_result::h2o_online_softmax_state state{};
    graph.init_online_softmax_state_h2o(inter_logits, v_inter, state);
    ggml_tensor * out = graph.update_online_softmax_state_h2o(state, intra_logits, v_intra);
    ggml_build_forward_expand(graph.gf, out);
    ggml_set_output(out);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to init CPU backend");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(graph.ctx0, backend);
    require(buf != nullptr, "failed to allocate backend buffer");

    std::vector<float> inter_logits_data(ggml_nelements(inter_logits));
    std::vector<float> intra_logits_data(ggml_nelements(intra_logits));
    std::vector<float> v_inter_data(ggml_nelements(v_inter));
    std::vector<float> v_intra_data(ggml_nelements(v_intra));

    // inter logits: lower max
    inter_logits_data[0] = 10.0f;
    inter_logits_data[1] = 12.0f;
    inter_logits_data[2] = -100.0f;
    inter_logits_data[3] = 11.0f;

    // intra logits: higher max and a very low value to trigger FTZ
    intra_logits_data[0] = 50.0f;
    intra_logits_data[1] = 48.0f;
    intra_logits_data[2] = -100.0f;
    intra_logits_data[3] = 49.0f;

    for (size_t i = 0; i < v_inter_data.size(); ++i) {
        v_inter_data[i] = 0.01f * static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < v_intra_data.size(); ++i) {
        v_intra_data[i] = 0.02f * static_cast<float>(i + 1);
    }

    fill_tensor(inter_logits, inter_logits_data);
    fill_tensor(intra_logits, intra_logits_data);
    fill_tensor(v_inter, v_inter_data);
    fill_tensor(v_intra, v_intra_data);

    ggml_status status = ggml_backend_graph_compute(backend, graph.gf);
    require(status == GGML_STATUS_SUCCESS, "graph compute failed");

    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));

    // Reference computation with max tracking and FTZ threshold.
    constexpr float k_ftz_threshold = -20.0f;
    std::vector<float> ref(out_data.size(), 0.0f);

    for (int64_t t = 0; t < n_tokens; ++t) {
        const int64_t h = 0;
        float m = -INFINITY;
        for (int64_t k = 0; k < n_kv_inter; ++k) {
            const size_t idx = static_cast<size_t>(k + t * n_kv_inter);
            m = std::max(m, inter_logits_data[idx]);
        }
        for (int64_t k = 0; k < n_kv_intra; ++k) {
            const size_t idx = static_cast<size_t>(k + t * n_kv_intra);
            m = std::max(m, intra_logits_data[idx]);
        }

        float l = 0.0f;
        std::vector<float> acc(n_embd_head_v, 0.0f);

        for (int64_t k = 0; k < n_kv_inter; ++k) {
            const size_t idx = static_cast<size_t>(k + t * n_kv_inter);
            const float shifted = inter_logits_data[idx] - m;
            if (shifted > k_ftz_threshold) {
                const float w = std::exp(shifted);
                l += w;
                for (int64_t d = 0; d < n_embd_head_v; ++d) {
                    const size_t v_idx = static_cast<size_t>(d + k * n_embd_head_v);
                    acc[d] += v_inter_data[v_idx] * w;
                }
            }
        }

        for (int64_t k = 0; k < n_kv_intra; ++k) {
            const size_t idx = static_cast<size_t>(k + t * n_kv_intra);
            const float shifted = intra_logits_data[idx] - m;
            if (shifted > k_ftz_threshold) {
                const float w = std::exp(shifted);
                l += w;
                for (int64_t d = 0; d < n_embd_head_v; ++d) {
                    const size_t v_idx = static_cast<size_t>(d + k * n_embd_head_v);
                    acc[d] += v_intra_data[v_idx] * w;
                }
            }
        }

        for (int64_t d = 0; d < n_embd_head_v; ++d) {
            const size_t out_idx = static_cast<size_t>(d + t * n_embd_head_v + h * n_embd_head_v * n_tokens);
            ref[out_idx] = acc[d] / l;
        }
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(ref[i] - out_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    require(max_diff < 1e-5f, "online softmax output mismatch");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    llama_backend_free();

    fprintf(stdout, "H2O online softmax checks passed\n");
    return EXIT_SUCCESS;
}
