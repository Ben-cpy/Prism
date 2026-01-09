#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "get-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "H2O score tracking test failed: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

void log_step(const char * message) {
    fprintf(stderr, "[H2O SCORE] %s\n", message);
    std::fflush(stderr);
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
    cparams.h2o_local_window = 0;
    cparams.h2o_heavy_budget = 0;
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

void run_colsum_test() {
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

    std::vector<float> attn_data(ggml_nelements(attn_weights));
    ggml_backend_tensor_get(attn_weights, attn_data.data(), 0, attn_data.size() * sizeof(float));

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
    GGML_UNUSED(out_h2o);
}

int32_t find_first_kv_layer(const llama_hparams & hparams) {
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (hparams.has_kv(il)) {
            return static_cast<int32_t>(il);
        }
    }
    return -1;
}

void run_score_init_test(llama_model & model, int32_t il) {
    const uint32_t kv_size = 128;
    const uint32_t h2o_local = 8;
    const uint32_t h2o_heavy = 8;
    const uint32_t chunk_len = 16;

    llama_kv_cache kv(
            model,
            GGML_TYPE_F16,
            GGML_TYPE_F16,
            true,
            false,
            true,
            kv_size,
            1,
            1,
            0,
            LLAMA_SWA_TYPE_H2O,
            nullptr,
            nullptr,
            h2o_local,
            h2o_heavy);

    const ggml_tensor * scores = kv.h2o_get_scores_tensor(il);
    require(scores != nullptr, "scores tensor missing");

    const uint32_t n_head = model.hparams.n_head_mem(il);
    std::vector<float> attn_colsum(n_head * chunk_len);
    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t pos = 0; pos < chunk_len; ++pos) {
            attn_colsum[ih * chunk_len + pos] = static_cast<float>(ih * 100 + pos);
        }
    }

    kv.h2o_init_chunk_scores(il, 0, chunk_len, attn_colsum.data());

    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t pos = 0; pos < chunk_len; ++pos) {
            const uint8_t * base = static_cast<const uint8_t *>(scores->data);
            const ggml_bf16_t * score_ptr = reinterpret_cast<const ggml_bf16_t *>(
                    base + ih * scores->nb[1] + pos * scores->nb[0]);
            const ggml_bf16_t expected = ggml_fp32_to_bf16(attn_colsum[ih * chunk_len + pos]);
            require(score_ptr->bits == expected.bits, "score initialization mismatch");
        }
    }
}

void run_block_diagonal_mask_test(llama_model & model, int32_t il) {
    GGML_UNUSED(il);
    const uint32_t n_tokens = 8;
    const uint32_t chunk_size = 4;
    const uint32_t kv_size = 64;
    const uint32_t h2o_local = 8;
    const uint32_t h2o_heavy = 8;

    llama_kv_cache kv(
            model,
            GGML_TYPE_F16,
            GGML_TYPE_F16,
            true,
            false,
            true,
            kv_size,
            1,
            1,
            0,
            LLAMA_SWA_TYPE_H2O,
            nullptr,
            nullptr,
            h2o_local,
            h2o_heavy);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = 1;
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    const bool ok = balloc.init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), 1, true);
    require(ok, "batch init failed");

    auto mctx = kv.init_batch(balloc, n_tokens, true);
    require(mctx != nullptr, "kv init_batch failed");
    require(mctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS, "kv init_batch status not success");
    require(mctx->apply(), "kv apply failed");

    const llama_ubatch & ubatch = mctx->get_ubatch();
    auto * kv_ctx = dynamic_cast<llama_kv_cache_context *>(mctx.get());
    require(kv_ctx != nullptr, "kv context cast failed");

    const uint32_t n_kv = kv_ctx->get_n_kv();

    std::vector<uint8_t> mem(1024 * 1024);
    ggml_init_params ip = {
        /*.mem_size   =*/ mem.size(),
        /*.mem_buffer =*/ mem.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    require(ctx != nullptr, "ggml_init failed");

    llama_hparams hparams = model.hparams;
    llama_cparams cparams{};
    cparams.n_ubatch = chunk_size;
    cparams.causal_attn = true;
    cparams.kv_unified = true;
    cparams.flash_attn = false;
    cparams.h2o_local_window = h2o_local;
    cparams.h2o_heavy_budget = h2o_heavy;

    llm_graph_input_attn_kv inp(hparams, cparams, kv_ctx);
    inp.self_k_idxs = kv_ctx->build_input_k_idxs(ctx, ubatch);
    inp.self_v_idxs = kv_ctx->build_input_v_idxs(ctx, ubatch);
    inp.self_kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tokens, 1, 1);
    ggml_set_input(inp.self_kq_mask);
    inp.self_kq_mask_cnv = inp.self_kq_mask;

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to init CPU backend");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buf != nullptr, "failed to allocate backend buffer");

    inp.set_input(&ubatch);

    const int64_t * k_idxs = (const int64_t *) inp.self_k_idxs->data;
    const float * mask = (const float *) inp.self_kq_mask->data;

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const uint32_t chunk_start = (i / chunk_size) * chunk_size;
        for (uint32_t j = 0; j < n_tokens; ++j) {
            const int64_t kv_idx = k_idxs[j];
            const float val = mask[i * n_kv + kv_idx];
            const bool masked = std::isinf(val) && val < 0;
            const bool should_mask = j < chunk_start || j > i;
            if (should_mask) {
                require(masked, "cross-chunk attention not masked");
            } else {
                require(!masked, "in-chunk attention unexpectedly masked");
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    llama_batch_free(batch);
}
}

int main(int argc, char ** argv) {
    llama_backend_init();

    log_step("colsum correctness");
    run_colsum_test();

    char * model_path = get_model_or_exit(argc, argv);

    log_step("loading model");
    auto mparams = llama_model_default_params();
    mparams.use_mmap = true;

    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "WARNING: failed to load model '%s', skipping H2O score tracking tests\n", model_path);
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const int32_t il = find_first_kv_layer(model->hparams);
    if (il < 0) {
        fprintf(stderr, "WARNING: model has no KV layers, skipping H2O score tracking tests\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    log_step("score init");
    run_score_init_test(*model, il);

    log_step("block-diagonal mask");
    run_block_diagonal_mask_test(*model, il);

    fprintf(stdout, "H2O score tracking checks passed\n");

    llama_backend_free();
    return EXIT_SUCCESS;
}
