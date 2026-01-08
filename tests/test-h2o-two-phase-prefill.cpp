#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-cpp.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "get-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "H2O two-phase prefill test failed: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

int32_t find_first_kv_layer(const llama_hparams & hparams) {
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (hparams.has_kv(il)) {
            return static_cast<int32_t>(il);
        }
    }
    return -1;
}

void fill_tensor(ggml_tensor * tensor, const std::vector<float> & data) {
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        devs[0] = cpu;
        mparams.devices = devs;
    }
    mparams.use_mmap = true;

    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "WARNING: failed to load model '%s', skipping H2O two-phase prefill test\n", model_path);
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const llama_hparams & hparams = model->hparams;
    const int32_t il = find_first_kv_layer(hparams);
    if (il < 0) {
        fprintf(stderr, "WARNING: model has no KV layers, skipping H2O two-phase prefill test\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const uint32_t n_tokens = 4;
    const uint32_t chunk_size = 2;
    const uint32_t kv_size = 32;
    const uint32_t h2o_local = 2;
    const uint32_t h2o_heavy = 2;

    llama_kv_cache kv(
            *model,
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

    llama_batch_allocr balloc(hparams.n_pos_per_embd());
    const bool ok = balloc.init(batch, model->vocab, nullptr, hparams.n_embd_inp(), 1, true);
    require(ok, "batch init failed");

    auto mctx_full = kv.init_batch(balloc, n_tokens, true);
    require(mctx_full != nullptr, "kv init_batch failed");
    require(mctx_full->get_status() == LLAMA_MEMORY_STATUS_SUCCESS, "kv init_batch status not success");
    require(mctx_full->apply(), "kv apply failed");

    const llama_ubatch & ubatch_full = mctx_full->get_ubatch();

    llama_cparams cparams{};
    cparams.n_ctx = n_tokens;
    cparams.n_ctx_seq = n_tokens;
    cparams.n_batch = n_tokens;
    cparams.n_ubatch = chunk_size;
    cparams.n_seq_max = 1;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    cparams.causal_attn = true;
    cparams.kv_unified = true;
    cparams.flash_attn = false;
    cparams.offload_kqv = true;
    cparams.h2o_local_window = h2o_local;
    cparams.h2o_heavy_budget = h2o_heavy;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf = true;
    cparams.warmup = false;

    h2o_prefill_cache cache;

    llm_graph_result res1(4096);
    llm_graph_params params1{};
    params1.arch = model->arch;
    params1.hparams = hparams;
    params1.cparams = cparams;
    params1.ubatch = ubatch_full;
    params1.gtype = LLM_GRAPH_TYPE_DEFAULT;
    params1.sched = nullptr;
    params1.backend_cpu = nullptr;
    params1.cvec = nullptr;
    params1.loras = nullptr;
    params1.mctx = mctx_full.get();
    params1.cross = nullptr;
    params1.samplers = {};
    params1.current_phase = 1;
    params1.h2o_prefill = &cache;
    params1.n_outputs = ubatch_full.n_tokens;
    params1.cb = nullptr;
    params1.res = &res1;

    llm_graph_context graph1(params1);
    auto * kv_ctx_full = dynamic_cast<llama_kv_cache_context *>(mctx_full.get());
    require(kv_ctx_full != nullptr, "kv context cast failed");

    llm_graph_input_attn_kv inp1(hparams, cparams, kv_ctx_full);
    inp1.self_k_idxs = kv_ctx_full->build_input_k_idxs(graph1.ctx0, ubatch_full);
    inp1.self_v_idxs = kv_ctx_full->build_input_v_idxs(graph1.ctx0, ubatch_full);
    inp1.self_kq_mask = ggml_new_tensor_4d(graph1.ctx0, GGML_TYPE_F32, kv_ctx_full->get_n_kv(), n_tokens, 1, 1);
    ggml_set_input(inp1.self_kq_mask);
    inp1.self_kq_mask_cnv = inp1.self_kq_mask;

    const int64_t n_head = hparams.n_head(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_head_v = hparams.n_embd_head_v;

    ggml_tensor * q1 = ggml_new_tensor_3d(graph1.ctx0, GGML_TYPE_F32, n_embd_head_k, n_head, n_tokens);
    ggml_tensor * k1 = ggml_new_tensor_3d(graph1.ctx0, GGML_TYPE_F32, n_embd_head_k, n_head, n_tokens);
    ggml_tensor * v1 = ggml_new_tensor_3d(graph1.ctx0, GGML_TYPE_F32, n_embd_head_v, n_head, n_tokens);

    ggml_set_input(q1);
    ggml_set_input(k1);
    ggml_set_input(v1);

    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(n_embd_head_k));
    ggml_tensor * out1 = graph1.build_attn_h2o(&inp1, nullptr, nullptr, q1, k1, v1, nullptr, nullptr, nullptr, kq_scale, il, nullptr);

    res1.set_outputs();

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to init CPU backend");

    ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors(graph1.ctx0, backend);
    require(buf1 != nullptr, "failed to allocate backend buffer");

    std::vector<float> q1_data(ggml_nelements(q1));
    std::vector<float> k1_data(ggml_nelements(k1));
    std::vector<float> v1_data(ggml_nelements(v1));
    for (size_t i = 0; i < q1_data.size(); ++i) {
        q1_data[i] = 0.01f * static_cast<float>(i % 17);
    }
    for (size_t i = 0; i < k1_data.size(); ++i) {
        k1_data[i] = 0.02f * static_cast<float>(i % 23);
    }
    for (size_t i = 0; i < v1_data.size(); ++i) {
        v1_data[i] = 0.03f * static_cast<float>(i % 29);
    }

    fill_tensor(q1, q1_data);
    fill_tensor(k1, k1_data);
    fill_tensor(v1, v1_data);

    inp1.set_input(&ubatch_full);

    ggml_status status = ggml_backend_graph_compute(backend, graph1.gf);
    require(status == GGML_STATUS_SUCCESS, "phase1 graph compute failed");

    std::vector<float> out1_data(ggml_nelements(out1));
    ggml_backend_tensor_get(out1, out1_data.data(), 0, out1_data.size() * sizeof(float));

    require(!res1.t_h2o_intra_o.empty(), "missing phase1 intra cache output");

    cache.intra_o.assign(hparams.n_layer, {});
    cache.intra_l.assign(hparams.n_layer, {});
    cache.base_pos = ubatch_full.pos ? ubatch_full.pos[0] : 0;

    for (const auto & [layer_id, t_intra_o] : res1.t_h2o_intra_o) {
        const auto it_l = res1.t_h2o_intra_l.find(layer_id);
        require(it_l != res1.t_h2o_intra_l.end(), "missing intra l for layer");

        const auto * t_intra_l = it_l->second;
        require(t_intra_l != nullptr, "missing intra l tensor");
        require(t_intra_o != nullptr, "missing intra o tensor");

        cache.n_tokens = static_cast<uint32_t>(t_intra_o->ne[1]);
        cache.n_head = static_cast<uint32_t>(t_intra_o->ne[2]);
        cache.n_embd_head_v = static_cast<uint32_t>(t_intra_o->ne[0]);
        cache.n_stream = static_cast<uint32_t>(t_intra_o->ne[3]);

        cache.intra_o[layer_id].resize(ggml_nelements(t_intra_o));
        ggml_backend_tensor_get(t_intra_o, cache.intra_o[layer_id].data(), 0, cache.intra_o[layer_id].size() * sizeof(float));

        cache.intra_l[layer_id].resize(ggml_nelements(t_intra_l));
        ggml_backend_tensor_get(t_intra_l, cache.intra_l[layer_id].data(), 0, cache.intra_l[layer_id].size() * sizeof(float));
    }

    cache.initialized = true;

    std::vector<float> dummy_scores(cache.n_head * chunk_size, 0.0f);
    kv.h2o_init_chunk_scores(il, 0, chunk_size, dummy_scores.data());
    kv.h2o_build_memory_set(il, chunk_size);

    auto mctx_chunk = kv.init_batch(balloc, chunk_size, true);
    require(mctx_chunk != nullptr, "kv init_batch (chunk) failed");
    require(mctx_chunk->get_status() == LLAMA_MEMORY_STATUS_SUCCESS, "kv init_batch (chunk) status not success");
    require(mctx_chunk->apply(), "kv apply (chunk) failed");

    const llama_ubatch & ubatch_chunk = mctx_chunk->get_ubatch();

    llm_graph_result res2(4096);
    llm_graph_params params2 = params1;
    params2.ubatch = ubatch_chunk;
    params2.mctx = mctx_chunk.get();
    params2.current_phase = 2;
    params2.res = &res2;

    llm_graph_context graph2(params2);
    auto * kv_ctx_chunk = dynamic_cast<llama_kv_cache_context *>(mctx_chunk.get());
    require(kv_ctx_chunk != nullptr, "kv context (chunk) cast failed");

    llm_graph_input_attn_kv inp2(hparams, cparams, kv_ctx_chunk);
    inp2.self_k_idxs = kv_ctx_chunk->build_input_k_idxs(graph2.ctx0, ubatch_chunk);
    inp2.self_v_idxs = kv_ctx_chunk->build_input_v_idxs(graph2.ctx0, ubatch_chunk);
    inp2.self_kq_mask = ggml_new_tensor_4d(graph2.ctx0, GGML_TYPE_F32, kv_ctx_chunk->get_n_kv(), chunk_size, 1, 1);
    ggml_set_input(inp2.self_kq_mask);
    inp2.self_kq_mask_cnv = inp2.self_kq_mask;

    ggml_tensor * q2 = ggml_new_tensor_3d(graph2.ctx0, GGML_TYPE_F32, n_embd_head_k, n_head, chunk_size);
    ggml_tensor * k2 = ggml_new_tensor_3d(graph2.ctx0, GGML_TYPE_F32, n_embd_head_k, n_head, chunk_size);
    ggml_tensor * v2 = ggml_new_tensor_3d(graph2.ctx0, GGML_TYPE_F32, n_embd_head_v, n_head, chunk_size);

    ggml_set_input(q2);
    ggml_set_input(k2);
    ggml_set_input(v2);

    ggml_tensor * out2 = graph2.build_attn_h2o(&inp2, nullptr, nullptr, q2, k2, v2, nullptr, nullptr, nullptr, kq_scale, il, nullptr);

    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(graph2.ctx0, backend);
    require(buf2 != nullptr, "failed to allocate phase2 buffer");

    const size_t qkv_chunk_elems_k = static_cast<size_t>(n_embd_head_k * n_head * chunk_size);
    const size_t qkv_chunk_elems_v = static_cast<size_t>(n_embd_head_v * n_head * chunk_size);

    std::vector<float> q2_data(q1_data.begin(), q1_data.begin() + qkv_chunk_elems_k);
    std::vector<float> k2_data(k1_data.begin(), k1_data.begin() + qkv_chunk_elems_k);
    std::vector<float> v2_data(v1_data.begin(), v1_data.begin() + qkv_chunk_elems_v);

    fill_tensor(q2, q2_data);
    fill_tensor(k2, k2_data);
    fill_tensor(v2, v2_data);

    inp2.set_input(&ubatch_chunk);
    res2.set_inputs(&ubatch_chunk);

    status = ggml_backend_graph_compute(backend, graph2.gf);
    require(status == GGML_STATUS_SUCCESS, "phase2 graph compute failed");

    std::vector<float> out2_data(ggml_nelements(out2));
    ggml_backend_tensor_get(out2, out2_data.data(), 0, out2_data.size() * sizeof(float));

    const size_t out_chunk_elems = static_cast<size_t>(n_embd_head_v * n_head * chunk_size);
    require(out2_data.size() == out_chunk_elems, "phase2 output size mismatch");

    for (size_t i = 0; i < out_chunk_elems; ++i) {
        const float diff = std::fabs(out1_data[i] - out2_data[i]);
        require(diff < 1e-3f, "phase2 output mismatch for chunk0");
    }

    for (auto & val : cache.intra_o[il]) {
        val *= 0.5f;
    }

    inp2.set_input(&ubatch_chunk);
    res2.set_inputs(&ubatch_chunk);
    status = ggml_backend_graph_compute(backend, graph2.gf);
    require(status == GGML_STATUS_SUCCESS, "phase2 graph compute after cache edit failed");

    std::vector<float> out2_mod(ggml_nelements(out2));
    ggml_backend_tensor_get(out2, out2_mod.data(), 0, out2_mod.size() * sizeof(float));

    bool changed = false;
    for (size_t i = 0; i < out_chunk_elems; ++i) {
        if (std::fabs(out2_mod[i] - out2_data[i]) > 1e-4f) {
            changed = true;
            break;
        }
    }
    require(changed, "phase2 output did not respond to cache change");

    ggml_backend_buffer_free(buf2);
    ggml_backend_buffer_free(buf1);
    ggml_backend_free(backend);
    llama_backend_free();
    llama_batch_free(batch);

    return EXIT_SUCCESS;
}
