#include "ggml.h"
#include "llama-cpp.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama.h"
#include "get-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "H2O KV cache test failed: %s\n", message);
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
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        ggml_backend_dev_t devs[2] = { cpu, nullptr };
        mparams.devices = devs;
    }

    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "WARNING: failed to load model '%s', skipping H2O KV cache test\n", model_path);
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const llama_hparams & hparams = model->hparams;
    const int32_t il = find_first_kv_layer(hparams);
    if (il < 0) {
        fprintf(stderr, "WARNING: model has no KV layers, skipping H2O KV cache test\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const uint32_t kv_size = 128;
    const uint32_t h2o_local = 8;
    const uint32_t h2o_heavy = 8;
    const uint32_t chunk_len = 16;

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

    const ggml_tensor * scores = kv.h2o_get_scores_tensor(il);
    const ggml_tensor * mem_idx = kv.h2o_get_memory_indices_tensor(il);
    const uint32_t n_head = hparams.n_head(il);
    const uint32_t memory_size = kv.h2o_get_memory_size();

    require(scores != nullptr, "scores tensor missing");
    require(mem_idx != nullptr, "memory indices tensor missing");
    require(ggml_is_contiguous(scores), "scores tensor not contiguous");
    require(ggml_is_contiguous(mem_idx), "memory indices tensor not contiguous");
    require(scores->ne[0] == static_cast<int64_t>(kv_size), "scores kv_size mismatch");
    require(scores->ne[1] == static_cast<int64_t>(n_head), "scores n_head mismatch");
    require(mem_idx->ne[0] == static_cast<int64_t>(memory_size), "memory size mismatch");
    require(mem_idx->ne[1] == static_cast<int64_t>(n_head), "memory n_head mismatch");
    require(mem_idx->nb[0] == sizeof(int32_t), "memory index stride mismatch (nb0)");
    require(mem_idx->nb[1] == mem_idx->ne[0] * sizeof(int32_t), "memory index stride mismatch (nb1)");

    std::vector<float> attn_colsum(n_head * chunk_len);
    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t pos = 0; pos < chunk_len; ++pos) {
            attn_colsum[ih * chunk_len + pos] = static_cast<float>(ih * 100 + pos);
        }
    }

    kv.h2o_init_chunk_scores(il, 0, chunk_len, attn_colsum.data());

    const ggml_bf16_t * score_data = static_cast<const ggml_bf16_t *>(scores->data);
    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t pos = 0; pos < chunk_len; ++pos) {
            const float stored = ggml_bf16_to_fp32(score_data[ih * kv_size + pos]);
            const float expected = attn_colsum[ih * chunk_len + pos];
            require(std::fabs(stored - expected) < 1e-3f, "score initialization mismatch");
        }
    }

    kv.h2o_build_memory_set(il, chunk_len);

    const int32_t * mem_idx_data = static_cast<const int32_t *>(mem_idx->data);
    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t m = 0; m < memory_size; ++m) {
            const int32_t idx = mem_idx_data[ih * memory_size + m];
            require(idx == static_cast<int32_t>(m), "memory indices not sorted or incomplete");
        }
    }

    std::vector<float> inter_colsum(n_head * memory_size, 1.0f);
    kv.h2o_accumulate_memory_scores(il, inter_colsum.data());

    for (uint32_t ih = 0; ih < n_head; ++ih) {
        const float stored = ggml_bf16_to_fp32(score_data[ih * kv_size + 0]);
        const float expected = attn_colsum[ih * chunk_len + 0] + 1.0f;
        require(std::fabs(stored - expected) < 1e-3f, "score accumulation mismatch");
    }

    kv.h2o_next_chunk(chunk_len);
    require(kv.h2o_get_chunk_idx() == 1, "chunk index did not increment");
    require(kv.h2o_get_total_tokens() == chunk_len, "total tokens did not update");

    fprintf(stdout, "H2O KV cache checks passed\n");

    llama_backend_free();
    return EXIT_SUCCESS;
}
