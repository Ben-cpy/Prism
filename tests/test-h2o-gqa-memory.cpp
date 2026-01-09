#include "ggml.h"
#include "llama-cpp.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama.h"
#include "get-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void require(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "H2O GQA memory test failed: %s\n", msg);
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

bool has_token(const ggml_tensor * mem_idx, uint32_t head, uint32_t M, int32_t token) {
    const uint8_t * base = static_cast<const uint8_t *>(mem_idx->data);
    const int32_t * head_data = reinterpret_cast<const int32_t *>(base + head * mem_idx->nb[1]);
    for (uint32_t m = 0; m < M; ++m) {
        if (head_data[m] == token) {
            return true;
        }
    }
    return false;
}

void reduce_colsum_to_kv_heads(
        const float * src, uint32_t n_cols, uint32_t n_head, uint32_t n_head_kv, float * dst) {
    if (n_head_kv == n_head) {
        std::memcpy(dst, src, n_head * n_cols * sizeof(float));
        return;
    }
    require(n_head_kv > 0, "n_head_kv == 0");
    require(n_head % n_head_kv == 0, "n_head not divisible by n_head_kv");
    const uint32_t n_gqa = n_head / n_head_kv;
    std::memset(dst, 0, n_head_kv * n_cols * sizeof(float));
    for (uint32_t kv = 0; kv < n_head_kv; ++kv) {
        float * dst_head = dst + kv * n_cols;
        for (uint32_t g = 0; g < n_gqa; ++g) {
            const uint32_t qh = kv * n_gqa + g;
            const float * src_head = src + qh * n_cols;
            for (uint32_t c = 0; c < n_cols; ++c) {
                dst_head[c] += src_head[c];
            }
        }
    }
}
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "WARNING: failed to load model '%s', skipping\n", model_path);
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const llama_hparams & hparams = model->hparams;
    const int32_t il = find_first_kv_layer(hparams);
    if (il < 0) {
        fprintf(stderr, "WARNING: no KV layers, skipping\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const uint32_t n_head = hparams.n_head(il);
    const uint32_t n_head_kv = hparams.n_head_kv(il);
    if (n_head_kv == n_head) {
        fprintf(stderr, "WARNING: model is MHA (n_head_kv == n_head), skipping GQA test\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }
    require(n_head_kv > 0, "n_head_kv == 0");
    require(n_head % n_head_kv == 0, "invalid GQA ratio");

    const uint32_t kv_size = 32;
    const uint32_t chunk_len = 8;
    const uint32_t h2o_local = 2;
    const uint32_t h2o_heavy = 2;

    llama_kv_cache kv(
        *model, GGML_TYPE_F16, GGML_TYPE_F16, true, false, true,
        kv_size, 1, 1, 0, LLAMA_SWA_TYPE_H2O, nullptr, nullptr, h2o_local, h2o_heavy);

    std::vector<float> q_colsum(n_head * chunk_len, 0.0f);
    const uint32_t n_gqa = n_head / n_head_kv;
    for (uint32_t kvh = 0; kvh < n_head_kv; ++kvh) {
        const uint32_t hot_token = (kvh * 2) % (chunk_len - h2o_local);
        for (uint32_t g = 0; g < n_gqa; ++g) {
            const uint32_t qh = kvh * n_gqa + g;
            for (uint32_t pos = 0; pos < chunk_len; ++pos) {
                q_colsum[qh * chunk_len + pos] = (pos == hot_token) ? 10.0f : 0.1f;
            }
        }
    }

    std::vector<float> kv_colsum(n_head_kv * chunk_len, 0.0f);
    reduce_colsum_to_kv_heads(q_colsum.data(), chunk_len, n_head, n_head_kv, kv_colsum.data());

    kv.h2o_init_chunk_scores(il, 0, chunk_len, kv_colsum.data());
    kv.h2o_build_memory_set(il, chunk_len);

    const ggml_tensor * mem_idx = kv.h2o_get_memory_indices_tensor(il);
    require(mem_idx != nullptr, "mem_idx missing");
    require(mem_idx->ne[1] == static_cast<int64_t>(n_head_kv), "mem_idx head count mismatch");

    const uint32_t M = kv.h2o_get_memory_size();
    const uint32_t local_start = chunk_len - h2o_local;

    for (uint32_t kvh = 0; kvh < n_head_kv; ++kvh) {
        for (uint32_t pos = local_start; pos < chunk_len; ++pos) {
            require(has_token(mem_idx, kvh, M, static_cast<int32_t>(pos)), "local token missing");
        }
        const int32_t hot_token = static_cast<int32_t>((kvh * 2) % (chunk_len - h2o_local));
        require(has_token(mem_idx, kvh, M, hot_token), "expected heavy token missing");
    }

    llama_backend_free();
    return EXIT_SUCCESS;
}
