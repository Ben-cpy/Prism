#include "ggml.h"
#include "llama-cpp.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama.h"
#include "get-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "H2O KV cache test failed: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

void log_step(const char * message) {
    fprintf(stderr, "[H2O KV] %s\n", message);
    std::fflush(stderr);
}

struct load_progress_ctx {
    float last_reported = -1.0f;
};

bool log_load_progress(float progress, void * user_data) {
    auto * ctx = static_cast<load_progress_ctx *>(user_data);
    if (!ctx) {
        return true;
    }

    if (progress >= 1.0f || progress - ctx->last_reported >= 0.05f) {
        fprintf(stderr, "[H2O KV] model load progress: %.1f%%\n", progress * 100.0f);
        std::fflush(stderr);
        ctx->last_reported = progress;
    }

    return true;
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

    llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
        fprintf(stderr, "[LLAMA %d] %s", static_cast<int>(level), text);
        std::fflush(stderr);
    }, nullptr);

    log_step("init backend");
    llama_backend_init();

    auto mparams = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        devs[0] = cpu;
        mparams.devices = devs;
    }

    load_progress_ctx progress_ctx;
    mparams.progress_callback = log_load_progress;
    mparams.progress_callback_user_data = &progress_ctx;
    mparams.use_mmap = true;
    log_step("using mmap=1");

    log_step("loading model");
    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "WARNING: failed to load model '%s', skipping H2O KV cache test\n", model_path);
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    log_step("model loaded");
    const llama_hparams & hparams = model->hparams;
    const int32_t il = find_first_kv_layer(hparams);
    if (il < 0) {
        fprintf(stderr, "WARNING: model has no KV layers, skipping H2O KV cache test\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    fprintf(stderr, "[H2O KV] first kv layer = %d\n", il);
    std::fflush(stderr);

    const uint32_t kv_size = 128;
    const uint32_t h2o_local = 8;
    const uint32_t h2o_heavy = 8;
    const uint32_t chunk_len = 16;

    log_step("constructing kv cache");
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

    log_step("kv cache constructed");
    const ggml_tensor * scores = kv.h2o_get_scores_tensor(il);
    const ggml_tensor * mem_idx = kv.h2o_get_memory_indices_tensor(il);
    const uint32_t n_head = hparams.n_head(il);
    const uint32_t memory_size = kv.h2o_get_memory_size();

    fprintf(stderr, "[H2O KV] n_head=%u kv_size=%u memory_size=%u\n", n_head, kv_size, memory_size);
    fprintf(stderr, "[H2O KV] scores=%p mem_idx=%p\n", (const void *) scores, (const void *) mem_idx);
    if (scores) {
        fprintf(stderr, "[H2O KV] scores ne0=%ld ne1=%ld nb0=%ld nb1=%ld data=%p\n",
                scores->ne[0], scores->ne[1], (long) scores->nb[0], (long) scores->nb[1], scores->data);
    }
    if (mem_idx) {
        fprintf(stderr, "[H2O KV] mem_idx ne0=%ld ne1=%ld nb0=%ld nb1=%ld data=%p\n",
                mem_idx->ne[0], mem_idx->ne[1], (long) mem_idx->nb[0], (long) mem_idx->nb[1], mem_idx->data);
    }
    std::fflush(stderr);

    require(scores != nullptr, "scores tensor missing");
    require(mem_idx != nullptr, "memory indices tensor missing");
    require(scores->data != nullptr, "scores tensor data missing");
    require(mem_idx->data != nullptr, "memory indices tensor data missing");
    require(ggml_is_contiguous(scores), "scores tensor not contiguous");
    require(ggml_is_contiguous(mem_idx), "memory indices tensor not contiguous");
    require(scores->type == GGML_TYPE_BF16, "scores tensor type mismatch");
    require(mem_idx->type == GGML_TYPE_I32, "memory indices tensor type mismatch");
    require(scores->ne[0] == static_cast<int64_t>(kv_size), "scores kv_size mismatch");
    require(scores->ne[1] == static_cast<int64_t>(n_head), "scores n_head mismatch");
    require(scores->nb[0] == sizeof(ggml_bf16_t), "scores stride mismatch (nb0)");
    require(scores->nb[1] == scores->ne[0] * scores->nb[0], "scores stride mismatch (nb1)");
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

    log_step("init chunk scores");
    kv.h2o_init_chunk_scores(il, 0, chunk_len, attn_colsum.data());
    log_step("init chunk scores done");

    const ggml_bf16_t * score_data = static_cast<const ggml_bf16_t *>(scores->data);
    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t pos = 0; pos < chunk_len; ++pos) {
            const uint8_t * base = static_cast<const uint8_t *>(scores->data);
            const ggml_bf16_t * score_ptr = reinterpret_cast<const ggml_bf16_t *>(
                    base + ih * scores->nb[1] + pos * scores->nb[0]);
            const ggml_bf16_t stored = *score_ptr;
            const ggml_bf16_t expected = ggml_fp32_to_bf16(attn_colsum[ih * chunk_len + pos]);
            require(stored.bits == expected.bits, "score initialization mismatch");
        }
    }

    log_step("build memory set");
    kv.h2o_build_memory_set(il, chunk_len);
    log_step("build memory set done");

    const int32_t * mem_idx_data = static_cast<const int32_t *>(mem_idx->data);
    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t m = 0; m < memory_size; ++m) {
            const int32_t idx = mem_idx_data[ih * memory_size + m];
            require(idx == static_cast<int32_t>(m), "memory indices not sorted or incomplete");
        }
    }

    std::vector<float> inter_colsum(n_head * memory_size, 1.0f);
    log_step("accumulate memory scores");
    kv.h2o_accumulate_memory_scores(il, inter_colsum.data());
    log_step("accumulate memory scores done");

    for (uint32_t ih = 0; ih < n_head; ++ih) {
        const ggml_bf16_t stored = score_data[ih * kv_size + 0];
        const float current = ggml_bf16_to_fp32(ggml_fp32_to_bf16(attn_colsum[ih * chunk_len + 0]));
        const ggml_bf16_t expected = ggml_fp32_to_bf16(current + 1.0f);
        require(stored.bits == expected.bits, "score accumulation mismatch");
    }

    log_step("next chunk");
    kv.h2o_next_chunk(chunk_len);
    log_step("next chunk done");
    require(kv.h2o_get_chunk_idx() == 1, "chunk index did not increment");
    require(kv.h2o_get_total_tokens() == chunk_len, "total tokens did not update");

    log_step("test complete");
    fprintf(stdout, "H2O KV cache checks passed\n");

    llama_backend_free();
    return EXIT_SUCCESS;
}
