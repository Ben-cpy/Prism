#include "ggml.h"
#include "llama-cpp.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama.h"
#include "get-model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "H2O memory selection test failed: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

void log_step(const char * message) {
    fprintf(stderr, "[H2O MEM] %s\n", message);
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
        fprintf(stderr, "[H2O MEM] model load progress: %.1f%%\n", progress * 100.0f);
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

std::unique_ptr<llama_kv_cache> make_kv_cache(
        const llama_model & model,
        uint32_t kv_size,
        uint32_t h2o_local,
        uint32_t h2o_heavy) {
    return std::make_unique<llama_kv_cache>(
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
}

const int32_t * get_mem_idx_head(const ggml_tensor * mem_idx, uint32_t head) {
    require(mem_idx != nullptr, "memory indices tensor missing");
    require(mem_idx->data != nullptr, "memory indices data missing");
    const uint8_t * base = static_cast<const uint8_t *>(mem_idx->data);
    return reinterpret_cast<const int32_t *>(base + head * mem_idx->nb[1]);
}

std::vector<int32_t> get_valid_memory_indices(const ggml_tensor * mem_idx, uint32_t head, uint32_t M) {
    const int32_t * head_data = get_mem_idx_head(mem_idx, head);
    std::vector<int32_t> indices;
    indices.reserve(M);
    for (uint32_t m = 0; m < M; ++m) {
        if (head_data[m] >= 0) {
            indices.push_back(head_data[m]);
        }
    }
    return indices;
}

std::set<int32_t> to_set(const std::vector<int32_t> & values) {
    return std::set<int32_t>(values.begin(), values.end());
}

bool has_token(const ggml_tensor * mem_idx, uint32_t head, uint32_t M, int32_t token) {
    const int32_t * head_data = get_mem_idx_head(mem_idx, head);
    for (uint32_t m = 0; m < M; ++m) {
        if (head_data[m] == token) {
            return true;
        }
    }
    return false;
}

int32_t find_token_slot(const ggml_tensor * mem_idx, uint32_t head, uint32_t M, int32_t token) {
    const int32_t * head_data = get_mem_idx_head(mem_idx, head);
    for (uint32_t m = 0; m < M; ++m) {
        if (head_data[m] == token) {
            return static_cast<int32_t>(m);
        }
    }
    return -1;
}

void check_sorted_indices(const ggml_tensor * mem_idx, uint32_t head, uint32_t M) {
    std::vector<int32_t> indices = get_valid_memory_indices(mem_idx, head, M);
    for (size_t i = 1; i < indices.size(); ++i) {
        require(indices[i] > indices[i - 1], "memory indices not strictly increasing");
    }
}

void check_indices_in_range(const ggml_tensor * mem_idx, uint32_t n_head, uint32_t M, uint32_t chunk_end) {
    for (uint32_t h = 0; h < n_head; ++h) {
        const int32_t * head_data = get_mem_idx_head(mem_idx, h);
        for (uint32_t m = 0; m < M; ++m) {
            const int32_t idx = head_data[m];
            if (idx >= 0) {
                require(idx >= 0, "memory index negative");
                require(static_cast<uint32_t>(idx) < chunk_end, "memory index out of bounds");
            }
        }
    }
}

void test_local_window(const llama_model & model, int32_t il) {
    log_step("local window correctness");

    const uint32_t kv_size = 64;
    const uint32_t h2o_local = 8;
    const uint32_t h2o_heavy = 4;
    const uint32_t chunk_len = 16;

    auto kv = make_kv_cache(model, kv_size, h2o_local, h2o_heavy);

    const uint32_t n_head = model.hparams.n_head_mem(il);
    std::vector<float> colsum(n_head * chunk_len, 0.1f);

    kv->h2o_init_chunk_scores(il, 0, chunk_len, colsum.data());
    kv->h2o_build_memory_set(il, chunk_len);

    const ggml_tensor * mem_idx = kv->h2o_get_memory_indices_tensor(il);
    const uint32_t M = kv->h2o_get_memory_size();
    const uint32_t local_start = (chunk_len > h2o_local) ? (chunk_len - h2o_local) : 0;

    for (uint32_t h = 0; h < n_head; ++h) {
        std::set<int32_t> memory = to_set(get_valid_memory_indices(mem_idx, h, M));
        for (uint32_t pos = local_start; pos < chunk_len; ++pos) {
            require(memory.count(static_cast<int32_t>(pos)) == 1, "local window token missing");
        }
    }
}

void test_heavy_hitters_and_heads(const llama_model & model, int32_t il) {
    log_step("heavy hitter selection and per-head independence");

    const uint32_t kv_size = 64;
    const uint32_t h2o_local = 8;
    const uint32_t h2o_heavy = 4;
    const uint32_t chunk_len = 16;

    auto kv = make_kv_cache(model, kv_size, h2o_local, h2o_heavy);

    const uint32_t n_head = model.hparams.n_head_mem(il);
    std::vector<float> colsum(n_head * chunk_len, 0.0f);

    for (uint32_t h = 0; h < n_head; ++h) {
        for (uint32_t pos = 0; pos < chunk_len; ++pos) {
            const float value = (h % 2 == 0)
                    ? static_cast<float>(pos)
                    : static_cast<float>(chunk_len - pos);
            colsum[h * chunk_len + pos] = value;
        }
    }

    kv->h2o_init_chunk_scores(il, 0, chunk_len, colsum.data());
    kv->h2o_build_memory_set(il, chunk_len);

    const ggml_tensor * mem_idx = kv->h2o_get_memory_indices_tensor(il);
    const uint32_t M = kv->h2o_get_memory_size();

    const uint32_t local_start = chunk_len - h2o_local;
    std::set<int32_t> local_tokens;
    for (uint32_t pos = local_start; pos < chunk_len; ++pos) {
        local_tokens.insert(static_cast<int32_t>(pos));
    }

    // heavy hitter check for head 0
    {
        std::vector<std::pair<int32_t, float>> candidates;
        for (uint32_t pos = 0; pos < local_start; ++pos) {
            candidates.push_back({static_cast<int32_t>(pos), colsum[pos]});
        }
        std::sort(candidates.begin(), candidates.end(),
                [](const auto & a, const auto & b) { return a.second > b.second; });
        std::set<int32_t> expected_heavy;
        for (uint32_t i = 0; i < h2o_heavy; ++i) {
            expected_heavy.insert(candidates[i].first);
        }

        std::vector<int32_t> memory = get_valid_memory_indices(mem_idx, 0, M);
        std::set<int32_t> actual_heavy;
        for (int32_t token : memory) {
            if (local_tokens.count(token) == 0) {
                actual_heavy.insert(token);
            }
        }

        require(actual_heavy == expected_heavy, "heavy hitter set mismatch");
    }

    // per-head independence check
    if (n_head >= 2) {
        std::set<int32_t> head0 = to_set(get_valid_memory_indices(mem_idx, 0, M));
        std::set<int32_t> head1 = to_set(get_valid_memory_indices(mem_idx, 1, M));
        require(head0 != head1, "per-head memory sets are identical");
    }

    // sorted indices for all heads
    for (uint32_t h = 0; h < n_head; ++h) {
        check_sorted_indices(mem_idx, h, M);
    }
}

void test_small_sequence(const llama_model & model, int32_t il) {
    log_step("edge case: N < L");

    const uint32_t kv_size = 64;
    const uint32_t h2o_local = 8;
    const uint32_t h2o_heavy = 4;
    const uint32_t chunk_len = 4;

    auto kv = make_kv_cache(model, kv_size, h2o_local, h2o_heavy);

    const uint32_t n_head = model.hparams.n_head_mem(il);
    std::vector<float> colsum(n_head * chunk_len, 0.5f);

    kv->h2o_init_chunk_scores(il, 0, chunk_len, colsum.data());
    kv->h2o_build_memory_set(il, chunk_len);

    const ggml_tensor * mem_idx = kv->h2o_get_memory_indices_tensor(il);
    const uint32_t M = kv->h2o_get_memory_size();

    for (uint32_t h = 0; h < n_head; ++h) {
        std::vector<int32_t> memory = get_valid_memory_indices(mem_idx, h, M);
        require(memory.size() == chunk_len, "memory size does not match short sequence length");
        for (uint32_t pos = 0; pos < chunk_len; ++pos) {
            require(memory[pos] == static_cast<int32_t>(pos), "short sequence memory content mismatch");
        }
    }
}

void test_cross_chunk_propagation(const llama_model & model, int32_t il) {
    log_step("cross-chunk propagation");

    const uint32_t kv_size = 128;
    const uint32_t h2o_local = 8;
    const uint32_t h2o_heavy = 4;
    const uint32_t chunk_len = 16;
    const int32_t important_token = 3;

    auto kv = make_kv_cache(model, kv_size, h2o_local, h2o_heavy);

    const uint32_t n_head = model.hparams.n_head_mem(il);
    const uint32_t M = kv->h2o_get_memory_size();

    std::vector<float> chunk0_scores(n_head * chunk_len, 0.1f);
    for (uint32_t h = 0; h < n_head; ++h) {
        chunk0_scores[h * chunk_len + important_token] = 5.0f;
    }

    kv->h2o_init_chunk_scores(il, 0, chunk_len, chunk0_scores.data());
    kv->h2o_build_memory_set(il, chunk_len);
    kv->h2o_set_memory_initialized(true);

    const ggml_tensor * mem_idx = kv->h2o_get_memory_indices_tensor(il);
    require(kv->h2o_is_memory_initialized(), "memory not initialized after first chunk");
    require(has_token(mem_idx, 0, M, important_token), "important token missing after first chunk");
    check_indices_in_range(mem_idx, n_head, M, chunk_len);

    kv->h2o_next_chunk(chunk_len);

    for (int chunk = 1; chunk <= 2; ++chunk) {
        mem_idx = kv->h2o_get_memory_indices_tensor(il);
        std::vector<float> inter_colsum(n_head * M, 0.05f);
        for (uint32_t h = 0; h < n_head; ++h) {
            const int32_t slot = find_token_slot(mem_idx, h, M, important_token);
            require(slot >= 0, "important token missing in memory slots");
            inter_colsum[h * M + static_cast<uint32_t>(slot)] = 1.0f;
        }
        kv->h2o_accumulate_memory_scores(il, inter_colsum.data());

        std::vector<float> chunk_scores(n_head * chunk_len, 0.1f);
        const uint32_t chunk_start = static_cast<uint32_t>(chunk) * chunk_len;
        const uint32_t chunk_end = chunk_start + chunk_len;

        kv->h2o_init_chunk_scores(il, chunk_start, chunk_len, chunk_scores.data());
        kv->h2o_build_memory_set(il, chunk_end);

        mem_idx = kv->h2o_get_memory_indices_tensor(il);
        require(has_token(mem_idx, 0, M, important_token), "important token not propagated to next chunk");
        check_indices_in_range(mem_idx, n_head, M, chunk_end);

        kv->h2o_next_chunk(chunk_len);
    }
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

    log_step("loading model");
    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "WARNING: failed to load model '%s', skipping H2O memory selection test\n", model_path);
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    log_step("model loaded");
    const llama_hparams & hparams = model->hparams;
    const int32_t il = find_first_kv_layer(hparams);
    if (il < 0) {
        fprintf(stderr, "WARNING: model has no KV layers, skipping H2O memory selection test\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    fprintf(stderr, "[H2O MEM] first kv layer = %d\n", il);
    std::fflush(stderr);

    test_local_window(*model, il);
    test_heavy_hitters_and_heads(*model, il);
    test_small_sequence(*model, il);
    test_cross_chunk_propagation(*model, il);

    log_step("test complete");
    fprintf(stdout, "H2O memory selection checks passed\n");

    llama_backend_free();
    return EXIT_SUCCESS;
}
