#include "ggml-backend.h"
#include "llama.h"
#include "llama-batch.h"
#include "llama-cpp.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "get-model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "H2O edge cases test failed: %s\n", message);
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

struct h2o_test_ctx {
    llama_context_ptr ctx;
    llama_kv_cache * kv = nullptr;
};

llama_context_params make_context_params(uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
        uint32_t h2o_local, uint32_t h2o_heavy) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.n_seq_max = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.h2o_local_window = h2o_local;
    params.h2o_heavy_budget = h2o_heavy;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.embeddings = false;
    params.offload_kqv = false;
    params.kv_unified = true;
    return params;
}

h2o_test_ctx make_h2o_context(llama_model_ptr & model, uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
        uint32_t h2o_local, uint32_t h2o_heavy, bool require_kv) {
    llama_context_params params = make_context_params(n_ctx, n_batch, n_ubatch, h2o_local, h2o_heavy);
    llama_context_ptr ctx(llama_init_from_model(model.get(), params));
    require(ctx != nullptr, "failed to create llama_context");

    llama_memory_t mem = llama_get_memory(ctx.get());
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (require_kv) {
        require(kv != nullptr, "context memory is not llama_kv_cache");
    }

    return h2o_test_ctx{ std::move(ctx), kv };
}

llama_batch make_batch(const std::vector<llama_token> & tokens, llama_pos base_pos, bool all_logits) {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(base_pos + static_cast<llama_pos>(i));
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        if (all_logits) {
            batch.logits[i] = 1;
        } else {
            batch.logits[i] = (i + 1 == tokens.size()) ? 1 : 0;
        }
    }
    return batch;
}

struct logits_block {
    std::vector<float> data;
    int32_t n_outputs = 0;
    int32_t n_vocab = 0;
};

void dump_mem_idx(const ggml_tensor * mem_idx, uint32_t head, uint32_t M, const char * label) {
    if (!mem_idx) {
        fprintf(stderr, "%s: mem_idx missing\n", label);
        return;
    }
    const uint8_t * base = static_cast<const uint8_t *>(mem_idx->data);
    const int32_t * head_data = reinterpret_cast<const int32_t *>(base + head * mem_idx->nb[1]);
    fprintf(stderr, "%s head %u: [", label, head);
    for (uint32_t m = 0; m < M; ++m) {
        fprintf(stderr, "%d%s", head_data[m], (m + 1 < M) ? ", " : "");
    }
    fprintf(stderr, "]\n");
}

logits_block decode_and_get_logits(llama_context * ctx, const std::vector<llama_token> & tokens,
        llama_pos base_pos, bool all_logits, int32_t n_vocab, const char * label) {
    llama_batch batch = make_batch(tokens, base_pos, all_logits);
    const int result = llama_decode(ctx, batch);
    require(result == 0, label);

    const int32_t n_outputs = all_logits ? static_cast<int32_t>(tokens.size()) : 1;
    logits_block out;
    out.n_outputs = n_outputs;
    out.n_vocab = n_vocab;
    out.data.resize(static_cast<size_t>(n_outputs) * static_cast<size_t>(n_vocab));

    for (int32_t i = 0; i < n_outputs; ++i) {
        const float * logits = llama_get_logits_ith(ctx, i);
        require(logits != nullptr, "missing logits");
        std::memcpy(out.data.data() + static_cast<size_t>(i) * static_cast<size_t>(n_vocab),
                logits, static_cast<size_t>(n_vocab) * sizeof(float));
    }

    llama_batch_free(batch);
    return out;
}

void compare_logits_token(const logits_block & ref, int32_t ref_idx,
        const logits_block & other, int32_t other_idx, float tol, const char * label) {
    require(ref.n_vocab == other.n_vocab, "vocab size mismatch");
    require(ref_idx < ref.n_outputs && other_idx < other.n_outputs, "output index out of range");

    const float * a = ref.data.data() + static_cast<size_t>(ref_idx) * static_cast<size_t>(ref.n_vocab);
    const float * b = other.data.data() + static_cast<size_t>(other_idx) * static_cast<size_t>(other.n_vocab);

    float max_diff = 0.0f;
    for (int32_t i = 0; i < ref.n_vocab; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    if (max_diff > tol) {
        fprintf(stderr, "Max logit diff = %.6f (tol=%.6f)\n", max_diff, tol);
        require(false, label);
    }
}

std::vector<llama_token> make_tokens(uint32_t n_tokens, llama_token base) {
    std::vector<llama_token> tokens(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        tokens[i] = static_cast<llama_token>(base + static_cast<llama_token>(i));
    }
    return tokens;
}

void test_cross_batch_continuity(llama_model_ptr & model, int32_t n_vocab) {
    fprintf(stderr, "\n=== Test 1: Cross-Batch Continuity ===\n");
    fprintf(stderr, "Scaled down example: -b=16, -ub=4 (16 tokens => 4 chunks; split into 8+8)\n");
    const llama_hparams & hparams = model->hparams;
    const int32_t il = find_first_kv_layer(hparams);
    require(il >= 0, "no KV layers found");

    const uint32_t chunk_size = 4;
    const uint32_t n_tokens_total = 16;
    const uint32_t n_ctx = 128;
    const uint32_t h2o_local = 2;
    const uint32_t h2o_heavy = 2;

    auto full_ctx = make_h2o_context(model, n_ctx, n_tokens_total, chunk_size, h2o_local, h2o_heavy, true);
    auto tokens = make_tokens(n_tokens_total, 100);

    logits_block full_logits = decode_and_get_logits(
            full_ctx.ctx.get(), tokens, 0, true, n_vocab, "full batch decode failed");

    require(full_ctx.kv != nullptr, "kv cache missing in full batch context");
    require(full_ctx.kv->h2o_is_memory_initialized(), "H2O memory not initialized after full batch");
    require(full_ctx.kv->h2o_get_total_tokens() == n_tokens_total, "total tokens mismatch after full batch");
    require(full_ctx.kv->h2o_get_chunk_idx() == 4, "chunk index mismatch after full batch");
    const ggml_tensor * mem_idx = full_ctx.kv->h2o_get_memory_indices_tensor(il);
    require(mem_idx != nullptr, "memory indices tensor missing");
    require(mem_idx->ne[1] == static_cast<int64_t>(hparams.n_head_kv(il)), "memory head count mismatch");
    if (std::getenv("H2O_DEBUG_MEMIDX")) {
        dump_mem_idx(mem_idx, 0, full_ctx.kv->h2o_get_memory_size(), "full mem_idx");
    }

    auto split_ctx = make_h2o_context(model, n_ctx, n_tokens_total, chunk_size, h2o_local, h2o_heavy, true);

    std::vector<llama_token> tokens1(tokens.begin(), tokens.begin() + 8);
    std::vector<llama_token> tokens2(tokens.begin() + 8, tokens.end());

    decode_and_get_logits(split_ctx.ctx.get(), tokens1, 0, true, n_vocab, "batch1 decode failed");
    require(split_ctx.kv != nullptr, "kv cache missing in split batch context");
    require(split_ctx.kv->h2o_is_memory_initialized(), "H2O memory not initialized after batch1");
    require(split_ctx.kv->h2o_get_total_tokens() == 8, "total tokens mismatch after batch1");
    require(split_ctx.kv->h2o_get_chunk_idx() == 2, "chunk index mismatch after batch1");
    if (std::getenv("H2O_DEBUG_MEMIDX")) {
        const ggml_tensor * mem_idx_split = split_ctx.kv->h2o_get_memory_indices_tensor(il);
        dump_mem_idx(mem_idx_split, 0, split_ctx.kv->h2o_get_memory_size(), "split mem_idx after batch1");
    }

    logits_block split_logits = decode_and_get_logits(
            split_ctx.ctx.get(), tokens2, 8, true, n_vocab, "batch2 decode failed");

    require(split_ctx.kv->h2o_get_total_tokens() == n_tokens_total, "total tokens mismatch after batch2");
    require(split_ctx.kv->h2o_get_chunk_idx() == 4, "chunk index mismatch after batch2");
    if (std::getenv("H2O_DEBUG_MEMIDX")) {
        const ggml_tensor * mem_idx_split = split_ctx.kv->h2o_get_memory_indices_tensor(il);
        dump_mem_idx(mem_idx_split, 0, split_ctx.kv->h2o_get_memory_size(), "split mem_idx after batch2");
    }

    const float tol = 1e-4f;
    for (int32_t i = 0; i < 8; ++i) {
        compare_logits_token(full_logits, 8 + i, split_logits, i, tol,
                "cross-batch logits mismatch");
    }

    fprintf(stderr, "✓ Cross-batch continuity test passed\n");
}

void test_short_sequence(llama_model_ptr & model, int32_t n_vocab) {
    fprintf(stderr, "\n=== Test 2: Short Sequence (< chunk_size) ===\n");
    fprintf(stderr, "Scaled down example: -b=8, -ub=8 (4 tokens => single chunk)\n");

    const uint32_t chunk_size = 8;
    const uint32_t n_tokens = 4;
    const uint32_t n_ctx = 128;
    const uint32_t h2o_local = 4;
    const uint32_t h2o_heavy = 4;

    auto h2o_ctx = make_h2o_context(model, n_ctx, chunk_size, chunk_size, h2o_local, h2o_heavy, true);
    auto base_ctx = make_h2o_context(model, n_ctx, chunk_size, chunk_size, 0, 0, false);

    auto tokens = make_tokens(n_tokens, 200);

    logits_block h2o_logits = decode_and_get_logits(
            h2o_ctx.ctx.get(), tokens, 0, true, n_vocab, "H2O short decode failed");
    logits_block base_logits = decode_and_get_logits(
            base_ctx.ctx.get(), tokens, 0, true, n_vocab, "baseline short decode failed");

    const float tol = 1e-4f;
    for (int32_t i = 0; i < static_cast<int32_t>(n_tokens); ++i) {
        compare_logits_token(h2o_logits, i, base_logits, i, tol,
                "short sequence logits mismatch");
    }

    fprintf(stderr, "✓ Short sequence test passed\n");
}

void test_non_uniform_chunks(llama_model_ptr & model, int32_t n_vocab) {
    fprintf(stderr, "\n=== Test 3: Non-Uniform Chunks ===\n");
    fprintf(stderr, "Scaled down example: -b=10, -ub=6 (chunks: 6 + 4)\n");

    const uint32_t chunk_size = 6;
    const uint32_t n_tokens = 10;
    const uint32_t n_ctx = 128;
    const uint32_t h2o_local = 4;
    const uint32_t h2o_heavy = 4;

    auto h2o_ctx = make_h2o_context(model, n_ctx, n_tokens, chunk_size, h2o_local, h2o_heavy, true);
    auto base_ctx = make_h2o_context(model, n_ctx, n_tokens, chunk_size, 0, 0, false);

    auto tokens = make_tokens(n_tokens, 300);

    logits_block h2o_logits = decode_and_get_logits(
            h2o_ctx.ctx.get(), tokens, 0, true, n_vocab, "H2O non-uniform decode failed");
    logits_block base_logits = decode_and_get_logits(
            base_ctx.ctx.get(), tokens, 0, true, n_vocab, "baseline non-uniform decode failed");

    require(h2o_ctx.kv != nullptr, "kv cache missing in non-uniform context");
    require(h2o_ctx.kv->h2o_get_total_tokens() == n_tokens, "total tokens mismatch after non-uniform decode");
    require(h2o_ctx.kv->h2o_get_chunk_idx() == 2, "chunk index mismatch after non-uniform decode");

    const float tol = 1e-4f;
    for (int32_t i = 0; i < static_cast<int32_t>(n_tokens); ++i) {
        compare_logits_token(h2o_logits, i, base_logits, i, tol,
                "non-uniform chunk logits mismatch");
    }

    fprintf(stderr, "✓ Non-uniform chunks test passed\n");
}

void test_edge_cases(llama_model_ptr & model, int32_t n_vocab) {
    fprintf(stderr, "\n=== Test 4: Edge Cases ===\n");

    const uint32_t n_ctx = 64;
    const uint32_t h2o_local = 4;
    const uint32_t h2o_heavy = 4;

    auto ctx = make_h2o_context(model, n_ctx, 1, 1, h2o_local, h2o_heavy, true);

    std::vector<llama_token> tokens = { 1 };
    logits_block logits = decode_and_get_logits(
            ctx.ctx.get(), tokens, 0, true, n_vocab, "single token decode failed");
    require(!logits.data.empty(), "single token logits missing");

    fprintf(stderr, "✓ Edge cases test passed\n");
}

} // namespace

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
    mparams.n_gpu_layers = 0;
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;

    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "WARNING: failed to load model '%s', skipping H2O edge cases test\n", model_path);
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const llama_hparams & hparams = model->hparams;
    const int32_t il = find_first_kv_layer(hparams);
    if (il < 0) {
        fprintf(stderr, "WARNING: model has no KV layers, skipping H2O edge cases test\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 0, "invalid vocab size");

    fprintf(stderr, "Testing H2O Edge Cases\n");
    fprintf(stderr, "======================\n");

    test_cross_batch_continuity(model, n_vocab);
    test_short_sequence(model, n_vocab);
    test_non_uniform_chunks(model, n_vocab);
    test_edge_cases(model, n_vocab);

    fprintf(stderr, "\n======================\n");
    fprintf(stderr, "All H2O edge cases tests passed! ✓\n");

    llama_backend_free();
    return EXIT_SUCCESS;
}
