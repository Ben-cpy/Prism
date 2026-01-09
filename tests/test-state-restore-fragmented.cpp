// Test for state restore with fragmented KV cache
// This tests the fix for: https://github.com/ggml-org/llama.cpp/issues/17527
// The issue was that state restore required contiguous KV cache slots,
// which fails when the cache is fragmented.
//
// The fix changes find_slot(ubatch, true) to find_slot(ubatch, false)
// in state_read_meta(), allowing non-contiguous slot allocation.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {
const char * get_env_non_empty(const char * name) {
    const char * value = std::getenv(name);
    return (value && value[0] != '\0') ? value : nullptr;
}

const char * get_test_model_path() {
    if (const char * value = get_env_non_empty("LLAMACPP_TEST_MODELFILE")) {
        return value;
    }
    if (const char * value = get_env_non_empty("LLAMA_ARG_MODEL")) {
        return value;
    }
    if (const char * value = get_env_non_empty("LLAMA_TEST_MODEL")) {
        return value;
    }
    return nullptr;
}

bool is_model_source_arg(const std::string & arg) {
    return arg == "-m" || arg == "--model" ||
           arg == "-hf" || arg == "-hfr" || arg == "--hf-repo" ||
           arg == "-hfd" || arg == "-hfrd" || arg == "--hf-repo-draft" ||
           arg == "-hff" || arg == "--hf-file" ||
           arg == "-mu" || arg == "--model-url" ||
           arg == "-dr" || arg == "--docker-repo";
}

void build_args_with_model(int argc, char ** argv, const char * model_path, std::vector<std::string> & out) {
    out.clear();
    out.reserve(static_cast<size_t>(argc) + 4);
    out.emplace_back(argv[0]);
    out.emplace_back("--model");
    out.emplace_back(model_path);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (is_model_source_arg(arg)) {
            if (i + 1 < argc) {
                ++i;
            }
            continue;
        }
        out.emplace_back(arg);
    }
}
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 3;
    params.n_ctx = 256;

    const char * test_model_path = get_test_model_path();
    std::vector<std::string> argv_storage;
    std::vector<char *> argv_override;
    if (test_model_path != nullptr) {
        build_args_with_model(argc, argv, test_model_path, argv_storage);
        argv_override.reserve(argv_storage.size());
        for (auto & arg : argv_storage) {
            argv_override.push_back(const_cast<char *>(arg.c_str()));
        }
    }

    char ** argv_used = argv_override.empty() ? argv : argv_override.data();
    int argc_used = argv_override.empty() ? argc : static_cast<int>(argv_override.size());

    if (!common_params_parse(argc_used, argv_used, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    // init
    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    GGML_UNUSED(model);

    // tokenize prompt
    std::vector<llama_token> tokens(70, 1);

    // interleave the 3 sequences:
    // 01201230123...
    llama_batch batch = llama_batch_init(params.n_parallel*tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            common_batch_add(batch, tokens[i], i, {s}, false);
        }
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : processed prompt on seq 0, 1, 2 (%zu tokens each)\n", __func__, tokens.size());

    // Save state of seq 1
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 1));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (ncopy != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 1 state\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : saved seq 1 state, %zu bytes\n", __func__, ncopy);

    // clear seq 1 to create a "hole" in the KV cache (fragmentation)
    // 0.20.20.20.2....
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    fprintf(stderr, "%s : cleared seq 1 to create fragmentation\n", __func__);

    // Now the cache has holes where seq 1 was
    // This creates fragmentation - there's no contiguous block large enough
    // for the seq 1 state if we only look for contiguous slots

    // Restore seq 1 state into seq 1 (should work with non-contiguous allocation)
    // We use seq 1 since it's a valid sequence ID (0 to n_parallel-1)
    // Before the fix, this would fail with "failed to find available cells in kv cache"
    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (nset != seq_state.size()) {
        fprintf(stderr, "%s : FAILED to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, seq_state.size());
        fprintf(stderr, "%s : This is the bug - state restore fails with fragmented KV cache\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    fprintf(stderr, "%s : restored state into seq 1, %zu bytes\n", __func__, nset);

    // Verify we can decode with the restored state
    // Generate one token to verify the restored state is usable
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto next_token = llama_sampler_sample(smpl, ctx, -1);
    auto next_token_str = common_token_to_piece(ctx, next_token);

    common_batch_clear(batch);
    common_batch_add(batch, next_token, (int)tokens.size(), {1}, true);

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode with restored state\n", __func__);
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        return 1;
    }

    fprintf(stderr, "%s : successfully decoded with restored state, generated: '%s'\n", __func__, next_token_str.c_str());
    fprintf(stderr, "%s : SUCCESS - state restore works with fragmented KV cache\n", __func__);

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return 0;
}
