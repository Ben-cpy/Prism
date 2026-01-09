// thread safety test
// - Loads a copy of the same model on each GPU, plus a copy on the CPU
// - Creates n_parallel (--parallel) contexts per model
// - Runs inference in parallel on each context

#include <array>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdlib>

#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"

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

bool should_force_cpu_for_test_model() {
    const char * force_gpu = get_env_non_empty("LLAMA_TEST_THREAD_SAFETY_GPU");
    return force_gpu == nullptr || force_gpu[0] == '0';
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

    const char * test_model_path = get_test_model_path();
    const bool force_cpu = test_model_path != nullptr && should_force_cpu_for_test_model();
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
    if (force_cpu) {
        params.n_gpu_layers = 0;
        params.main_gpu = -1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    //llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
    //    if (level == GGML_LOG_LEVEL_ERROR) {
    //        common_log_add(common_log_main(), level, "%s", text);
    //    }
    //}, NULL);

    auto cparams = common_context_params_to_llama(params);

    // each context has a single sequence
    cparams.n_seq_max = 1;

    int dev_count = ggml_backend_dev_count();
    std::vector<std::array<ggml_backend_dev_t, 2>> gpus;
    if (!force_cpu) {
        for (int i = 0; i < dev_count; ++i) {
            auto * dev = ggml_backend_dev_get(i);
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                gpus.push_back({dev, nullptr});
            }
        }
    }
    const int gpu_dev_count = (int)gpus.size();
    const int num_models = force_cpu ? 1 : (gpu_dev_count + 1 + 1); // GPUs + 1 CPU model + 1 layer split
    //const int num_models = std::max(1, gpu_dev_count);
    const int num_contexts = std::max(1, params.n_parallel);

    std::vector<llama_model_ptr> models;
    std::vector<std::thread> threads;
    std::atomic<bool> failed = false;

    for (int m = 0; m < num_models; ++m) {
        auto mparams = common_model_params_to_llama(params);

        if (!force_cpu && m < gpu_dev_count) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.devices = gpus[m].data();
        } else if (!force_cpu && m == gpu_dev_count) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu = -1; // CPU model
        } else if (!force_cpu) {
            mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
        } else {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu = -1;
        }

        llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model == NULL) {
            LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
            return 1;
        }

        models.emplace_back(model);
    }

    for  (int m = 0; m < num_models; ++m) {
        auto * model = models[m].get();
        for (int c = 0; c < num_contexts; ++c) {
            threads.emplace_back([&, m, c, model]() {
                LOG_INF("Creating context %d/%d for model %d/%d\n", c + 1, num_contexts, m + 1, num_models);

                llama_context_ptr ctx { llama_init_from_model(model, cparams) };
                if (ctx == NULL) {
                    LOG_ERR("failed to create context\n");
                    failed.store(true);
                    return;
                }

                std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler { common_sampler_init(model, params.sampling), common_sampler_free };
                if (sampler == NULL) {
                    LOG_ERR("failed to create sampler\n");
                    failed.store(true);
                    return;
                }

                llama_batch batch = {};
                {
                    auto prompt = common_tokenize(ctx.get(), params.prompt, true);
                    if (prompt.empty()) {
                        LOG_ERR("failed to tokenize prompt\n");
                        failed.store(true);
                        return;
                    }
                    batch = llama_batch_get_one(prompt.data(), prompt.size());
                    if (llama_decode(ctx.get(), batch)) {
                        LOG_ERR("failed to decode prompt\n");
                        failed.store(true);
                        return;
                    }
                }

                const auto * vocab = llama_model_get_vocab(model);
                std::string result = params.prompt;

                for (int i = 0; i < params.n_predict; i++) {
                    llama_token token;
                    if (batch.n_tokens > 0) {
                        token = common_sampler_sample(sampler.get(), ctx.get(), batch.n_tokens - 1);
                    } else {
                        token = llama_vocab_bos(vocab);
                    }

                    result += common_token_to_piece(ctx.get(), token);

                    if (llama_vocab_is_eog(vocab, token)) {
                        break;
                    }

                    batch = llama_batch_get_one(&token, 1);

                    int ret = llama_decode(ctx.get(), batch);
                    if (ret == 1 && i > 0) {
                        LOG_INF("Context full, stopping generation.\n");
                        break;
                    }

                    if (ret != 0) {
                        LOG_ERR("Model %d/%d, Context %d/%d: failed to decode\n", m + 1, num_models, c + 1, num_contexts);
                        failed.store(true);
                        return;
                    }
                }

                LOG_INF("Model %d/%d, Context %d/%d: %s\n\n", m + 1, num_models, c + 1, num_contexts, result.c_str());
            });
        }
    }

    for (auto & thread : threads) {
        thread.join();
    }

    if (failed) {
        LOG_ERR("One or more threads failed.\n");
        return 1;
    }

    LOG_INF("All threads finished without errors.\n");
    return 0;
}
