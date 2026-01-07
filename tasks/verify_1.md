
LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf LLAMA_CACHE=/tmp/llama-cache ctest --test-dir /workspace/Prism/build -R test-h2o-kv-cache -V --output-on-failure

ctest --test-dir /workspace/Prism/build -R test-h2o-kv-cache -V --output-on-failure