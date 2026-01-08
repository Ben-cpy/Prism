## Task 2 Verification (H2O Intra-Chunk Attention)

### Build (use full cores)
```bash
cmake --build build -j12
```

### Run (single test)
```bash
./build/bin/test-h2o-intra-attn
```

### Run (ctest with model path)
```bash
LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf \
LLAMA_CACHE=/tmp/llama-cache \
ctest --test-dir /root/workspace/Prism/build -R test-h2o-intra-attn -V --output-on-failure
```

### Expected result
- `test-h2o-intra-attn` prints: `H2O intra-attn checks passed`
- Exit code `0`
