该测试文件验证了什么（tests/test-h2o-score-tracking.cpp）

1. Colsum correctness
    用一个小的 Q/K/V 张量，比较 build_attn_colsum() 输出与手工逐列求和的结果，要求误差 < 1e-4。
2. Score init
    调 kv.h2o_init_chunk_scores() 后，检查 H2O score buffer 中每个头、每个 token 的 BF16 值与输入的 colsum 对应位置完全
    一致。
3. Block‑diagonal mask
    使用 n_tokens=8, chunk_size=4，验证跨 chunk 的 attention 被 mask（-inf），同 chunk 内（且因果范围内）不被 mask。

期望的结果

- 程序打印：H2O score tracking checks passed
- 进程退出码为 0（ctest 通过）

对应命令

构建
cmake --build build -j12 --target test-h2o-score-tracking

LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf ctest --test-dir build -R test-h2o-score-tracking -V 
--output-on-failure

- ctest（推荐，与任务文档一致）：

LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf \
ctest --test-dir build -R test-h2o-score-tracking -V --output-on-failure
- 直接运行二进制：

./build/bin/test-h2o-score-tracking /models/Qwen_Qwen3-1.7B-Q8_0.gguf