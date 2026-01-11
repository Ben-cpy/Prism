# Task 5.5: Two‑Phase Prefill (Parallel Intra → Sequential Inter) for H2O

## Overview

Implement the **true two‑phase prefill schedule** required by the plan:

1) **Phase 1 (Parallel Intra)**: Use full batch (`-b`) to compute **all chunk intra‑attention in one pass**.  
2) **Phase 2 (Sequential Inter)**: Use micro‑batch (`-ub`) to process chunks **in order**, each chunk attends to the memory set from the previous chunk, then fuses inter + cached intra via online softmax.

This aligns with the expectation: *“一次性并行算出所有 chunk 的 intra（4 个 chunk），然后顺序逐 chunk 处理 inter/fusion”。*

**Constraints / Design Intent**
- RoPE uses **absolute positions** (`pos(t) = t`), no chunk‑relative encoding.
- **Traditional attention** path only (no flash‑attn changes).
- KV cache written **once** in Phase 1.
- Phase 2 uses **cached intra results**, not recomputed.

---

## Implementation Steps

### Step 1: Add Two‑Phase Prefill Orchestration

**File**: `/root/workspace/Prism/src/llama-context.cpp`

Add explicit two‑pass logic inside `llama_decode_internal` when:
`h2o_enabled && prefill (ubatch.n_tokens > 1)`.

Pseudo‑flow:
```
Phase 1:
  ubatch_size = n_batch
  current_phase = 1
  process_ubatch(full prompt)
  extract intra stats (o_intra / l_intra) + intra colsum
  init scores for each chunk
  build memory set for chunk0 only

Phase 2:
  ubatch_size = n_ubatch
  current_phase = 2
  for each chunk (sequential):
     process_ubatch(chunk)
     accumulate memory scores from inter colsum
     build memory set for chunk c
```

**Verification**:
- `Phase 1` executes exactly **one** ubatch when `n_tokens <= n_batch`.
- `Phase 2` executes `ceil(n_tokens / n_ubatch)` ubatches.

---

### Step 2: Add Phase‑1 Intra Cache (Host Buffers)

**Goal**: Cache intra results once, avoid recomputation in Phase 2.

**File**: `/root/workspace/Prism/src/llama-context.h`

Add a new cache container (host‑side, per layer):
```cpp
struct h2o_prefill_cache {
    // per layer:
    std::vector<std::vector<float>> intra_o; // [il] -> [n_embd_head_v * n_tokens * n_head * n_stream] normalized output
    std::vector<std::vector<float>> intra_l; // [il] -> [n_tokens * n_head * n_stream] sum exp(logit - m)
    std::vector<std::vector<float>> intra_m; // [il] -> [n_tokens * n_head * n_stream] row-wise max logits
    uint32_t n_tokens = 0;
    uint32_t n_head = 0;
    uint32_t n_embd_head_v = 0;
    uint32_t n_stream = 1;
    llama_pos base_pos = 0;
    bool initialized = false;
};
```

**Notes**:
- Store **online softmax state** (o, l, m) instead of full logits to avoid O(N²) memory.
- Shapes match per‑head outputs: `o_intra` in per‑head layout `[d_head, n_tokens, n_head, n_stream]`.

---

### Step 3: Phase‑1 Graph Outputs (Intra Only)

**Files**:  
`/root/workspace/Prism/src/llama-graph.h`  
`/root/workspace/Prism/src/llama-graph.cpp`

In `current_phase == 1`:
- Use `build_attn_h2o` to compute **intra** only (no inter).
- Compute and **export** online‑softmax state from **intra logits**:
  - `m_intra = rowwise_max(logits_intra)`
  - `l_intra = sum(exp(logits_intra - m_intra))`
  - `o_intra = softmax(logits_intra) @ V_intra` (normalized output)

These become graph outputs so Phase‑1 results can be copied to host:
```cpp
res->t_h2o_intra_o[il] = o_intra;
res->t_h2o_intra_l[il] = l_intra;
res->t_h2o_intra_m[il] = m_intra;
```

**Verification**:
- `ggml_set_output` called on these tensors.
- Host copies fill `h2o_prefill_cache`.

---

### Step 4: Phase‑2 Graph Inputs (Use Cached Intra)

**Files**:  
`/root/workspace/Prism/src/llama-graph.h`  
`/root/workspace/Prism/src/llama-graph.cpp`

In `current_phase == 2`:
- Create **input tensors** from `h2o_prefill_cache`:
  - `intra_o_input`
  - `intra_l_input`
  - `intra_m_input`
- Slice to current chunk (`[chunk_start, chunk_end)`) using `ggml_view_*`.
  - Use `token_offset = ubatch.pos[0] - base_pos` to index into cached full‑prompt tensors.
- Compute inter attention to memory set:
  - `inter_logits` (for m/l)
  - `inter_out` (normalized output)
- Initialize inter state and **fuse with cached intra**:
```
m_total = max(m_inter, m_intra)
scale_inter = exp(m_inter - m_total)
scale_intra = exp(m_intra - m_total)
l_inter = l_inter * scale_inter
l_intra = l_intra * scale_intra
o_sum = (o_inter * l_inter) + (o_intra * l_intra)
out = o_sum / (l_inter + l_intra)
```
If `use_inter == false` (chunk 0), `out = o_intra`.

---

### Step 5: Avoid Double KV Writes

**Files**:  
`/root/workspace/Prism/src/llama-graph.cpp`

Add a **skip‑store** path for Phase‑2:
- Phase 1 writes K/V to KV cache.
- Phase 2 **skips** `cpy_k/cpy_v` when `current_phase == 2` inside `build_attn_h2o`.

---

### Step 6: Score Update & Memory Set Timing

**File**: `/root/workspace/Prism/src/llama-context.cpp`

**Phase 1**:
- Initialize scores for **all chunks** from intra colsum.
- Build memory set **only for chunk 0**.

**Phase 2**:
- After inter attention for chunk c:
  - `h2o_accumulate_memory_scores` on `M_{c-1}`
  - `h2o_build_memory_set` for chunk c

**Final partial chunk (sync with `tasks/whole_plan.md`)**:
- “Final chunk” here means the **tail of the user’s full prompt**, i.e., the last chunk in the entire prompt sequence.
  It is **not** the last `ubatch` of each `-b` batch if the prompt requires multiple `-b` calls.
- The final chunk may be shorter than `n_ubatch`.
- After finishing inter + intra fusion for the final chunk:
  - **Do not** build a new memory set (no next prefill chunk will consume it).
  - **Do not** advance chunk state; prefill ends here.
  - KV for the final chunk is still stored at positions `[chunk_start, N)`.

This matches the plan’s scoring rules.

---

### Step 7: Graph Reuse Safety

**Files**:
`/root/workspace/Prism/src/llama-graph.h`  
`/root/workspace/Prism/src/llama-context.cpp`

Ensure `current_phase` is part of `llm_graph_params::allow_reuse` so:
- Phase‑1 graph won’t be reused for Phase‑2 (or vice versa).
Also ensure `llm_graph_input_h2o_intra::can_reuse` checks `ubatch.pos[0]` and cache sizes.

---

## Verification Checklist

### Correctness
- [ ] Phase‑1 runs once at `-b` and computes intra for all chunks.
- [ ] Phase‑2 runs `ceil(N / -ub)` ubatches sequentially.
- [ ] Chunk 0 uses only intra (no inter).
- [ ] Chunks 1..k use inter + cached intra fusion.
- [ ] Scores update: init from intra, accumulate from inter only.

### Performance
- [ ] Phase‑1 saturates GPU (parallel intra).
- [ ] Phase‑2 time scales with `k-1` chunks, not N².

### Robustness
- [ ] `N <= S` works (only phase‑1).
- [ ] `N == 2S-1` and `N == kS` work.

---

## Detailed Test Plan (Task 5.5)

### A) Unit / Component Tests (ctest)
These cover the building blocks used by the two‑phase schedule.

1) `test-h2o-online-softmax`
   - Purpose: verify online‑softmax state (m/l/o) and fusion math used in Phase 2.
   - Pass criteria: test prints success line and exits 0.

2) `test-h2o-intra-attn`
   - Purpose: validate intra‑attention correctness (Phase 1 core).
   - Pass criteria: "H2O intra-attn checks passed" and exit 0.

3) `test-h2o-score-tracking`
   - Purpose: verify score initialization (from intra colsum) and accumulation (from inter colsum).
   - Pass criteria: "H2O score tracking checks passed" and exit 0.

4) `test-h2o-memory-selection`
   - Purpose: validate memory set construction / selection based on scores.
   - Pass criteria: exits 0 (or prints pass line if present).

5) `test-h2o-gqa-memory`
   - Purpose: validate GQA head reduction for memory scores (n_head -> n_head_kv).
   - Pass criteria: exits 0.

6) `test-h2o-edge-cases`
   - Purpose: boundary cases for chunking and memory init.
   - Pass criteria: exits 0.

7) `test-h2o-kv-cache`
   - Purpose: verify KV cache layout and sizes used by H2O.
   - Pass criteria: exits 0.

### B) Two‑Phase Schedule Log Check (llama-cli)
Use the debug env vars to confirm Phase 1 runs once and Phase 2 runs per‑chunk, in order.

Command (make a prompt long enough to exceed `-ub`, but <= `-b` so there is only one phase‑1 pass):
```bash
python3 - <<'PY'
text = ("hello ") * 1500
with open('/tmp/h2o_prompt_1500.txt','w') as f:
    f.write(text)
PY

H2O_DEBUG_PHASES=1 H2O_DEBUG_CACHE=1 \
GGML_CUDA=ON ./build/bin/llama-cli \
  -m /models/Qwen_Qwen3-1.7B-Q8_0.gguf \
  -f /tmp/h2o_prompt_1500.txt -n 0 \
  -b 2048 -ub 1024 --h2o-local 256 --h2o-heavy 256 -st
```

Expected log checks (manual):
- Exactly one line with `"[H2O] phase=1"` and `ubatch.n_tokens` close to the prompt length.
- Two lines with `"[H2O] phase=2"`: first `ubatch.n_tokens=1024`, second is the tail (`<1024`), `chunk_idx` increasing.
- A `cache ready` line (from `H2O_DEBUG_CACHE=1`) confirming intra cache was filled.
- For the first phase‑2 chunk (`chunk_idx=0`), `has_prev=0`.
- For later chunks, `has_prev=1` and `mem_init=1`.

### C) Robustness Smoke (N <= S and partial tail)
Run a short prompt (`-b 1024 -ub 1024`) to confirm Phase 2 is skipped, then a partial tail (`-b 2048 -ub 1024` with shorter prompt length if possible) to validate last‑chunk behavior. Use `H2O_DEBUG_PHASES=1` to confirm.

---

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j12

LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf \
./build/bin/llama-cli -m /models/Qwen_Qwen3-1.7B-Q8_0.gguf \
  -p "Hello world from H2O! " -n 0 \
  -b 4096 -ub 1024 --h2o-local 256 --h2o-heavy 256 -st
```

---

## Success Criteria

- ✅ Phase‑1 computes all chunk intra **in one pass**
- ✅ Phase‑2 sequential inter + fusion uses cached intra
- ✅ Memory sets update correctly across chunks
- ✅ No double KV writes
- ✅ End‑to‑end prefill runs without crashes

---

## 附录：ggml_get_rows 扩展实现

### 问题背景
H2O 的 memory gathering 需要支持 per-head 索引：每个 KV-head 有独立的 memory set。
原 `ggml_get_rows` 限制索引 tensor 的第 4 维必须为 1（广播索引），无法支持此需求。

### 解决方案
扩展 `ggml_get_rows` 支持 per-batch 索引。

**修改点**：`ggml/src/ggml.c:3804`
```c
// 原限制：
GGML_ASSERT(b->ne[3] == 1);

// 放宽为：
GGML_ASSERT(b->ne[3] == 1 || b->ne[3] == a->ne[3]);
```

**语义**：
- `b->ne[3] == 1`：原有行为，索引广播到所有 batch（向后兼容）
- `b->ne[3] == a->ne[3]`：新增行为，每个 batch 使用独立索引

### 实现细节
**CPU backend** (`ggml-cpu/ops.cpp:4768-4807`)：
- 已有代码在 line 4799 访问 `i12*nb12`，天然支持多维索引
- **无需修改内核实现**

**CUDA backend** (`ggml-cuda/getrows.cu`)：
- 已有代码在 line 17-20 迭代 `ne11*ne12`，天然支持
- **无需修改内核实现**

### H2O 代码优化
**优化前**：4 个 ggml 操作
1. view (4D reshape)
2. permute (swap dims 1,2)
3. get_rows (失败：不支持 per-batch 索引)
4. permute (swap back dims 1,2)

**优化后**：3 个 ggml 操作（K cache）/ 4 个操作（V cache 带 transpose）
1. view (4D reshape)
2. permute (swap dims 1,2)
3. view (mem_idx reshape to 4D)
4. get_rows (支持 per-batch 索引，直接输出正确形状)
5. permute (仅 V cache 需要，如果 v_trans=true)

**性能提升**：
- K cache: 消除 1 个 permute 操作（从 4 步减少到 3 步）
- V cache: 保持相同操作数（4 步），但启用了 per-head gathering
- 关键改进：启用 per-batch 索引，使每个 KV-head 可以有独立的 memory set
- 更清晰的语义：直接表达 per-head gathering

### 测试覆盖
- `test-h2o-edge-cases`: 验证非均匀 chunk 场景
- `test-h2o-gqa-memory`: 验证 GQA per-head 索引
- `test-h2o-kv-cache`: 验证 KV cache 集成

### 关键修改文件
1. `ggml/src/ggml.c` - Line 3804: 放宽 `ggml_get_rows` 断言
2. `ggml/include/ggml.h` - Line 1636-1647: 更新 API 文档
3. `src/llama-kv-cache.cpp`:
   - Line 1248-1276: 修改 `h2o_gather_k_memory`（添加 permute，使用扩展的 get_rows）
   - Line 1297-1328: 修改 `h2o_gather_v_memory`（添加 permute，使用扩展的 get_rows）
