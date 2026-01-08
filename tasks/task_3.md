# Task 3: Attention Score Tracking + Main-Path H2O Prefill (Chunk = ubatch)

## Overview

This task wires H2O score tracking into the **main inference path** and enables **parallel prefill** by treating each ubatch as a chunk. We reuse the existing `llama_kv_cache` (non-iSWA path) and use H2O intra-attention weights to initialize per-token scores. The chunk size is **`-ub/--ubatch-size`**, and the total prompt processed per call is **`-b/--batch-size`** (e.g., 1024/4096).

**Key decisions (confirmed):**
- ✅ Reuse `llama_kv_cache` (no new cache types)
- ✅ Non-iSWA path for H2O
- ✅ Single-request (n_stream = 1) for now
- ✅ Chunk size = `n_ubatch`, logical batch = `n_batch`
- ✅ H2O params exposed in CLI, defaults 256/256

---

## Implementation Steps

### Step 1: Add H2O params to context + CLI

**Files**: `include/llama.h`, `src/llama-context.cpp`, `src/llama-cparams.h`, `common/common.h`, `common/common.cpp`, `common/arg.cpp`

Add two params with defaults (256/256):
- `h2o_local_window` (L)
- `h2o_heavy_budget` (H)

**Changes**:
- `include/llama.h`: extend `struct llama_context_params` with:
  - `uint32_t h2o_local_window;`
  - `uint32_t h2o_heavy_budget;`
- `src/llama-context.cpp`: set defaults in `llama_context_default_params()` to 256/256
- `src/llama-cparams.h`: carry these params into `llama_cparams`
- `src/llama-context.cpp`: propagate `params.h2o_*` into `cparams.h2o_*`
- `common/common.h`: add to `common_params` (defaults 256/256)
- `common/common.cpp`: map `common_params -> llama_context_params`
- `common/arg.cpp`: CLI options
  - `--h2o-local N` (default 256)
  - `--h2o-heavy N` (default 256)

**Behavior**:
- H2O enabled when `h2o_local_window + h2o_heavy_budget > 0`.
- If H2O enabled, force `cparams.flash_attn = false` to guarantee weight extraction path.

---

### Step 2: Force non-iSWA memory path for H2O

**File**: `src/llama-model.cpp`

In `llama_model::create_memory(...)`:
- If `cparams.h2o_local_window + cparams.h2o_heavy_budget > 0`, treat as **non-iSWA**.
- Instantiate `llama_kv_cache` directly with `swa_type = LLAMA_SWA_TYPE_H2O` and pass the H2O params.
- **Do not** use `llama_kv_cache_iswa` for H2O.

**Target logic** (pseudo):
```cpp
const bool h2o_enabled = cparams.h2o_local_window + cparams.h2o_heavy_budget > 0;
const llama_swa_type attn_swa_type = h2o_enabled ? LLAMA_SWA_TYPE_H2O : hparams.swa_type;

if (!h2o_enabled && hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
    // keep existing iSWA path
} else {
    // normal kv cache, with H2O enabled if requested
    res = new llama_kv_cache(..., attn_swa_type, ..., cparams.h2o_local_window, cparams.h2o_heavy_budget);
}
```

---

### Step 3: Parallel prefill via block-diagonal causal mask

**Files**: `src/llama-graph.cpp`, `src/llama-graph.h`

We treat each ubatch as a chunk and **compute all chunks within one logical batch** using a block-diagonal causal mask. This removes cross-chunk dependencies and allows intra-attn to run in parallel.

**Where**:
- Modify `llm_graph_input_attn_kv::set_input(...)` to **special-case H2O**.

**Mask rule** (n_stream=1):
- Let `chunk_size = cparams.n_ubatch`.
- For each query position `i`, allow keys only in `[chunk_start, i]` where `chunk_start = (i / chunk_size) * chunk_size`.
- Everything else = `-INFINITY`.

**Notes**:
- If `n_tokens <= chunk_size`, mask is standard causal.
- This ensures chunk0..chunk3 can be computed together when `n_batch=4096` and `n_ubatch=1024`.

---

### Step 4: Wire H2O attention + colsum in main path

**Files**: `src/llama-graph.cpp`, `src/llama-graph.h`

**Graph build changes**:
- In `llm_graph_context::build_attn(...)` (KV-cache overload), if H2O enabled:
  - call `build_attn_h2o(...)` to get attention weights
  - compute column sums via `build_attn_colsum(...)`
  - store colsum in `llm_graph_result`

**Add to llm_graph_result**:
- New container (e.g., `std::map<int, ggml_tensor *> t_h2o_colsum;`)
- Set outputs for these tensors in `llm_graph_result::set_outputs()` so they can be read post-compute.

---

### Step 5: Update scores after compute (chunk = ubatch)

**File**: `src/llama-context.cpp`

After `graph_compute(...)`:
- If H2O enabled and `ubatch.n_tokens > 1` (prefill path), read `t_h2o_colsum` tensors back to host.
- For each chunk in the batch:
  - `chunk_start = kv.h2o_get_total_tokens() + chunk_idx * chunk_size`
  - `chunk_len = min(chunk_size, remaining)`
  - call `kv.h2o_init_chunk_scores(il, chunk_start, chunk_len, colsum_slice)`
  - call `kv.h2o_next_chunk(chunk_len)` to advance chunk state

**Important**:
- For Task 3, we **only initialize scores** from intra-attn colsum.
- Inter-attn score accumulation (`h2o_accumulate_memory_scores`) will be wired in Task 5.

---

## Validation & Verification

### Unit tests (new)
**File**: `tests/test-h2o-score-tracking.cpp`

Test cases:
1. **Colsum correctness**: verify `build_attn_colsum` equals manual column sum on a tiny tensor.
2. **Score init**: after `h2o_init_chunk_scores`, score buffer matches expected BF16 values at global positions.
3. **Block-diagonal mask**: ensure cross-chunk attention weights are zero.

### Integration checks
- Run H2O with `-b 4096 -ub 1024` on a short prompt; confirm:
  - `h2o_state.total_tokens_processed == 4096` after one prefill
  - scores are initialized for all 4 chunks
  - no crash when input > 4096 (next chunk processed next call)

### Metrics to verify
- Score monotonicity (no decreases)
- Score alignment (global indices match chunk offsets)
- Cross-chunk masking (no leakage across chunk boundary)
- Throughput (prefill tokens/s improves vs sequential ubatch path)

---

## Deliverables

- **Core code**
  - `include/llama.h` / `src/llama-context.cpp`: H2O params + defaults
  - `src/llama-cparams.h`: carry H2O params
  - `common/common.h`, `common/arg.cpp`, `common/common.cpp`: CLI + mapping
  - `src/llama-model.cpp`: non-iSWA memory + H2O kv cache
  - `src/llama-graph.cpp`: H2O main-path attn + colsum capture
  - `src/llama-context.cpp`: score init + chunk progression

- **Tests**
  - `tests/test-h2o-score-tracking.cpp`

---

## Suggested build/verify commands

```bash
cmake --build build -j12

LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf \
ctest --test-dir build -R test-h2o-score-tracking -V --output-on-failure
```

---

## Notes for later tasks
- Inter-attn score updates (`h2o_accumulate_memory_scores`) are deferred to Task 5.
- Memory selection (Local + Heavy) is Task 4.
- Online softmax fusion is Task 5.
