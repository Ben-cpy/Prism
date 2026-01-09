# Task 6: Full Pipeline Integration for Chunked Sparse Attention (H2O Prefill)

## Overview

Integrate the existing H2O components (KV cache, graph attention, intra/inter fusion, score tracking) into a coherent, end-to-end chunked prefill pipeline. This task focuses on phase orchestration, per-layer execution, and edge-case correctness.

Design decisions (confirmed):
- Phase 1 prefill uses n_batch to run all physical chunks in parallel (ubatches inside the logical batch).
- Single-batch text-only support (no multimodal or multi-sequence paths).
- Memory selection remains per-chunk during phase 1 (plan A).
- H2O cache lifetime is one prefill only; reset at the start of each prefill.

---

## Implementation Steps

### Step 1: Confirm H2O activation and chunk sizing in decode

**Files**
- `src/llama-context.cpp`

**What to do**
1. Keep H2O enabled when `h2o_local_window + h2o_heavy_budget > 0`.
2. Set `h2o_chunk_size` from `n_ubatch` (physical chunk size) and ensure it matches the KV cache chunk calculations.
3. Ensure `h2o_cache.initialized = false` at the start of each H2O prefill (single-prefill lifetime).
4. **Phase-1 parallelization (critical clarification)**:
   - Phase 1 processes the entire `n_batch` (e.g., 4096 tokens) in **one forward pass**
   - However, the attention computation uses a **block-diagonal causal mask**, NOT full attention
   - This means:
     - Token at position 0-1023 (chunk 0) only attends to positions 0-1023
     - Token at position 1024-2047 (chunk 1) only attends to positions 1024-2047
     - etc.
   - Computationally: **4 × (1024×1024) = 4M dot products**, not **4096×4096 = 16.7M**
   - Memory: Higher than sequential processing (all 4 chunks' KV in memory), but acceptable
   - Benefit: All chunks compute intra-attention **simultaneously** in a single GPU kernel

**Block-diagonal mask visualization**:
```
For n_batch=4096, n_ubatch=1024 (4 chunks):

Query positions →
    0    1024   2048   3072   4096
    ┌────┬────┬────┬────┐
  0 │████│    │    │    │  Chunk 0: causal within [0, 1023]
    │████│    │    │    │
1024├────┼────┼────┼────┤
    │    │████│    │    │  Chunk 1: causal within [1024, 2047]
    │    │████│    │    │
2048├────┼────┼────┼────┤
    │    │    │████│    │  Chunk 2: causal within [2048, 3071]
    │    │    │████│    │
3072├────┼────┼────┼────┤
    │    │    │    │████│  Chunk 3: causal within [3072, 4095]
4096└────┴────┴────┴────┘

█ = attention allowed (causal within chunk)
  = attention blocked (cross-chunk, handled in Phase 2)
```

---

### Step 2: Phase-1 intra pass + cache capture

**Files**
- `src/llama-context.cpp`
- `src/llama-graph.cpp`
- `src/llama-graph.h`
- `src/llama-context.h`

**What to do**
1. In phase 1, ensure `build_attn_h2o()`:
   - Computes intra attention.
   - Emits `t_h2o_intra_o`, `t_h2o_intra_l`, `t_h2o_intra_m`.
   - Emits `t_h2o_colsum` for score initialization.
2. Copy these tensors into `h2o_cache` once per prefill:
   - Validate `n_head`, `n_embd_head_v`, `n_tokens`, `n_stream`.
   - Record `h2o_cache.base_pos = ubatch.pos[0]`.
3. Set `h2o_cache.initialized = true` only after all layers are captured.

---

### Step 3: Score initialization from intra colsum (per chunk)

**Files**
- `src/llama-context.cpp`
- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`

**What to do**
1. Use `t_h2o_colsum` from phase 1 to initialize scores:
   - `kv->h2o_init_chunk_scores(il, chunk_start, chunk_len, colsum)`
2. Build memory set for chunk 0 immediately after scores:
   - `kv->h2o_build_memory_set(il, chunk_end)`
3. Compute `chunk_start` / `chunk_end` with `base_tokens + offset` to ensure absolute positions.

---

### Step 4: Phase-2 inter attention + fusion

**Files**
- `src/llama-graph.cpp`
- `src/llama-context.cpp`
- `src/llama-kv-cache.cpp`

**What to do**
1. In phase 2, ensure `build_attn_h2o()`:
   - Uses `h2o_gather_k_memory()` / `h2o_gather_v_memory()` for inter attention.
   - Builds online softmax state from inter logits.
   - Fuses with cached intra output from `h2o_cache`.
2. Ensure `t_h2o_inter_colsum` is produced for score accumulation.

---

### Step 5: Score accumulation + memory set update per chunk

**Files**
- `src/llama-context.cpp`
- `src/llama-kv-cache.cpp`

**What to do**
1. After phase-2 inter attention:
   - `kv->h2o_accumulate_memory_scores(il, inter_colsum)`
   - `kv->h2o_build_memory_set(il, chunk_end)`
2. Call `kv->h2o_next_chunk(chunk_len)` after all updates.
3. Increment chunk index and keep it aligned with the ubatch progression.

---

### Step 6: Edge-case handling (single-batch text only)

**Files**
- `src/llama-context.cpp`
- `src/llama-graph.cpp`

**What to do**
1. Single chunk (`n_tokens <= chunk_size`): in here the chunk_size equals to ubatch size
   - Bypass H2O inter path and run standard attention.
2. Partial final chunk:
   - Use `chunk_len = min(h2o_chunk_size, remaining_tokens)`.
   - Ensure no out-of-bounds access when slicing KV or colsum buffers.
3. Disable H2O for non-text or multi-sequence inputs (keep scope to single-batch text).


---

## Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| End-to-end correctness | Reasonable output quality | Compare perplexity vs baseline on fixed prompts |
| Phase correctness | No missing H2O tensors | Validate `h2o_cache.initialized` and per-layer sizes |
| Edge cases | No crashes | Run single chunk, partial chunk, multi-chunk tests |
| Memory stability | Constant peak memory | Repeat prefills and confirm stable memory |

---

## Test & Validation Programs (Design)

### 1) Single-chunk parity test
- Prompt length < `chunk_size`.
- Compare logits vs baseline (H2O disabled).

### 2) Multi-chunk correctness test
- Prompt length = `2 * chunk_size + 128`.
- Confirm `h2o_cache.initialized` after phase 1.
- Confirm inter colsum updates in phase 2.

### 3) Partial tail chunk test
- Prompt length not divisible by chunk size.
- Ensure no out-of-bounds, and shapes are consistent.

### 4) Stability test
- Run repeated prefill calls on the same prompt.
- Confirm no memory leak or unexpected cache reuse.

---

## Success Criteria

✅ H2O prefill works end-to-end for multi-chunk prompts  
✅ Single chunk path uses standard attention with no H2O overhead  
✅ No crashes on partial tail chunks  
✅ Score tracking and memory set updates occur once per chunk  
✅ Online softmax fusion is numerically stable (no NaN/Inf)
