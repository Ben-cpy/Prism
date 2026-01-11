# Task 6: Full Pipeline Integration for Chunked Sparse Attention (H2O Prefill)

## Overview

Integrate the existing H2O components (KV cache, graph attention, intra/inter fusion, score tracking) into a coherent, end-to-end chunked prefill pipeline. This task focuses on phase orchestration, per-layer execution, and edge-case correctness.

Design decisions (confirmed):
- Phase 1 prefill uses n_batch to run all physical chunks in parallel (ubatches inside the logical batch).
- Single-batch text-only support (no multimodal or multi-sequence paths).
- Memory selection remains per-chunk during phase 1 (plan A).
- H2O intra cache (`h2o_cache`) is per prefill call (`llama_decode`); reset at the start of each prefill call.  
  KV H2O state (scores + memory set) persists across multiple calls of the **same prompt**.

---

## Implementation Steps

### Step 1: Confirm H2O activation and chunk sizing in prefill

**Files**
- `src/llama-context.cpp`

**What to do**
1. Feature gate: H2O is **available** when `h2o_local_window + h2o_heavy_budget > 0`.  
   H2O **prefill path** is used only when:
   - `cparams.causal_attn == true`
   - single-stream text (`n_stream == 1`, not `pos_2d`)
   - `n_ubatch > 0`
   - `n_tokens_all > h2o_chunk_size` (i.e., > 1 chunk in this call)
2. Set `h2o_chunk_size = (n_ubatch > 0 ? n_ubatch : n_batch)` and keep it consistent with KV cache chunk calculations.  
   **Note**: block-diagonal masking relies on `n_ubatch > 0`; if `n_ubatch == 0`, treat H2O prefill as disabled.
3. Ensure `h2o_cache.initialized = false` at the start of **each prefill call** (per `llama_decode`).  
   Do **not** reset KV H2O state when continuing the same prompt across multiple calls.
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
2. Copy these tensors into `h2o_cache` **once per prefill call**:
   - Validate `n_head`, `n_embd_head_v`, `n_tokens`, `n_stream`.
   - Record `h2o_cache.base_pos = ubatch.pos[0]` (absolute position of this call’s first token).
3. Set `h2o_cache.initialized = true` only after all layers are captured.

---

### Step 3: Score initialization from intra colsum (per chunk)

**Files**
- `src/llama-context.cpp`
- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`

**What to do**
1. Use `t_h2o_colsum` from phase 1 to initialize scores.  
   For GQA, **reduce Q-head colsum → KV-head colsum** (`n_head_mem`) before calling `h2o_init_chunk_scores`.  
   (For MHA this is a no-op.)
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
   For GQA, **reduce to KV-heads** before `h2o_accumulate_memory_scores`.

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
1. Single chunk (`n_tokens_all <= h2o_chunk_size` for this call):  
   - Standard attention semantics (full causal within the call).  
   - Implementation may still route through `build_attn_h2o`; mask degenerates to standard and H2O updates are skipped.
2. Partial final chunk:
   - Use `chunk_len = min(h2o_chunk_size, remaining_tokens)`.
   - Ensure no out-of-bounds access when slicing KV or colsum buffers.
3. Disable H2O for non-text or multi-sequence inputs (keep scope to single-batch text).

### Step 6.5: Cross-Batch Memory Set Inheritance

**Files**
- `src/llama-context.cpp`

**What to do**

When the **prompt length > n_batch**, the caller must split the prompt into **multiple `llama_decode` calls** (logical batches).  
Cross-batch H2O **memory set** must persist across these calls.

1. **First call (batch_idx = 0)**:
   - Process as normal: Phase 1 → Phase 2 → build M_{k-1}
   - **DO NOT reset** KV H2O state after completion

2. **Subsequent calls (batch_idx > 0)**:
   - **Inherit** memory set M_{prev} from previous call
   - First chunk of new call uses inter-attention with M_{prev}
   - Continue Phase 2 sequential processing

**Example**: 16384 tokens, n_batch=4096, n_ubatch=1024

```
Call 0: tokens [0, 4095]
  Phase 1: all 4 chunks compute intra in parallel
  Phase 2: sequential inter → produces M₃

Call 1: tokens [4096, 8191]
  Phase 1: all 4 chunks compute intra in parallel
  Phase 2: Chunk 4 uses M₃ (from Call 0!) → produces M₄, M₅, M₆, M₇

Call 2: tokens [8192, 12287]
  Phase 1: all 4 chunks compute intra in parallel
  Phase 2: Chunk 8 uses M₇ (from Call 1!) → produces M₈, M₉, M₁₀, M₁₁
```

**Implementation**:
```cpp
// In llama_decode_internal, at the START of each call:
const uint32_t base_tokens = kv->h2o_get_total_tokens();
const bool is_first_batch = (base_tokens == 0);

if (!is_first_batch) {
    // Memory set from previous call is still valid
    // h2o_memory_initialized == true
    // First chunk of this call will use it for inter-attention
}
```
**Important**:
- Reset `h2o_cache` per call (phase-1 intra cache), but **do not reset KV H2O state** until a new request starts.
- Only the **final chunk of the final prefill call** should skip building M; intermediate calls must still build M for the next call.
---

### Step 7: Decode phase handling (Full KV cache, H2O updates disabled)

**Files**
- `src/llama-context.cpp`
- `src/llama-graph.cpp`

**What to do**

#### 7.1 Prefill-to-Decode Transition

After prefill completes, the KV cache contains:
1. **Memory set tokens**: The final $M = L + H$ tokens selected by H2O from all previous chunks (**with $M < \text{chunk\_size}$**)
2. **Final chunk tokens**: Complete KV for the last chunk (may be partial, i.e., < `chunk_size`)

**Critical**: No additional scoring or eviction happens after the final chunk. The KV cache state at prefill end becomes the initial state for decode.

#### 7.2 Final Chunk Special Handling

The final chunk requires special attention.  
**Note**: This refers to the **last chunk of the last prefill call**. If more prompt tokens remain (next `llama_decode` call), you must still build M for cross-batch continuation.

```
Example: N = 4096 + 500 tokens, chunk_size = 1024

Prefill chunks:
  Chunk 0: [0, 1023]      → intra-attn only, build M₀
  Chunk 1: [1024, 2047]   → inter-attn(M₀) + intra, build M₁
  Chunk 2: [2048, 3071]   → inter-attn(M₁) + intra, build M₂
  Chunk 3: [3072, 4095]   → inter-attn(M₂) + intra, build M₃
  Chunk 4: [4096, 4595]   → inter-attn(M₃) + intra, **NO M₄ build**
                            ↑ Final partial chunk (500 tokens)

KV cache at decode start:
  - Memory set M₃: 512 tokens (indices scattered across [0, 4095])
  - Final chunk: 500 tokens at positions [4096, 4595]
  - Total effective KV: M₃ ∪ [4096, 4595] (but stored as full cache)
```

**Implementation**:
```cpp
// In llama_decode_internal, after all chunks processed:

const bool is_final_chunk = (chunk_idx == n_chunks - 1);
const uint32_t final_chunk_len = n_tokens_all - chunk_idx * h2o_chunk_size;

if (is_final_chunk) {
    // 1. Compute inter-attention with M_{k-1}
    // 2. Fuse with intra-attention via online softmax
    // 3. DO NOT call h2o_build_memory_set() - prefill ends here
    // 4. DO NOT call h2o_next_chunk() - no more chunks to process
    
    // The KV cache now contains:
    //   - Memory set indices at their original global positions
    //   - Final chunk KV at positions [final_chunk_start, final_chunk_end)
}
```

#### 7.3 Decode Phase Execution

**Detection**: Decode mode is detected when `ubatch.n_tokens == 1`.

**Behavior when H2O enabled but in decode mode**:

```cpp
const bool is_decode = (ubatch.n_tokens == 1);
const bool h2o_enabled = cparams.h2o_local_window + cparams.h2o_heavy_budget > 0;

if (h2o_enabled && is_decode) {
    // === STANDARD DECODE SEMANTICS ===
    // Preferred implementation: keep build_attn_h2o() path.
    // - For n_tokens == 1, the mask degenerates to standard causal attention.
    // - H2O scoring/memory updates are skipped (ubatch.n_tokens == 1 gate).
    // Optional micro-optimization: skip colsum generation in decode.
}
```

**Why full attention during decode**:
- Decode complexity is $O(N + t)$ per token where $t$ = tokens generated
- This is **not** the bottleneck (prefill's $O(N^2)$ is)
- Full KV ensures maximum generation quality
- Simpler implementation: reuse existing llama.cpp decode path entirely

#### 7.4 KV Cache State During Decode

```
Decode token t₁ (first generated token):
  ┌────────────────────────────────────────────────────────┐
  │ KV Cache after prefill:                                │
  │   [0, N-1] = prompt KV (sparse during prefill,         │
  │              but all positions written)                │
  │                                                        │
  │ For decode token at position N:                        │
  │   Q(t₁) attends to K[0:N], V[0:N]  (FULL attention)    │
  │   Store K(t₁), V(t₁) at position N                     │
  └────────────────────────────────────────────────────────┘

Decode token t₂:
  ┌────────────────────────────────────────────────────────┐
  │ Q(t₂) attends to K[0:N+1], V[0:N+1]                    │
  │ Store K(t₂), V(t₂) at position N+1                     │
  └────────────────────────────────────────────────────────┘

... and so on for subsequent tokens
```

#### 7.5 Implementation Checklist

1. **Detect decode mode**: `ubatch.n_tokens == 1`
2. **Full KV access**: Attention covers the entire cache `[0, current_pos)`
3. **Standard KV write**: Store K, V at `kv_head + n_past` as normal
4. **Skip H2O state updates**: No `h2o_init_chunk_scores`, `h2o_accumulate_memory_scores`, `h2o_build_memory_set`, or `h2o_next_chunk`
5. **Implementation choice**: keep `build_attn_h2o()` (simplest, already aligned), or explicitly bypass it if you want to shave decode overhead.

---

## Verification Metrics (Decode Phase)

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Decode uses full attention | No H2O masking | Check attention covers [0, current_pos) |
| KV correctly appended | Position N+t stored | Verify `kv_cache.cells[N+t]` populated |
| No H2O state changes | Scores unchanged | Compare scores before/after decode |
| Generation quality | Match baseline | Compare output text vs non-H2O model |
| Decode latency | No regression | Benchmark decode tok/s |

---

---

## Verification Metrics (Overall)

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

## Success Criteria

✅ H2O prefill works end-to-end for multi-chunk prompts  
✅ Single chunk path uses standard attention with no H2O overhead  
✅ No crashes on partial tail chunks  
✅ Score tracking and memory set updates occur once per chunk  
✅ Online softmax fusion is numerically stable (no NaN/Inf)  
✅ **Final chunk does NOT build memory set (prefill termination)**  
✅ **Decode phase uses standard full attention (H2O bypassed)**  
✅ **KV cache correctly stores decode tokens at positions N, N+1, ...**  
✅ **Generation quality matches baseline (non-H2O) decode**
