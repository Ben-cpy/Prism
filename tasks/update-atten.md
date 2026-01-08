# H2O Online Softmax vs Flash-Attention Comparison Analysis

## Executive Summary

This document compares the current H2O attention implementation (Tasks 4/5) with llama.cpp's flash-attention implementation to identify optimization opportunities.

---

## Detailed Comparison Table

| Aspect | Current H2O Implementation | Flash-Attention (llama.cpp) | Optimization Opportunity |
|--------|---------------------------|----------------------------|--------------------------|
| **1. Numerical Stability - Max Tracking** | Clamps logits to [-50, 50] before `exp()`. Does NOT track row-wise max. | Tracks `m = row_max` explicitly, computes `exp(x - m)`. Uses `FATTN_KQ_MAX_OFFSET = 3*0.6931f` shift. | **HIGH**: H2O should track `m` per row to avoid overflow with large logits. Clamping loses precision. |
| **2. State Representation** | Stores `o = kqv * l` (unnormalized weighted sum) and `l = Σexp(logit)`. No max stored. | Stores `(m, l, o)` where `m = max`, `l = Σexp(logit-m)`, `o = Σv*exp(logit-m)/l` (normalized). | **HIGH**: Current approach can overflow when `l` becomes very large. Should store normalized state. |
| **3. Block Fusion Formula** | `o_total = o_inter + o_intra`, `l_total = l_inter + l_intra`, `output = o_total / l_total` | `m_new = max(m_old, m_block)`, `l_new = l_old*exp(m_old-m_new) + l_block*exp(m_block-m_new)`, `o_new = o_old*exp(m_old-m_new) + o_block*exp(m_block-m_new)` | **CRITICAL**: H2O's direct addition assumes same max across blocks. This is **mathematically incorrect** when inter/intra have different max values! |
| **4. Flush-to-Zero Handling** | No explicit handling | Uses `SOFTMAX_FTZ_THRESHOLD = -20.0f`. Values with `exp(x) < exp(-20)` are zeroed. | **MEDIUM**: Prevents denormals and underflow accumulation. |
| **5. Data Types** | Scores: BF16, o/l cache: F32, attention weights: native (F16/F32) | Uses F16/F32 with careful scaling. Q scaled by 0.25x, results by 4x for FP16 overflow prevention. | **MEDIUM**: Current BF16 for scores may lose precision. Consider F32 for critical paths. |
| **6. Memory Layout** | Host-side `h2o_prefill_cache` stores `intra_o[n_embd_head_v * n_tokens * n_head]`. Per-layer vectors. | Shared memory with padding (`D_padded = D + 8`). Avoids bank conflicts. | **LOW** (for CPU), **HIGH** (for CUDA): Should add padding for CUDA shared memory. |
| **7. Warp Reductions** | Not implemented (CPU-style loops) | Uses `warp_reduce_max<warp_size>()` and warp-level sum reductions. | **HIGH** (CUDA): Missing warp-level optimizations for GPU. |
| **8. Memory Coalescing** | `ggml_get_rows` for KV gathering. Indices sorted by position. | Decreasing granularity loads (16-byte cp.async, then 4-byte). | **MEDIUM**: Current approach is reasonable but could use vectorized loads. |
| **9. Tiling Strategy** | Two-phase: Phase 1 (full batch), Phase 2 (chunk-size ubatches). Chunk size = n_ubatch. | Multi-level: `FATTN_KQ_STRIDE=256` rows per iteration, configurable `nbatch_fa`. | **LOW**: Current approach is simpler but less flexible. |
| **10. Parallel Block Combination** | Sequential phase 2 processing. No parallel block reduction. | Multiple blocks work on different KV ranges, combine via: `VKQ_num = Σ(KQ_scale[l] * VKQ_parts[l])` | **MEDIUM**: Could parallelize memory set processing across layers. |

---

## Critical Bug Identified

### Issue: Clamping Destroys Relative Scale Information

**Current H2O code** (`llama-graph.cpp:2048-2058, 2190-2236`):
```cpp
// h2o_logits_sum_exp - the root cause
ggml_tensor * h2o_logits_sum_exp(ggml_context * ctx, ggml_tensor * logits) {
    constexpr float k_clamp_min = -50.0f;
    constexpr float k_clamp_max =  50.0f;
    ggml_tensor * logits_clamped = ggml_clamp(ctx, logits_f32, k_clamp_min, k_clamp_max);
    ggml_tensor * exp_logits = ggml_exp(ctx, logits_clamped);
    return ggml_sum_rows(ctx, exp_logits);  // z = Σexp(clamp(logit))
}

// Fusion in update_online_softmax_state_h2o
o_inter = kqv_inter * z_inter;  // o = (softmax @ V) * Σexp(clamp(s))
o_intra = kqv_intra * z_intra;
o_total = o_inter + o_intra;    // Direct addition IS mathematically valid
z_total = z_inter + z_intra;    // IF both use same reference scale
output = o_total / z_total;
```

**Why the formula is correct but clamping breaks it**:

The fusion formula `o_total/z_total` is mathematically equivalent to standard softmax attention:
- `o = Σ(v * exp(s))` and `z = Σexp(s)`
- `output = Σ(v * exp(s)) / Σexp(s)` = correct softmax

**BUT** clamping corrupts the relative scale:
```
Example: inter_logit_max = 30, intra_logit_max = 60

TRUE relative weight of intra vs inter:
  exp(60) / exp(30) = exp(30) ≈ 10^13

WITH CLAMPING ([-50, 50]):
  inter: exp(30) (unchanged)
  intra: exp(50) (clamped from 60!)
  Apparent ratio: exp(50) / exp(30) = exp(20) ≈ 10^9

  ERROR: 10^4 times wrong!
```

**Flash-attention solution** (proper max tracking):
```cpp
// Compute per-row max FIRST
m_inter = row_max(inter_logits);  // [n_tokens, n_head]
m_intra = row_max(intra_logits);

// Compute sum with max subtraction (stays in safe range)
z_inter = Σexp(inter_logits - m_inter);  // values in (0, n_kv]
z_intra = Σexp(intra_logits - m_intra);

// When fusing, find global max and rescale
m_total = max(m_inter, m_intra);
scale_inter = exp(m_inter - m_total);
scale_intra = exp(m_intra - m_total);

// Properly scaled combination
z_total = z_inter * scale_inter + z_intra * scale_intra;
o_total = o_inter * scale_inter + o_intra * scale_intra;
```

**Why current code appears to work in practice**:
1. Typical attention logits stay in [-30, 30] range (well within [-50, 50])
2. Similar tokens in inter/intra tend to have similar max logits
3. Quality degradation is subtle (affects edge cases more than common cases)

---

## Optimization Priority Matrix

| Priority | Optimization | Effort | Impact | Files Affected |
|----------|--------------|--------|--------|----------------|
| **P0** | Fix fusion formula with max tracking | Medium | Correctness | `llama-graph.cpp` |
| **P1** | Track row-wise max (m) in online softmax state | Medium | Stability | `llama-graph.cpp`, `llama-graph.h` |
| **P2** | Add flush-to-zero threshold | Low | Stability | `llama-graph.cpp` |
| **P3** | Use normalized state storage | Medium | Memory | `llama-context.h`, `llama-graph.cpp` |
| **P4** | Add warp reductions (CUDA) | High | Performance | New CUDA kernels |
| **P5** | Optimize memory gathering patterns | Medium | Performance | `llama-kv-cache.cpp` |

---

## Recommended Code Changes

### Change 1: Replace `h2o_logits_sum_exp` with Max-Aware Version

**File**: `src/llama-graph.cpp` (lines 2048-2058)

```cpp
// CURRENT (problematic):
ggml_tensor * h2o_logits_sum_exp(ggml_context * ctx, ggml_tensor * logits) {
    ggml_tensor * logits_clamped = ggml_clamp(ctx, logits_f32, -50.0f, 50.0f);
    ggml_tensor * exp_logits = ggml_exp(ctx, logits_clamped);
    return ggml_sum_rows(ctx, exp_logits);
}

// PROPOSED (max-aware):
struct h2o_sum_exp_result {
    ggml_tensor * m;  // row-wise max: [n_tokens, n_head, n_stream]
    ggml_tensor * l;  // sum of exp(logit - m): [n_tokens, n_head, n_stream]
};

h2o_sum_exp_result h2o_logits_sum_exp_with_max(ggml_context * ctx, ggml_tensor * logits) {
    // logits shape: [n_kv, n_tokens, n_head, n_stream]
    ggml_tensor * logits_f32 = logits->type == GGML_TYPE_F32
        ? logits : ggml_cast(ctx, logits, GGML_TYPE_F32);
    if (!ggml_is_contiguous(logits_f32)) {
        logits_f32 = ggml_cont(ctx, logits_f32);
    }

    // Step 1: Find row-wise max (over KV dimension)
    // Use ggml_reduce_max or implement via comparison
    ggml_tensor * m = ggml_amax(ctx, logits_f32, /* dim=0 */ nullptr);  // [1, n_tokens, n_head, n_stream]
    m = ggml_reshape_3d(ctx, m, logits_f32->ne[1], logits_f32->ne[2], logits_f32->ne[3]);

    // Step 2: Subtract max for numerical stability
    ggml_tensor * m_broadcast = ggml_repeat(ctx, m, logits_f32);
    ggml_tensor * logits_shifted = ggml_sub(ctx, logits_f32, m_broadcast);

    // Step 3: Compute exp and sum
    ggml_tensor * exp_logits = ggml_exp(ctx, logits_shifted);
    ggml_tensor * l = ggml_sum_rows(ctx, exp_logits);

    return {m, l};
}
```

**Note**: GGML may not have `ggml_amax` for row-wise max. Alternative using `ggml_soft_max`:
```cpp
// Trick: softmax with very high temperature approximates argmax
// Better approach: use ggml custom op or implement per-row max via comparison loop
```

### Change 2: Update State Structure (semantic change)

**File**: `src/llama-graph.h` (lines 606-611)

```cpp
// Current state (semantics unclear):
struct h2o_online_softmax_state {
    ggml_tensor * m = nullptr; // Comment says "running max/logsumexp proxy (optional)"
    ggml_tensor * l = nullptr; // [n_tokens, n_head] running sum exp(logit - m)
    ggml_tensor * o = nullptr; // [n_embd_head_v, n_tokens, n_head] running weighted sum
    bool initialized = false;
};

// Proposed (clear semantics matching flash-attention):
struct h2o_online_softmax_state {
    ggml_tensor * m = nullptr; // [n_tokens, n_head] row-wise max of logits
    ggml_tensor * l = nullptr; // [n_tokens, n_head] Σexp(logit - m)
    ggml_tensor * o = nullptr; // [n_embd_head_v, n_tokens, n_head] Σ(v * softmax_weight)
    bool initialized = false;
    // NOTE: o stores NORMALIZED weighted sum (like flash-attn), not o*l
};
```

### Change 3: Fix init_online_softmax_state_h2o

**File**: `src/llama-graph.cpp` (lines 2153-2188)

```cpp
void llm_graph_context::init_online_softmax_state_h2o(
        ggml_tensor * inter_logits,
        ggml_tensor * v_mem,
        llm_graph_result::h2o_online_softmax_state & state) const {

    // NEW: Get max and normalized sum
    auto [m_inter, l_inter] = h2o_logits_sum_exp_with_max(ctx0, inter_logits);

    // Compute softmax weights (already done correctly in original via ggml_soft_max_ext)
    ggml_tensor * kq = ggml_soft_max_ext(ctx0, inter_logits, nullptr, 1.0f, 0.0f);

    // Compute weighted output kqv (same as original)
    // ... (existing v permutation and mul_mat code)

    // NEW: Store normalized output, not o*l
    state.m = m_inter;           // NEW: track max
    state.l = l_inter;           // Changed: now stores Σexp(s-m), not Σexp(s)
    state.o = kqv;               // Changed: stores normalized output, not kqv*l
    state.initialized = true;
}
```

### Change 4: Fix update_online_softmax_state_h2o (Critical)

**File**: `src/llama-graph.cpp` (lines 2190-2236)

```cpp
ggml_tensor * llm_graph_context::update_online_softmax_state_h2o(
        llm_graph_result::h2o_online_softmax_state & state,
        ggml_tensor * intra_logits,
        ggml_tensor * v_intra) const {

    // Step 1: Get intra statistics with max
    auto [m_intra, l_intra] = h2o_logits_sum_exp_with_max(ctx0, intra_logits);

    // Step 2: Compute intra softmax output
    ggml_tensor * kq_intra = ggml_soft_max_ext(ctx0, intra_logits, nullptr, 1.0f, 0.0f);
    ggml_tensor * kqv_intra = /* mul_mat with v_intra */;

    if (!state.initialized) {
        // First block - just store
        state.m = m_intra;
        state.l = l_intra;
        state.o = kqv_intra;
        state.initialized = true;
        return kqv_intra;
    }

    // Step 3: Find new global max (element-wise max)
    ggml_tensor * m_new = ggml_maximum(ctx0, state.m, m_intra);

    // Step 4: Compute rescaling factors
    // scale_old = exp(m_old - m_new), scale_new = exp(m_intra - m_new)
    ggml_tensor * scale_old = ggml_exp(ctx0, ggml_sub(ctx0, state.m, m_new));
    ggml_tensor * scale_new = ggml_exp(ctx0, ggml_sub(ctx0, m_intra, m_new));

    // Step 5: Rescale and combine l values
    // l_total = l_old * scale_old + l_intra * scale_new
    ggml_tensor * l_old_scaled = ggml_mul(ctx0, state.l, scale_old);
    ggml_tensor * l_intra_scaled = ggml_mul(ctx0, l_intra, scale_new);
    ggml_tensor * l_total = ggml_add(ctx0, l_old_scaled, l_intra_scaled);

    // Step 6: Rescale and combine o values (with proper broadcasting)
    // o_total = (o_old * l_old * scale_old + o_intra * l_intra * scale_new) / l_total
    ggml_tensor * scale_old_bc = ggml_repeat(ctx0, scale_old, state.o);
    ggml_tensor * scale_new_bc = ggml_repeat(ctx0, scale_new, kqv_intra);
    ggml_tensor * l_old_bc = ggml_repeat(ctx0, state.l, state.o);
    ggml_tensor * l_intra_bc = ggml_repeat(ctx0, l_intra, kqv_intra);

    ggml_tensor * o_old_contrib = ggml_mul(ctx0, ggml_mul(ctx0, state.o, l_old_bc), scale_old_bc);
    ggml_tensor * o_new_contrib = ggml_mul(ctx0, ggml_mul(ctx0, kqv_intra, l_intra_bc), scale_new_bc);
    ggml_tensor * o_sum = ggml_add(ctx0, o_old_contrib, o_new_contrib);

    ggml_tensor * l_total_bc = ggml_repeat(ctx0, l_total, o_sum);
    ggml_tensor * o_total = ggml_div(ctx0, o_sum, l_total_bc);

    // Update state
    state.m = m_new;
    state.l = l_total;
    state.o = o_total;  // Normalized output

    return o_total;
}
```

### Change 5: Update h2o_prefill_cache to Store m Values

**File**: `src/llama-context.h` (lines 35-45)

```cpp
struct h2o_prefill_cache {
    // Per-layer cached intra attention results
    std::vector<std::vector<float>> intra_o;  // [il] -> [n_embd_head_v * n_tokens * n_head]
    std::vector<std::vector<float>> intra_l;  // [il] -> [n_tokens * n_head]
    std::vector<std::vector<float>> intra_m;  // NEW: [il] -> [n_tokens * n_head] row-wise max

    uint32_t n_tokens = 0;
    uint32_t n_head = 0;
    uint32_t n_embd_head_v = 0;
    uint32_t n_stream = 1;
    llama_pos base_pos = 0;
    bool initialized = false;
};
```

---

## GGML Operation Availability Analysis

### Required Operations for Fix

| Operation | Available in GGML? | Notes |
|-----------|-------------------|-------|
| `ggml_sum_rows` | Yes | Sums along rows, returns `[1, b, c, d]` |
| `ggml_argmax` | Yes | Returns **indices** of max, not values |
| `ggml_max_rows` | **NO** | Need to implement |
| `ggml_exp` | Yes | Element-wise exponential |
| `ggml_sub` | Yes | Element-wise subtraction |
| `ggml_mul` | Yes | Element-wise multiplication |
| `ggml_repeat` | Yes | Broadcast smaller tensor to larger shape |

### Workaround for Missing `ggml_max_rows`

**Option 1: Use argmax + gather**
```cpp
// Get indices of max values
ggml_tensor * max_idx = ggml_argmax(ctx, logits);  // [1, n_tokens, n_head]
// Gather values at those indices
ggml_tensor * m = ggml_get_rows(ctx, logits, max_idx);
```
**Problem**: `ggml_get_rows` may not support this shape/indexing pattern.

**Option 2: Derive max from softmax (clever trick)**
```cpp
// softmax computes: exp(x - max) / sum(exp(x - max))
// sum(exp(x - max)) = sum(exp(x)) / exp(max)
// Therefore: max = log(sum_exp) - log(softmax_sum)

ggml_tensor * sum_exp = ggml_sum_rows(ctx, ggml_exp(ctx, logits));  // Σexp(x)
ggml_tensor * softmax = ggml_soft_max(ctx, logits);
ggml_tensor * softmax_sum = ggml_sum_rows(ctx, softmax);  // Should be ~1.0

// This doesn't work because softmax_sum ≈ 1 always!
```
**Problem**: Softmax normalizes so sum is always 1.

**Option 3: Add custom GGML operation (Recommended)**
```cpp
// In ggml.c, add:
GGML_API struct ggml_tensor * ggml_max_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);
```
This mirrors `ggml_sum_rows` but computes max instead of sum.

**Option 4: Compute iteratively in CPU callback (temporary)**
```cpp
// During graph computation, hook into custom callback
// Manually compute row-wise max on CPU
```

### Recommendation

For initial implementation, use **Option 3** (add `ggml_max_rows` to GGML). This is a simple modification following the pattern of `ggml_sum_rows`:

```cpp
// In ggml.c, similar to GGML_OP_SUM_ROWS
case GGML_OP_MAX_ROWS:
    // Implementation: max over dimension 0
```

---

## Verification Plan

1. **Unit Test**: Compare H2O output vs full attention for various max differences
2. **Numerical Test**: Stress test with logits in range [-100, 100]
3. **Perplexity Test**: Measure quality before/after fix
4. **Performance Test**: Measure overhead of additional max tracking

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `src/llama-graph.h` | Update `h2o_online_softmax_state` struct |
| `src/llama-graph.cpp` | Fix `init_online_softmax_state_h2o`, `update_online_softmax_state_h2o` |
| `src/llama-context.h` | Update `h2o_prefill_cache` to store m values |
| `src/llama-context.cpp` | Update cache read/write for new state format |

---

## Summary

The most critical finding is that the **current fusion formula is mathematically incorrect** when inter and intra attention blocks have different maximum logit values. While the clamping provides some protection, it does not guarantee correct results. The fix requires tracking row-wise max values and applying proper rescaling during fusion, following the same pattern used in flash-attention's online softmax algorithm.
