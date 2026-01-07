# Task 2: Intra-Chunk Causal Attention Implementation Plan

## Overview

Implement intra-chunk causal attention with correct RoPE positions for the chunked sparse attention pipeline. This task builds on Task 1 (H2O KV-cache data structures) and provides the foundation for Task 3 (score tracking) and Task 5 (inter-chunk attention with online softmax).

**Scope**: This task implements the H2O attention **functions only**. Model integration (modifying `models/llama.cpp` to call these functions) is deferred to Task 6.

## Key Insight

**Flash Attention cannot return attention weights.** For H2O score tracking, we need attention weights. The solution is to:
1. Use **standard (non-flash) attention path** when H2O mode is active
2. Extract attention weights after softmax for score initialization
3. RoPE positions are already global in llama.cpp (no changes needed for position encoding)

---

## Implementation Steps

### Step 1: Add H2O Attention MHA Function

**File**: `src/llama-graph.h` (after line 776)

Add new function declaration that returns attention weights:

```cpp
// H2O intra-chunk attention with weight extraction (uses standard path, not flash)
ggml_tensor * build_attn_mha_h2o(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * kq_b,
        ggml_tensor * kq_mask,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
              float   kq_scale,
                int   il,
        ggml_tensor ** attn_weights_out) const;  // OUT: attention weights [n_kv, n_tokens, n_head]
```

---

### Step 2: Implement H2O Attention MHA

**File**: `src/llama-graph.cpp` (after line 1611, after `build_attn_mha`)

```cpp
ggml_tensor * llm_graph_context::build_attn_mha_h2o(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il,
         ggml_tensor ** attn_weights_out) const {

    // Force standard attention path to extract weights
    // Based on build_attn_mha lines 1543-1606

    const bool v_trans = v->nb[1] > v->nb[2];
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream,
                     q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    // QK^T
    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    cb(kq, "kq_h2o", il);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    // Optional: attention logit soft-capping
    if (hparams.attn_soft_cap) {
        kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
        kq = ggml_tanh(ctx0, kq);
        kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
    }

    if (kq_b) {
        kq = ggml_add(ctx0, kq, kq_b);
    }

    // Softmax with mask
    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
    ggml_soft_max_add_sinks(kq, sinks);
    cb(kq, "kq_soft_max_h2o", il);

    // *** OUTPUT ATTENTION WEIGHTS FOR H2O ***
    *attn_weights_out = kq;  // Shape: [n_kv, n_tokens, n_head, n_stream]

    // KQV
    if (!v_trans) {
        v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
    }

    ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
    cb(kqv, "kqv_h2o", il);

    // MLA decompression if needed
    if (v_mla) {
        kqv = ggml_mul_mat(ctx0, v_mla, kqv);
    }

    ggml_tensor * cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    if (!cparams.offload_kqv) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}
```

---

### Step 3: Add H2O Build Attention Wrapper

**File**: `src/llama-graph.h` (after the new MHA declaration)

```cpp
// H2O attention with KV cache storage and weight extraction
ggml_tensor * build_attn_h2o(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
              float   kq_scale,
                int   il,
        ggml_tensor ** attn_weights_out) const;
```

---

### Step 4: Implement H2O Build Attention Wrapper

**File**: `src/llama-graph.cpp` (after `build_attn_mha_h2o`)

```cpp
ggml_tensor * llm_graph_context::build_attn_h2o(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
              float   kq_scale,
                int   il,
        ggml_tensor ** attn_weights_out) const {

    // Expand nodes to prevent reordering
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // Store K, V in cache (same as standard path)
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // Use H2O MHA to get attention weights
    ggml_tensor * cur = build_attn_mha_h2o(q, k, v, kq_b, kq_mask, sinks, v_mla,
                                            kq_scale, il, attn_weights_out);
    cb(cur, "kqv_out_h2o", il);

    // Output projection
    if (wo) {
        cur = build_lora_mm(wo, cur);
    }
    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}
```

---

### Step 5: Add Attention Weight Column Sum Helper

**File**: `src/llama-graph.h` (near H2O functions)

```cpp
// Compute column sum of attention weights for H2O score tracking
// Input: attn_weights [n_kv, n_tokens, n_head, n_stream]
// Output: colsum [n_kv, n_head] (sum over n_tokens dimension)
ggml_tensor * build_attn_colsum(ggml_tensor * attn_weights, int il) const;
```

**File**: `src/llama-graph.cpp`

```cpp
ggml_tensor * llm_graph_context::build_attn_colsum(ggml_tensor * attn_weights, int il) const {
    // attn_weights shape: [n_kv, n_tokens, n_head, n_stream]
    // We want to sum over n_tokens (dim 1) to get per-key scores

    // Use ggml_sum_rows which sums over dim 0, so we need to permute first
    ggml_tensor * permuted = ggml_permute(ctx0, attn_weights, 1, 0, 2, 3); // [n_tokens, n_kv, n_head, n_stream]
    ggml_tensor * colsum = ggml_sum_rows(ctx0, permuted); // [1, n_kv, n_head, n_stream]
    colsum = ggml_reshape_3d(ctx0, colsum, attn_weights->ne[0], attn_weights->ne[2], attn_weights->ne[3]); // [n_kv, n_head, n_stream]

    cb(colsum, "h2o_attn_colsum", il);
    ggml_build_forward_expand(gf, colsum);

    return colsum;
}
```

---

### Step 6: Create Test File

**File**: `tests/test-h2o-intra-attn.cpp`

Comprehensive test covering:
1. **Numerical correctness**: H2O attention vs standard attention match
2. **RoPE position correctness**: Global positions verified at chunk boundaries
3. **Causal mask correctness**: No future token leakage
4. **Attention weights correctness**: Row sums equal 1, column sums computed correctly
5. **KV cache population**: Correct global positions

---

## Verification Metrics and Experiments

### Metric 1: Numerical Correctness

**Test**: Compare H2O intra-attention output vs standard `build_attn_mha` (non-flash path)

```cpp
void test_numerical_correctness() {
    // Setup identical Q, K, V, mask
    // Run standard attention
    auto output_std = build_attn_mha(q, k, v, nullptr, mask, nullptr, nullptr, scale, 0);

    // Run H2O attention
    ggml_tensor * attn_weights = nullptr;
    auto output_h2o = build_attn_mha_h2o(q, k, v, nullptr, mask, nullptr, nullptr, scale, 0, &attn_weights);

    // Compare outputs
    float max_diff = compute_max_diff(output_std, output_h2o);
    ASSERT(max_diff < 1e-5f);  // FP32 tolerance
}
```
**Target**: Max difference < 1e-5

---

### Metric 2: RoPE Position Correctness

**Test**: Verify positions are global, not chunk-relative

```cpp
void test_rope_global_positions() {
    // Process chunk 0: positions [0, 1, ..., S-1]
    // Process chunk 1: positions [S, S+1, ..., 2S-1]

    // Verify inp_pos tensor values
    for (uint32_t c = 0; c < num_chunks; c++) {
        uint32_t chunk_start = c * S;
        for (uint32_t i = 0; i < S; i++) {
            llama_pos expected_pos = chunk_start + i;
            llama_pos actual_pos = ubatch.pos[i];
            ASSERT(actual_pos == expected_pos);
        }
    }
}
```
**Target**: All positions match expected global values

---

### Metric 3: Causal Mask Correctness

**Test**: Verify no future token leakage in attention weights

```cpp
void test_causal_mask() {
    ggml_tensor * attn_weights = nullptr;
    auto output = build_attn_mha_h2o(..., &attn_weights);

    // attn_weights: [n_kv, n_tokens, n_head, ...]
    float * weights = (float *)attn_weights->data;

    for (int q_pos = 0; q_pos < n_tokens; q_pos++) {
        for (int k_pos = q_pos + 1; k_pos < n_kv; k_pos++) {
            // Future positions should have zero weight (masked)
            float w = weights[...];
            ASSERT(w == 0.0f || fabs(w) < 1e-9f);
        }
    }
}
```
**Target**: All future positions have zero attention weight

---

### Metric 4: Attention Weight Row Sum

**Test**: Softmax normalization - each query row sums to 1

```cpp
void test_attn_row_sum() {
    ggml_tensor * attn_weights = nullptr;
    build_attn_mha_h2o(..., &attn_weights);

    float * weights = (float *)attn_weights->data;

    for (int h = 0; h < n_head; h++) {
        for (int q = 0; q < n_tokens; q++) {
            float row_sum = 0.0f;
            for (int k = 0; k < n_kv; k++) {
                row_sum += weights[k + q * n_kv + h * n_kv * n_tokens];
            }
            ASSERT(fabs(row_sum - 1.0f) < 1e-5f);
        }
    }
}
```
**Target**: All row sums within 1e-5 of 1.0

---

### Metric 5: Column Sum Correctness

**Test**: Column sum matches manual computation

```cpp
void test_colsum_correctness() {
    ggml_tensor * attn_weights = nullptr;
    build_attn_mha_h2o(..., &attn_weights);

    // Manual column sum
    std::vector<float> expected_colsum(n_kv * n_head, 0.0f);
    float * weights = (float *)attn_weights->data;
    for (int h = 0; h < n_head; h++) {
        for (int k = 0; k < n_kv; k++) {
            for (int q = 0; q < n_tokens; q++) {
                expected_colsum[k + h * n_kv] += weights[k + q * n_kv + h * n_kv * n_tokens];
            }
        }
    }

    // Computed column sum
    ggml_tensor * colsum = build_attn_colsum(attn_weights, 0);
    float * computed = (float *)colsum->data;

    for (int i = 0; i < n_kv * n_head; i++) {
        ASSERT(fabs(expected_colsum[i] - computed[i]) < 1e-4f);
    }
}
```
**Target**: Max difference < 1e-4

---

### Metric 6: KV Cache Population

**Test**: Verify K, V written at correct global positions

```cpp
void test_kv_cache_positions() {
    // After processing chunk c at positions [cS, cS+1, ..., (c+1)S-1]
    // Verify KV cache cells at those positions are populated

    for (uint32_t c = 0; c < num_chunks; c++) {
        process_chunk(c);

        uint32_t chunk_start = c * S;
        uint32_t chunk_end = (c + 1) * S;

        for (uint32_t pos = chunk_start; pos < chunk_end; pos++) {
            // Check cell is not empty
            ASSERT(!kv_cache.cells[pos].is_empty());
            // Check position matches
            ASSERT(kv_cache.cells[pos].pos == pos);
        }
    }
}
```
**Target**: All positions correctly populated

---

### Metric 7: Performance Comparison

**Benchmark**: Compare H2O attention vs Flash attention latency

```cpp
void benchmark_h2o_vs_flash() {
    const int num_runs = 100;

    // Flash attention timing
    auto start_flash = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        build_attn_mha(q, k, v, nullptr, mask, nullptr, nullptr, scale, 0);
    }
    auto end_flash = std::chrono::high_resolution_clock::now();

    // H2O attention timing
    auto start_h2o = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        ggml_tensor * weights = nullptr;
        build_attn_mha_h2o(q, k, v, nullptr, mask, nullptr, nullptr, scale, 0, &weights);
    }
    auto end_h2o = std::chrono::high_resolution_clock::now();

    // Report overhead
    float flash_ms = duration_cast<milliseconds>(end_flash - start_flash).count() / (float)num_runs;
    float h2o_ms = duration_cast<milliseconds>(end_h2o - start_h2o).count() / (float)num_runs;
    printf("Flash: %.2f ms, H2O: %.2f ms, Overhead: %.1f%%\n", flash_ms, h2o_ms,
           100.0f * (h2o_ms - flash_ms) / flash_ms);
}
```
**Target**: H2O overhead < 50% vs Flash attention (acceptable for prefill phase)

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/llama-graph.h` | Add `build_attn_mha_h2o`, `build_attn_h2o`, `build_attn_colsum` declarations |
| `src/llama-graph.cpp` | Implement the three new functions |
| `tests/test-h2o-intra-attn.cpp` | New test file with all verification metrics |
| `tests/CMakeLists.txt` | Register new test |

---

## Dependencies

### From Task 1 (Complete)
- `h2o_init_chunk_scores()` - Used to store initial scores from column sums
- `h2o_get_scores_tensor()` - Access to score storage
- `LLAMA_SWA_TYPE_H2O` - SWA type check

### To Task 3
- `attn_weights_out` tensor - Passed to score tracking
- `build_attn_colsum()` - Computes column sums for score initialization

### To Task 5
- `build_attn_h2o()` - Reused for intra-chunk attention in fused pipeline
- KV cache at global positions - Required for inter-chunk attention

---

## Success Criteria

- [ ] All 7 verification metrics pass
- [ ] H2O attention produces identical output to standard attention
- [ ] RoPE uses global positions (verified at chunk boundaries)
- [ ] Causal mask prevents future token leakage
- [ ] Attention weights correctly normalized (rows sum to 1)
- [ ] Column sums computed accurately for H2O scoring
- [ ] KV cache populated at correct global positions
- [ ] Compiles on CPU and CUDA backends
- [ ] Performance overhead acceptable (< 50% vs Flash)
