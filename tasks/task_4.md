# Task 4: Memory Set Selection Integration for H2O Chunked Sparse Attention

## Overview

This task integrates the H2O memory set selection mechanism into the main prefill execution path. The **core implementation already exists** from Task 1 (in `llama-kv-cache.cpp`), but it is not yet called during inference. This task wires `h2o_build_memory_set` into the chunk processing loop so that memory sets (Local + Heavy hitters) are constructed and ready for inter-chunk attention in Task 5.

**Scope**: Integration and validation only. Inter-chunk attention and online softmax fusion are deferred to Task 5.

**Dependencies**:
- ✅ Task 1: H2O data structures (`h2o_scores`, `h2o_memory_indices`, `h2o_build_memory_set`)
- ✅ Task 2: Intra-chunk attention with weight extraction
- ✅ Task 3: Score tracking integration (`h2o_init_chunk_scores`, `h2o_next_chunk`)

---

## Current Status Analysis

### Already Implemented (Task 1)

**File**: `src/llama-kv-cache.cpp:1118`

```cpp
void llama_kv_cache::h2o_build_memory_set(int32_t il, uint32_t chunk_end);
```

This function implements:
1. ✅ Local window extraction (tail L tokens)
2. ✅ Candidate set construction (previous memory + current chunk - local)
3. ✅ Heavy hitter selection (top-H by score, per-head, per-layer)
4. ✅ Memory set assembly (Local + Heavy, sorted)
5. ✅ Cross-chunk propagation (tokens can survive multiple chunks)

**Also implemented**:
- `h2o_gather_k_memory(ctx, il)` - Gather K for memory set
- `h2o_gather_v_memory(ctx, il)` - Gather V for memory set

### Current Execution Flow (Task 3)

**File**: `src/llama-context.cpp:1612-1636`

```cpp
for (uint32_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
    // ... extract chunk_colsum from t_h2o_colsum ...

    kv->h2o_init_chunk_scores(layer.il, chunk_start, chunk_len, chunk_colsum.data());

    kv->h2o_next_chunk(chunk_len);  // ← Task 4: Add h2o_build_memory_set HERE
}
```

**What's missing**: `h2o_build_memory_set` is never called!

---

## Implementation Steps

### Step 1: Add Memory Set Construction to Chunk Loop

**File**: `src/llama-context.cpp` (around line 1635)

**Modification**: Add `h2o_build_memory_set` call after score initialization for each layer, before `h2o_next_chunk`.

```cpp
for (uint32_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    const uint32_t chunk_len = std::min(chunk_size, n_tokens - chunk_offset);
    const uint32_t chunk_start = base_tokens + chunk_offset;
    const uint32_t chunk_end = chunk_start + chunk_len;

    // Initialize scores from intra-attention colsum
    for (const auto & layer : layers) {
        // ... existing code to extract and set chunk_colsum ...

        kv->h2o_init_chunk_scores(layer.il, chunk_start, chunk_len, chunk_colsum.data());

        // *** NEW: Build memory set for next chunk ***
        kv->h2o_build_memory_set(layer.il, chunk_end);
    }

    kv->h2o_next_chunk(chunk_len);
}
```

**Key details**:
- Call `h2o_build_memory_set` **once per layer** after initializing scores
- Pass `chunk_end` (not `chunk_len`) - it needs the global end position
- This prepares memory set M_c for the **next** chunk C_{c+1}
- For chunk 0, this builds M_0 for chunk 1
- For the final chunk, memory set is built but won't be used (OK for now, Task 5 will use it)

---

### Step 2: Add Debug Logging (Optional but Recommended)

**File**: `src/llama-context.cpp`

Add logging to track memory set construction:

```cpp
if (kv->debug >= 1) {
    LLAMA_LOG_DEBUG("%s: [H2O] chunk %u: built memory set for layer %d (chunk_end=%u)\n",
                    __func__, chunk_idx, layer.il, chunk_end);
}
```

**File**: `src/llama-kv-cache.cpp:1184` (end of `h2o_build_memory_set`)

Add logging to show memory set stats:

```cpp
if (debug >= 2) {
    LLAMA_LOG_DEBUG("%s: [H2O] layer %d chunk_end=%u: memory set initialized (L=%u H=%u)\n",
                    __func__, il, chunk_end, L, H);
}
```

---

### Step 3: Verify Memory Set Initialization Flag

**File**: `src/llama-kv-cache.cpp:1184`

**Current code** already sets the flag:

```cpp
h2o_memory_initialized = true;
```

**Verification**: Ensure this flag is checked before calling gather methods:

```cpp
ggml_tensor * llama_kv_cache::h2o_gather_k_memory(...) const {
    GGML_ASSERT(h2o_memory_initialized);  // ✅ Already present
    // ...
}
```

---

## Validation & Verification Metrics

### Metric 1: Local Window Correctness

**Test**: Verify local window contains exactly the tail L tokens of each chunk

**File**: `tests/test-h2o-memory-selection.cpp` (new)

```cpp
void test_local_window_correctness() {
    // Process chunk 0 (positions 0..S-1)
    kv.h2o_init_chunk_scores(il, 0, S, intra_colsum.data());
    kv.h2o_build_memory_set(il, S);

    // Verify local window = {S-L, S-L+1, ..., S-1}
    const int32_t * mem_idx = get_memory_indices(kv, il, head=0);
    std::vector<int32_t> local_expected;
    for (uint32_t pos = S - L; pos < S; ++pos) {
        local_expected.push_back(pos);
    }

    // First L indices should be local window (sorted)
    for (uint32_t m = 0; m < L; ++m) {
        ASSERT_EQ(mem_idx[m], local_expected[m]);
    }
}
```

**Target**: Exact match for all heads, all layers, all chunks

---

### Metric 2: Heavy Hitter Selection Correctness

**Test**: Top-H tokens by score are selected from candidates

```cpp
void test_heavy_hitter_selection() {
    // Set known scores for chunk 0
    std::vector<float> known_scores(S);
    for (uint32_t i = 0; i < S; i++) {
        known_scores[i] = (i % 10) * 0.1f;  // Known pattern
    }

    kv.h2o_init_chunk_scores(il, 0, S, known_scores.data());
    kv.h2o_build_memory_set(il, S);

    // Candidates = C0[0:S-L] (exclude local window)
    // Expected heavy hitters: top-H by score from candidates
    std::vector<std::pair<int, float>> candidates;
    for (uint32_t pos = 0; pos < S - L; ++pos) {
        candidates.push_back({pos, known_scores[pos]});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](auto &a, auto &b) { return a.second > b.second; });

    std::set<int> expected_heavy;
    for (uint32_t i = 0; i < H; ++i) {
        expected_heavy.insert(candidates[i].first);
    }

    // Verify actual heavy hitters match
    const int32_t * mem_idx = get_memory_indices(kv, il, 0);
    std::set<int> actual_heavy;
    for (uint32_t m = L; m < M; ++m) {  // Heavy region [L, M)
        if (mem_idx[m] >= 0) {
            actual_heavy.insert(mem_idx[m]);
        }
    }

    ASSERT_EQ(actual_heavy, expected_heavy);
}
```

**Target**: Exact top-H selection across all test cases

---

### Metric 3: Cross-Chunk Propagation

**Test**: High-score token from C0 survives into M1, M2, M3

```cpp
void test_cross_chunk_propagation() {
    const uint32_t important_token = 42;

    // Chunk 0: Set token 42 with very high score
    std::vector<float> chunk0_scores(S, 0.1f);
    chunk0_scores[important_token] = 10.0f;  // Very high

    kv.h2o_init_chunk_scores(il, 0, S, chunk0_scores.data());
    kv.h2o_build_memory_set(il, S);
    kv.h2o_next_chunk(S);

    // Verify token 42 is in M0
    ASSERT_TRUE(is_token_in_memory(kv, il, important_token));

    // Process chunk 1 (scores stay high via accumulation simulation)
    std::vector<float> chunk1_inter(M, 0.05f);
    chunk1_inter[find_memory_index(important_token)] = 1.0f;  // High inter-attn
    kv.h2o_accumulate_memory_scores(il, chunk1_inter.data());

    std::vector<float> chunk1_intra(S, 0.1f);
    kv.h2o_init_chunk_scores(il, S, S, chunk1_intra.data());
    kv.h2o_build_memory_set(il, 2*S);
    kv.h2o_next_chunk(S);

    // Verify token 42 STILL in M1 (propagated from C0 to C1)
    ASSERT_TRUE(is_token_in_memory(kv, il, important_token));

    // Repeat for chunk 2
    // ... (similar code)

    // Verify token 42 STILL in M2 (propagated across 3 chunks!)
    ASSERT_TRUE(is_token_in_memory(kv, il, important_token));
}
```

**Target**: Important tokens survive across at least 3 chunks

---

### Metric 4: Per-Head Independence

**Test**: Different heads select different heavy hitters

```cpp
void test_per_head_independence() {
    const uint32_t n_head = kv.hparams.n_head(il);

    // Set different score patterns per head
    std::vector<float> scores(n_head * S);
    for (uint32_t h = 0; h < n_head; ++h) {
        for (uint32_t pos = 0; pos < S; ++pos) {
            // Each head has unique pattern
            scores[h * S + pos] = std::sin(h * 1.5f + pos * 0.1f);
        }
    }

    kv.h2o_init_chunk_scores(il, 0, S, scores.data());
    kv.h2o_build_memory_set(il, S);

    // Extract memory sets per head
    std::vector<std::set<int>> memory_per_head(n_head);
    for (uint32_t h = 0; h < n_head; ++h) {
        const int32_t * mem_idx = get_memory_indices(kv, il, h);
        for (uint32_t m = 0; m < M; ++m) {
            if (mem_idx[m] >= 0) {
                memory_per_head[h].insert(mem_idx[m]);
            }
        }
    }

    // Count how many adjacent heads have different selections
    int different_count = 0;
    for (uint32_t h = 0; h < n_head - 1; ++h) {
        if (memory_per_head[h] != memory_per_head[h+1]) {
            different_count++;
        }
    }

    // At least 50% should differ (independence check)
    ASSERT_GT(different_count, n_head / 2);
}
```

**Target**: > 50% of adjacent head pairs have different memory sets

---

### Metric 5: Sorted Output

**Test**: Memory indices are in ascending order (for coalesced access)

```cpp
void test_sorted_indices() {
    kv.h2o_init_chunk_scores(il, 0, S, intra_colsum.data());
    kv.h2o_build_memory_set(il, S);

    const uint32_t n_head = kv.hparams.n_head(il);
    for (uint32_t h = 0; h < n_head; ++h) {
        const int32_t * mem_idx = get_memory_indices(kv, il, h);

        std::vector<int32_t> valid_indices;
        for (uint32_t m = 0; m < M; ++m) {
            if (mem_idx[m] >= 0) {
                valid_indices.push_back(mem_idx[m]);
            }
        }

        // Check strictly increasing
        for (size_t i = 1; i < valid_indices.size(); ++i) {
            ASSERT_GT(valid_indices[i], valid_indices[i-1]);
        }
    }
}
```

**Target**: All indices strictly ascending within each head

---

### Metric 6: Edge Case - Single Chunk (N = S)

**Test**: When sequence length equals chunk size

```cpp
void test_edge_case_single_chunk() {
    const uint32_t N = S;  // Exactly one chunk

    std::vector<float> scores(S, 0.5f);
    kv.h2o_init_chunk_scores(il, 0, S, scores.data());
    kv.h2o_build_memory_set(il, S);

    // Memory set should exist and be valid
    ASSERT_TRUE(kv.h2o_is_memory_initialized());

    // Local window should be tail L tokens
    // Heavy hitters from first S-L tokens
    verify_memory_set_structure(kv, il, S);
}
```

**Target**: No crashes, valid memory set constructed

---

### Metric 7: Edge Case - Partial Final Chunk (N = 2S - 1)

**Test**: Final chunk has less than S tokens

```cpp
void test_edge_case_partial_chunk() {
    const uint32_t N = 2 * S - 1;

    // Chunk 0 (full)
    kv.h2o_init_chunk_scores(il, 0, S, scores0.data());
    kv.h2o_build_memory_set(il, S);
    kv.h2o_next_chunk(S);

    // Chunk 1 (partial: S-1 tokens)
    const uint32_t chunk1_len = S - 1;
    kv.h2o_init_chunk_scores(il, S, chunk1_len, scores1.data());
    kv.h2o_build_memory_set(il, S + chunk1_len);

    // Should handle partial chunk gracefully
    ASSERT_TRUE(kv.h2o_is_memory_initialized());
    verify_no_oob_access(kv, il, S + chunk1_len);
}
```

**Target**: Correct behavior for partial chunks, no buffer overflow

---

### Metric 8: Edge Case - Very Small Sequence (N < L)

**Test**: Sequence shorter than local window size

```cpp
void test_edge_case_very_small_sequence() {
    const uint32_t N = L / 2;  // Only 128 tokens (L=256)

    std::vector<float> scores(N, 0.5f);
    kv.h2o_init_chunk_scores(il, 0, N, scores.data());
    kv.h2o_build_memory_set(il, N);

    // Local window should be ALL tokens (since N < L)
    // Heavy hitters set should be empty
    const int32_t * mem_idx = get_memory_indices(kv, il, 0);

    // Count valid indices
    int count = 0;
    for (uint32_t m = 0; m < M; ++m) {
        if (mem_idx[m] >= 0) count++;
    }

    ASSERT_EQ(count, N);  // Exactly N tokens in memory (all are "local")
}
```

**Target**: Graceful handling when N < L

---

### Metric 9: Memory Set Size Constraint

**Test**: Memory set never exceeds M = L + H

```cpp
void test_memory_size_constraint() {
    // Process multiple chunks
    for (uint32_t c = 0; c < 4; ++c) {
        uint32_t chunk_start = c * S;
        kv.h2o_init_chunk_scores(il, chunk_start, S, scores.data());
        kv.h2o_build_memory_set(il, chunk_start + S);
        kv.h2o_next_chunk(S);

        // Count valid indices
        const int32_t * mem_idx = get_memory_indices(kv, il, 0);
        int count = 0;
        for (uint32_t m = 0; m < M; ++m) {
            if (mem_idx[m] >= 0) count++;
        }

        // Should be exactly M (or less for early chunks)
        ASSERT_LE(count, M);
        if (chunk_start + S >= M) {
            ASSERT_EQ(count, M);  // Should be full after enough tokens
        }
    }
}
```

**Target**: Memory set size ≤ M always

---

### Metric 10: No Index Out-of-Bounds

**Test**: All memory indices point to valid KV cache positions

```cpp
void test_no_oob_indices() {
    const uint32_t n_chunks = 4;

    for (uint32_t c = 0; c < n_chunks; ++c) {
        uint32_t chunk_start = c * S;
        uint32_t chunk_end = (c + 1) * S;

        kv.h2o_init_chunk_scores(il, chunk_start, S, scores.data());
        kv.h2o_build_memory_set(il, chunk_end);
        kv.h2o_next_chunk(S);

        // Check all heads
        const uint32_t n_head = kv.hparams.n_head(il);
        for (uint32_t h = 0; h < n_head; ++h) {
            const int32_t * mem_idx = get_memory_indices(kv, il, h);

            for (uint32_t m = 0; m < M; ++m) {
                int32_t idx = mem_idx[m];
                if (idx >= 0) {
                    // Must be within processed tokens
                    ASSERT_GE(idx, 0);
                    ASSERT_LT(idx, chunk_end);
                }
            }
        }
    }
}
```

**Target**: No out-of-bounds indices in any scenario

---

## Integration Test: End-to-End Prefill

**Test**: Run actual model inference with H2O enabled

**File**: `tests/test-h2o-memory-selection.cpp`

```cpp
void test_integration_end_to_end() {
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_batch = 4096;
    cparams.n_ubatch = 1024;
    cparams.h2o_local_window = 256;
    cparams.h2o_heavy_budget = 256;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_new_context_with_model(model, cparams);

    // Create a 4096-token prompt
    std::vector<llama_token> tokens(4096);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = i % 1000 + 1;  // Valid token IDs
    }

    // Run prefill
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    int ret = llama_decode(ctx, batch);
    ASSERT_EQ(ret, 0);

    // Verify H2O state
    auto * memory = llama_get_memory(ctx);
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    ASSERT_NE(kv, nullptr);

    // Should have processed 4 chunks
    ASSERT_EQ(kv->h2o_get_chunk_idx(), 4);
    ASSERT_EQ(kv->h2o_get_total_tokens(), 4096);
    ASSERT_TRUE(kv->h2o_is_memory_initialized());

    // Memory sets should be built for all layers
    const uint32_t n_layer = model->hparams.n_layer;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (model->hparams.has_kv(il)) {
            const ggml_tensor * mem_idx = kv->h2o_get_memory_indices_tensor(il);
            ASSERT_NE(mem_idx, nullptr);
            ASSERT_NE(mem_idx->data, nullptr);
        }
    }

    llama_free(ctx);
    llama_free_model(model);
}
```

**Command to run**:
```bash
LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf \
./build/bin/llama-cli -m /models/Qwen_Qwen3-1.7B-Q8_0.gguf \
  -p "Hello world" -n 0 -b 4096 -ub 1024 \
  --h2o-local 256 --h2o-heavy 256 -st
```

**Target**: Clean execution, no crashes, correct chunk/token counts

---

## Performance Benchmarks

### Benchmark 1: Memory Set Construction Speed

```cpp
void benchmark_memory_set_construction() {
    const int num_runs = 1000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        kv.h2o_build_memory_set(il, (i % 4 + 1) * S);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_us = duration.count() / (float)num_runs;

    printf("Average h2o_build_memory_set time: %.2f μs\n", avg_us);
}
```

**Target**: < 1000 μs (1 ms) per layer per chunk

---

### Benchmark 2: End-to-End Prefill Throughput

```cpp
void benchmark_prefill_throughput() {
    // Compare H2O enabled vs disabled

    // Run 1: H2O disabled
    auto start1 = std::chrono::high_resolution_clock::now();
    llama_decode(ctx_baseline, batch);
    auto end1 = std::chrono::high_resolution_clock::now();
    float time_baseline = std::chrono::duration<float>(end1 - start1).count();

    // Run 2: H2O enabled
    auto start2 = std::chrono::high_resolution_clock::now();
    llama_decode(ctx_h2o, batch);
    auto end2 = std::chrono::high_resolution_clock::now();
    float time_h2o = std::chrono::duration<float>(end2 - start2).count();

    printf("Baseline: %.3f s (%.1f tok/s)\n", time_baseline, 4096/time_baseline);
    printf("H2O:      %.3f s (%.1f tok/s)\n", time_h2o, 4096/time_h2o);
    printf("Overhead: %.1f%%\n", 100.0f * (time_h2o - time_baseline) / time_baseline);
}
```

**Target**: < 20% overhead compared to baseline (memory set construction should be negligible compared to attention computation)

---

## Critical Files Summary

| File | Lines to Modify | Description |
|------|----------------|-------------|
| `src/llama-context.cpp` | ~1635 (after `h2o_next_chunk`) | Add `h2o_build_memory_set` call in chunk loop |
| `tests/test-h2o-memory-selection.cpp` | ~800 (new file) | Comprehensive tests for all metrics |
| `tests/CMakeLists.txt` | +3 | Register new test |
| `tasks/task_4.md` | New | This document |
| `tasks/verify_4.md` | New | Verification results |

---

## Deliverables

### Code Changes
- [ ] `src/llama-context.cpp`: Integrate `h2o_build_memory_set` into chunk loop
- [ ] Optional: Add debug logging for memory set construction

### Tests
- [ ] `tests/test-h2o-memory-selection.cpp`: 10 correctness metrics
- [ ] Integration test: End-to-end prefill with H2O
- [ ] 2 performance benchmarks

### Documentation
- [ ] `tasks/task_4.md`: This implementation plan
- [ ] `tasks/verify_4.md`: Test results and verification

---

## Suggested Build & Verify Commands

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j12

# Run unit tests
LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf \
ctest --test-dir build -R test-h2o-memory-selection -V --output-on-failure

# Run integration test (manual)
./build/bin/llama-cli \
  -m /models/Qwen_Qwen3-1.7B-Q8_0.gguf \
  -p "The quick brown fox jumps over the lazy dog. " \
  -n 0 -b 4096 -ub 1024 \
  --h2o-local 256 --h2o-heavy 256 -st

# Expected output:
# - 4 chunks processed
# - Memory sets built for all layers
# - Total tokens = prompt length
# - No crashes or assertions
```

---

## Success Criteria

✅ **Integration**:
- [ ] `h2o_build_memory_set` called after each chunk in main execution path
- [ ] Memory sets built for all KV layers
- [ ] No crashes or assertion failures

✅ **Correctness** (10 metrics):
- [ ] Metric 1: Local window correctness
- [ ] Metric 2: Heavy hitter selection
- [ ] Metric 3: Cross-chunk propagation (3+ chunks)
- [ ] Metric 4: Per-head independence (>50% different)
- [ ] Metric 5: Sorted indices
- [ ] Metric 6: Edge case - single chunk (N=S)
- [ ] Metric 7: Edge case - partial chunk (N=2S-1)
- [ ] Metric 8: Edge case - very small sequence (N<L)
- [ ] Metric 9: Memory size constraint (≤M)
- [ ] Metric 10: No out-of-bounds indices

✅ **Performance**:
- [ ] Memory set construction < 1ms per layer
- [ ] End-to-end overhead < 20% vs baseline

✅ **Integration Test**:
- [ ] End-to-end prefill with 4096 tokens, 1024 ubatch
- [ ] Correct chunk count and token count
- [ ] Memory sets initialized for all layers

---

## Next Steps After Task 4

**Task 5**: Inter-Chunk Attention + Online Softmax Fusion
- Use `h2o_gather_k_memory` / `h2o_gather_v_memory` to get memory set KV
- Implement inter-chunk attention (Q_c × K_M_{c-1})
- Fuse with intra-chunk attention using online softmax
- Update scores via `h2o_accumulate_memory_scores`

**Task 6**: Full Pipeline Integration
- Integrate inter-chunk + intra-chunk attention into model graph
- Handle all edge cases (chunk 0, final chunk, etc.)

**Task 7**: CUDA Memory Optimization
- Optimize KV gathering for coalesced access
- Kernel fusion opportunities

**Task 8**: Benchmarking & Quality Validation
- Throughput vs sequence length
- Perplexity evaluation
- Attention pattern visualization

---

## Notes

- Inter-chunk attention is **NOT** part of Task 4 (deferred to Task 5)
- Memory sets are built but not yet used (will be used in Task 5)
- Focus on correctness and integration, not performance optimization
- All core algorithms already exist from Task 1 - this is just wiring them up
