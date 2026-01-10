# Task 1: Core Data Structures and KV Cache Management for Chunked Sparse Attention with H2O

## Overview

Implement the foundational data structures for chunked sparse attention with H2O-based heavy hitter selection in llama.cpp. This task extends the existing `llama_kv_cache` infrastructure to support:

1. **Per-token importance scores** (BF16, per-layer, per-head) for heavy hitter selection
2. **Memory set indices** tracking Local + Heavy hitter tokens for cross-chunk propagation
3. **Chunk metadata** for boundary tracking and progress monitoring
4. **New SWA type**: `LLAMA_SWA_TYPE_H2O = 4` integrated into existing architecture

**Scope Clarification**:
- H2O sparse attention is applied **only during prefill** (processing user input)
- **Decode phase** (generating output tokens) uses **standard full attention** with all KV stored
- This design choice simplifies implementation and ensures generation quality

**Design Decisions** (from user input):
- ✅ Extend existing `llama_kv_cache` structures (not create new classes)
- ✅ Use BF16 (GGML_TYPE_BF16) for score accumulation
- ✅ Support CPU + CUDA backends initially
- ✅ Add new SWA type `LLAMA_SWA_TYPE_H2O` to enum

---

## Implementation Steps

### Step 1: Add H2O SWA Type to Enum

**File**: `/root/workspace/Prism/src/llama-hparams.h` (line 18-23)

```cpp
enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE      = 0,
    LLAMA_SWA_TYPE_STANDARD  = 1,
    LLAMA_SWA_TYPE_CHUNKED   = 2,
    LLAMA_SWA_TYPE_SYMMETRIC = 3,
    LLAMA_SWA_TYPE_H2O       = 4,  // NEW: Chunked sparse attention with H2O selection
};
```

**Verification**:
- Compile check: `make clean && make llama-cli`
- Enum value check: Assert `LLAMA_SWA_TYPE_H2O == 4`

---

### Step 2: Extend llama_kv_cache Private Fields

**File**: `/root/workspace/Prism/src/llama-kv-cache.h` (after line 250)

Add H2O-specific state:

```cpp
// ========== H2O-specific state (LLAMA_SWA_TYPE_H2O only) ==========

// H2O hyperparameters
const uint32_t h2o_local_window = 0;  // L: local window size (e.g., 256)
const uint32_t h2o_heavy_budget = 0;  // H: heavy hitter budget (e.g., 256)
const uint32_t h2o_memory_size  = 0;  // M = L + H (must satisfy M < n_ubatch)

// Score tracking: accumulated attention scores for heavy hitter selection
// Shape per tensor: [kv_size, n_head] (2D, one per layer)
// Type: GGML_TYPE_BF16
std::vector<ggml_tensor *> h2o_scores;

// Memory set indices: positions of Local + Heavy hitters
// Shape per tensor: [M, n_head] (2D, one per layer)
// Type: GGML_TYPE_I32
std::vector<ggml_tensor *> h2o_memory_indices;

// Chunk metadata
struct h2o_chunk_state {
    uint32_t current_chunk_idx;
    uint32_t total_tokens_processed;
    std::vector<uint32_t> chunk_boundaries;  // [0, S, 2S, 3S, ...]
} h2o_state;

bool h2o_memory_initialized = false;
```

**Memory Layout** (matches llama.cpp implementation):
```
h2o_scores[il]:
  Type: BF16 (2 bytes per element)
  Shape: [kv_size, n_head] where kv_size = max_seq_len, n_head = num_heads
  Stride: ne[0] = kv_size, ne[1] = n_head
  Access: score_data[head * kv_size + pos]
  
  Note: Conceptually [num_heads, max_seq_len] per layer, but stored as [kv_size, n_head]
        for contiguous memory layout per position.

h2o_memory_indices[il]:
  Type: I32 (4 bytes per element)
  Shape: [M, n_head] where M = h2o_memory_size = L + H
  Stride: ne[0] = M, ne[1] = n_head
  Access: mem_idx_data[head * M + m]
  Layout: Per head, indices [0..L-1] = local window, [L..M-1] = heavy hitters
          All indices within each head are sorted for coalesced memory access.
```

**Verification**:
- Static size check: `sizeof(h2o_chunk_state) < 128 bytes`
- Vector capacity pre-allocation check

---

### Step 3: Extend Constructor Signature

**File**: `/root/workspace/Prism/src/llama-kv-cache.h` (line 96-109)

```cpp
llama_kv_cache(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
                 uint32_t   h2o_local = 0,  // NEW
                 uint32_t   h2o_heavy = 0); // NEW
```

**Verification**:
- Backward compatibility: Old calls compile without new params (default = 0)
- New calls: `llama_kv_cache(..., LLAMA_SWA_TYPE_H2O, ..., 256, 256)`

---

### Step 4: Implement H2O Tensor Allocation in Constructor

**File**: `/root/workspace/Prism/src/llama-kv-cache.cpp` (after line 170 in constructor)

```cpp
// Allocate H2O tensors if enabled
if (swa_type == LLAMA_SWA_TYPE_H2O && h2o_local + h2o_heavy > 0) {
    const uint32_t M = h2o_local + h2o_heavy;

    LLAMA_LOG_INFO("%s: allocating H2O tensors (L=%d, H=%d, M=%d)\n",
                   __func__, h2o_local, h2o_heavy, M);

    auto ctx_h2o = ctx_for_buft(ggml_backend_cpu_buffer_type());

    h2o_scores.reserve(layers.size());
    h2o_memory_indices.reserve(layers.size());

    for (const auto & layer : layers) {
        uint32_t il = layer.il;
        uint32_t n_head = hparams.n_head(il);

        // Score tensor: [kv_size, n_head] BF16
        ggml_tensor * scores = ggml_new_tensor_2d(ctx_h2o, GGML_TYPE_BF16,
                                                   kv_size, n_head);
        ggml_format_name(scores, "h2o_scores_l%d", il);
        h2o_scores.push_back(scores);

        // Index tensor: [M, n_head] I32
        ggml_tensor * mem_idx = ggml_new_tensor_2d(ctx_h2o, GGML_TYPE_I32,
                                                    M, n_head);
        ggml_format_name(mem_idx, "h2o_mem_idx_l%d", il);
        h2o_memory_indices.push_back(mem_idx);
    }

    // Allocate buffer and clear to zero
    ggml_backend_buffer_t buf_h2o = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_h2o, ggml_backend_cpu_buffer_type());
    if (!buf_h2o) {
        throw std::runtime_error("failed to allocate H2O buffers");
    }
    ggml_backend_buffer_clear(buf_h2o, 0);

    LLAMA_LOG_INFO("%s: H2O buffer size = %8.2f MiB\n",
                   __func__, ggml_backend_buffer_get_size(buf_h2o)/1024.0/1024.0);

    ctxs_bufs.emplace_back(std::move(ctx_h2o), buf_h2o);

    // Initialize chunk state
    h2o_state.current_chunk_idx = 0;
    h2o_state.total_tokens_processed = 0;
    h2o_state.chunk_boundaries.push_back(0);
}
```

**Verification Metrics**:
- Memory allocation success: No exception thrown
- Buffer clearing: First 100 bytes == 0
- Tensor naming: `ggml_get_name(h2o_scores[0]) == "h2o_scores_l0"`
- Context storage: `ctxs_bufs.size()` increments by 1

---

### Step 5: Add H2O Public API Methods

**File**: `/root/workspace/Prism/src/llama-kv-cache.h` (after line 201)

```cpp
//
// H2O-specific API (valid when swa_type == LLAMA_SWA_TYPE_H2O)
//

// Score management
void h2o_init_chunk_scores(
    int32_t il, uint32_t chunk_start, uint32_t chunk_len,
    const float * attn_weights_colsum);

void h2o_accumulate_memory_scores(
    int32_t il, const float * inter_weights_colsum);

// Memory set construction
void h2o_build_memory_set(int32_t il, uint32_t chunk_end);

// KV gathering for memory set
ggml_tensor * h2o_gather_k_memory(ggml_context * ctx, int32_t il) const;
ggml_tensor * h2o_gather_v_memory(ggml_context * ctx, int32_t il) const;

// Chunk metadata
void h2o_next_chunk(uint32_t chunk_len);
uint32_t h2o_get_chunk_idx() const { return h2o_state.current_chunk_idx; }
uint32_t h2o_get_total_tokens() const { return h2o_state.total_tokens_processed; }
bool h2o_is_memory_initialized() const { return h2o_memory_initialized; }
```

---

### Step 6: Implement Core H2O Methods

**File**: `/root/workspace/Prism/src/llama-kv-cache.cpp` (new methods at end of file)

#### 6.1 Score Initialization

```cpp
void llama_kv_cache::h2o_init_chunk_scores(
        int32_t il, uint32_t chunk_start, uint32_t chunk_len,
        const float * attn_weights_colsum) {

    GGML_ASSERT(swa_type == LLAMA_SWA_TYPE_H2O);
    GGML_ASSERT(map_layer_ids.find(il) != map_layer_ids.end());

    const int32_t il_cache = map_layer_ids[il];
    ggml_tensor * scores = h2o_scores[il_cache];
    const uint32_t n_head = hparams.n_head(il);

    ggml_bf16_t * score_data = (ggml_bf16_t *)scores->data;

    // attn_weights_colsum: [n_head, chunk_len]
    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t pos = 0; pos < chunk_len; ++pos) {
            uint32_t global_pos = chunk_start + pos;
            size_t score_idx = ih * scores->ne[0] + global_pos;
            score_data[score_idx] = ggml_fp32_to_bf16(
                attn_weights_colsum[ih * chunk_len + pos]);
        }
    }
}
```

#### 6.2 Score Accumulation

```cpp
void llama_kv_cache::h2o_accumulate_memory_scores(
        int32_t il, const float * inter_weights_colsum) {

    GGML_ASSERT(swa_type == LLAMA_SWA_TYPE_H2O);
    GGML_ASSERT(h2o_memory_initialized);

    const int32_t il_cache = map_layer_ids[il];
    ggml_tensor * scores = h2o_scores[il_cache];
    ggml_tensor * mem_idx = h2o_memory_indices[il_cache];

    const uint32_t n_head = hparams.n_head(il);
    const uint32_t M = h2o_memory_size;

    ggml_bf16_t * score_data = (ggml_bf16_t *)scores->data;
    const int32_t * mem_idx_data = (const int32_t *)mem_idx->data;

    for (uint32_t ih = 0; ih < n_head; ++ih) {
        for (uint32_t m = 0; m < M; ++m) {
            int32_t global_pos = mem_idx_data[ih * M + m];
            if (global_pos >= 0 && global_pos < (int32_t)scores->ne[0]) {
                size_t score_idx = ih * scores->ne[0] + global_pos;
                float current = ggml_bf16_to_fp32(score_data[score_idx]);
                float delta = inter_weights_colsum[ih * M + m];
                score_data[score_idx] = ggml_fp32_to_bf16(current + delta);
            }
        }
    }
}
```

#### 6.3 Memory Set Construction

```cpp
void llama_kv_cache::h2o_build_memory_set(int32_t il, uint32_t chunk_end) {
    GGML_ASSERT(swa_type == LLAMA_SWA_TYPE_H2O);

    const int32_t il_cache = map_layer_ids[il];
    ggml_tensor * scores = h2o_scores[il_cache];
    ggml_tensor * mem_idx = h2o_memory_indices[il_cache];

    const uint32_t n_head = hparams.n_head(il);
    const uint32_t L = h2o_local_window;
    const uint32_t H = h2o_heavy_budget;
    const uint32_t M = h2o_memory_size;

    uint32_t chunk_start = h2o_state.chunk_boundaries.back();

    const ggml_bf16_t * score_data = (const ggml_bf16_t *)scores->data;
    int32_t * mem_idx_data = (int32_t *)mem_idx->data;

    for (uint32_t ih = 0; ih < n_head; ++ih) {
        // Step 1: Local window (tail L tokens)
        std::vector<int32_t> local;
        uint32_t local_start = (chunk_end > L) ? (chunk_end - L) : 0;
        for (uint32_t pos = local_start; pos < chunk_end; ++pos) {
            local.push_back(pos);
        }

        // Step 2: Build candidate set
        std::set<int32_t> local_set(local.begin(), local.end());
        std::vector<std::pair<int32_t, float>> candidates;

        // Add previous memory (excluding local)
        if (h2o_memory_initialized) {
            for (uint32_t m = 0; m < M; ++m) {
                int32_t pos = mem_idx_data[ih * M + m];
                if (pos >= 0 && local_set.find(pos) == local_set.end()) {
                    size_t score_idx = ih * scores->ne[0] + pos;
                    candidates.push_back({pos, ggml_bf16_to_fp32(score_data[score_idx])});
                }
            }
        }

        // Add current chunk (excluding local)
        for (uint32_t pos = chunk_start; pos < chunk_end; ++pos) {
            if (local_set.find(pos) == local_set.end()) {
                size_t score_idx = ih * scores->ne[0] + pos;
                candidates.push_back({pos, ggml_bf16_to_fp32(score_data[score_idx])});
            }
        }

        // Step 3: Select top-H
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });

        std::vector<int32_t> heavy;
        for (size_t i = 0; i < std::min((size_t)H, candidates.size()); ++i) {
            heavy.push_back(candidates[i].first);
        }

        // Step 4: Combine and sort
        std::vector<int32_t> memory(local.begin(), local.end());
        memory.insert(memory.end(), heavy.begin(), heavy.end());
        std::sort(memory.begin(), memory.end());

        // Write to tensor
        for (uint32_t m = 0; m < M; ++m) {
            mem_idx_data[ih * M + m] = (m < memory.size()) ? memory[m] : -1;
        }
    }

    h2o_memory_initialized = true;
}
```

#### 6.4 KV Gathering

```cpp
ggml_tensor * llama_kv_cache::h2o_gather_k_memory(
        ggml_context * ctx, int32_t il) const {

    GGML_ASSERT(swa_type == LLAMA_SWA_TYPE_H2O);
    GGML_ASSERT(h2o_memory_initialized);

    const int32_t il_cache = map_layer_ids.at(il);
    const kv_layer & layer = layers[il_cache];
    ggml_tensor * mem_idx = h2o_memory_indices[il_cache];

    ggml_tensor * k_mem = ggml_get_rows(ctx, layer.k, mem_idx);
    ggml_set_name(k_mem, "h2o_k_mem");

    return k_mem;
}

ggml_tensor * llama_kv_cache::h2o_gather_v_memory(
        ggml_context * ctx, int32_t il) const {

    GGML_ASSERT(swa_type == LLAMA_SWA_TYPE_H2O);
    GGML_ASSERT(h2o_memory_initialized);

    const int32_t il_cache = map_layer_ids.at(il);
    const kv_layer & layer = layers[il_cache];
    ggml_tensor * mem_idx = h2o_memory_indices[il_cache];

    ggml_tensor * v_mem = ggml_get_rows(ctx, layer.v, mem_idx);
    ggml_set_name(v_mem, "h2o_v_mem");

    return v_mem;
}
```

#### 6.5 Chunk Management

```cpp
void llama_kv_cache::h2o_next_chunk(uint32_t chunk_len) {
    GGML_ASSERT(swa_type == LLAMA_SWA_TYPE_H2O);

    h2o_state.total_tokens_processed += chunk_len;
    h2o_state.current_chunk_idx++;
    h2o_state.chunk_boundaries.push_back(h2o_state.total_tokens_processed);

    if (debug >= 1) {
        LLAMA_LOG_DEBUG("%s: chunk %d complete, total tokens %d\n",
                        __func__, h2o_state.current_chunk_idx - 1,
                        h2o_state.total_tokens_processed);
    }
}
```

---

## Verifiable Metrics and Experiments

### Metric 1: Memory Layout Correctness

**Test**: Contiguous and cache-friendly memory layout

```cpp
// Unit test
void test_memory_layout() {
    llama_kv_cache kv(..., LLAMA_SWA_TYPE_H2O, ..., 256, 256);

    // Check tensor contiguity
    for (auto * scores : kv.h2o_scores) {
        assert(ggml_is_contiguous(scores));
        assert(scores->ne[0] == kv_size);
        assert(scores->ne[1] == n_head);
    }

    // Check stride alignment
    for (auto * mem_idx : kv.h2o_memory_indices) {
        assert(mem_idx->nb[0] == sizeof(int32_t));
        assert(mem_idx->nb[1] == M * sizeof(int32_t));
    }
}
```

**Target**: All tensors contiguous, strides match expected values

---

### Metric 2: Access Pattern Performance (O(1) Index Lookup)

**Test**: Profile with `torch.profiler` equivalent or timing

```cpp
void test_access_pattern() {
    // Warm-up
    for (int i = 0; i < 100; i++) {
        float score = ggml_bf16_to_fp32(score_data[rand() % total_elements]);
    }

    // Benchmark random access
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        uint32_t head = rand() % n_head;
        uint32_t pos = rand() % kv_size;
        float score = ggml_bf16_to_fp32(score_data[head * kv_size + pos]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Target: < 1μs per access on average
    assert(duration.count() / 10000 < 1);
}
```

**Target**: Random access < 1μs per lookup

---

### Metric 3: Memory Overhead

**Test**: Compare peak memory usage vs baseline

```cpp
void test_memory_overhead() {
    // Baseline: standard KV cache
    llama_kv_cache kv_baseline(..., LLAMA_SWA_TYPE_NONE, ...);
    size_t baseline_size = kv_baseline.total_size();

    // H2O: with score tracking
    llama_kv_cache kv_h2o(..., LLAMA_SWA_TYPE_H2O, ..., 256, 256);
    size_t h2o_size = kv_h2o.total_size();

    // Calculate overhead
    size_t overhead = h2o_size - baseline_size;
    float overhead_percent = 100.0f * overhead / baseline_size;

    // Expected overhead:
    // 32 layers × 32 heads × 4096 kv_size × 2 bytes (BF16) = 8 MB scores
    // 32 layers × 32 heads × 512 M × 4 bytes (I32) = 2 MB indices
    // Total: ~10 MB overhead (~1-2% of 1GB KV cache)

    assert(overhead_percent < 5.0f);
}
```

**Target**: < 5% additional memory vs baseline KV cache

---

### Metric 4: Index Bounds Safety

**Test**: Edge cases and boundary conditions

```cpp
void test_index_bounds() {
    llama_kv_cache kv(..., LLAMA_SWA_TYPE_H2O, ..., 256, 256);

    // Edge case 1: N = S (single chunk)
    kv.h2o_build_memory_set(0, 1024);
    verify_no_oob_access(kv);

    // Edge case 2: N = 2S - 1 (partial second chunk)
    kv.h2o_next_chunk(1024);
    kv.h2o_build_memory_set(0, 2047);
    verify_no_oob_access(kv);

    // Edge case 3: Very small chunk (< L)
    kv.h2o_build_memory_set(0, 100);
    verify_no_oob_access(kv);
}

void verify_no_oob_access(const llama_kv_cache & kv) {
    for (auto * mem_idx : kv.h2o_memory_indices) {
        int32_t * data = (int32_t *)mem_idx->data;
        for (size_t i = 0; i < mem_idx->ne[0] * mem_idx->ne[1]; i++) {
            int32_t idx = data[i];
            assert(idx == -1 || (idx >= 0 && idx < kv_size));
        }
    }
}
```

**Target**: No out-of-bounds access in any edge case

---

### Metric 5: Score Initialization Correctness

**Test**: Column sum matches reference

```cpp
void test_score_initialization() {
    // Create mock attention weights [n_head, S, S]
    std::vector<float> attn_weights(n_head * S * S);
    fill_with_random_softmax(attn_weights);

    // Compute column sum manually
    std::vector<float> expected_colsum(n_head * S, 0.0f);
    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t j = 0; j < S; j++) {
            for (uint32_t i = 0; i < S; i++) {
                expected_colsum[h * S + j] += attn_weights[h * S * S + i * S + j];
            }
        }
    }

    // Initialize scores via API
    kv.h2o_init_chunk_scores(0, 0, S, expected_colsum.data());

    // Verify
    ggml_bf16_t * score_data = (ggml_bf16_t *)kv.h2o_scores[0]->data;
    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t pos = 0; pos < S; pos++) {
            float stored = ggml_bf16_to_fp32(score_data[h * kv_size + pos]);
            float expected = expected_colsum[h * S + pos];
            assert(fabs(stored - expected) < 1e-3);  // BF16 tolerance
        }
    }
}
```

**Target**: Max error < 1e-3 (BF16 precision)

---

### Metric 6: Score Accumulation Correctness

**Test**: Monotonic increase, no overflow

```cpp
void test_score_accumulation() {
    // Initialize with intra-attention scores
    std::vector<float> intra_colsum(n_head * S);
    fill_with_values(intra_colsum, 0.1f, 1.0f);
    kv.h2o_init_chunk_scores(0, 0, S, intra_colsum.data());

    // Build memory set
    kv.h2o_build_memory_set(0, S);

    // Record initial scores
    std::vector<float> scores_before = read_all_scores(kv);

    // Accumulate from inter-attention
    std::vector<float> inter_colsum(n_head * M);
    fill_with_values(inter_colsum, 0.05f, 0.5f);
    kv.h2o_accumulate_memory_scores(0, inter_colsum.data());

    // Verify monotonic increase
    std::vector<float> scores_after = read_all_scores(kv);
    for (size_t i = 0; i < scores_before.size(); i++) {
        assert(scores_after[i] >= scores_before[i]);
    }

    // Verify no overflow
    for (float score : scores_after) {
        assert(!std::isnan(score) && !std::isinf(score));
    }
}
```

**Target**: Scores never decrease, no NaN/inf

---

### Metric 7: Heavy Hitter Selection Correctness

**Test**: Top-H selection matches manual sort

```cpp
void test_heavy_selection() {
    // Set known scores
    ggml_bf16_t * score_data = (ggml_bf16_t *)kv.h2o_scores[0]->data;
    std::vector<float> known_scores = {0.1, 0.9, 0.3, 0.8, 0.2, ...};
    for (size_t i = 0; i < known_scores.size(); i++) {
        score_data[i] = ggml_fp32_to_bf16(known_scores[i]);
    }

    // Build memory set
    kv.h2o_build_memory_set(0, S);

    // Manually compute expected top-H
    std::vector<std::pair<int, float>> indexed_scores;
    for (size_t i = 0; i < known_scores.size(); i++) {
        if (i < S - L) {  // Exclude local window
            indexed_scores.push_back({i, known_scores[i]});
        }
    }
    std::sort(indexed_scores.begin(), indexed_scores.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    std::set<int> expected_heavy;
    for (size_t i = 0; i < H && i < indexed_scores.size(); i++) {
        expected_heavy.insert(indexed_scores[i].first);
    }

    // Extract actual heavy hitters
    int32_t * mem_idx_data = (int32_t *)kv.h2o_memory_indices[0]->data;
    std::set<int> actual_heavy;
    for (uint32_t m = L; m < M; m++) {  // Heavy region
        if (mem_idx_data[m] >= 0) {
            actual_heavy.insert(mem_idx_data[m]);
        }
    }

    assert(actual_heavy == expected_heavy);
}
```

**Target**: Exact match with manual top-K selection

---

### Metric 8: Cross-Chunk Propagation

**Test**: Token from C0 survives to M2

```cpp
void test_cross_chunk_propagation() {
    // Process chunk 0
    kv.h2o_init_chunk_scores(0, 0, S, chunk0_scores.data());
    kv.h2o_build_memory_set(0, S);
    kv.h2o_next_chunk(S);

    // Mark a specific token with high score
    int32_t important_token = 42;
    ggml_bf16_t * score_data = (ggml_bf16_t *)kv.h2o_scores[0]->data;
    score_data[important_token] = ggml_fp32_to_bf16(10.0f);

    // Process chunk 1
    kv.h2o_accumulate_memory_scores(0, chunk1_inter_scores.data());
    kv.h2o_init_chunk_scores(0, S, S, chunk1_intra_scores.data());
    kv.h2o_build_memory_set(0, 2*S);
    kv.h2o_next_chunk(S);

    // Verify token 42 is in M1
    bool found_in_m1 = check_token_in_memory(kv, 0, important_token);
    assert(found_in_m1);

    // Process chunk 2
    kv.h2o_accumulate_memory_scores(0, chunk2_inter_scores.data());
    kv.h2o_init_chunk_scores(0, 2*S, S, chunk2_intra_scores.data());
    kv.h2o_build_memory_set(0, 3*S);

    // Verify token 42 is STILL in M2 (cross-chunk survival)
    bool found_in_m2 = check_token_in_memory(kv, 0, important_token);
    assert(found_in_m2);
}
```

**Target**: High-score tokens survive across 3+ chunks

---

### Metric 9: Per-Head Independence

**Test**: Different heads select different tokens

```cpp
void test_per_head_independence() {
    // Set different scores for each head
    ggml_bf16_t * score_data = (ggml_bf16_t *)kv.h2o_scores[0]->data;
    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t pos = 0; pos < S; pos++) {
            // Each head has different score pattern
            float score = (pos * 7 + h * 11) % 100 / 100.0f;
            score_data[h * kv_size + pos] = ggml_fp32_to_bf16(score);
        }
    }

    kv.h2o_build_memory_set(0, S);

    // Extract memory sets for each head
    int32_t * mem_idx_data = (int32_t *)kv.h2o_memory_indices[0]->data;
    std::vector<std::set<int>> memory_per_head(n_head);
    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t m = 0; m < M; m++) {
            int32_t idx = mem_idx_data[h * M + m];
            if (idx >= 0) memory_per_head[h].insert(idx);
        }
    }

    // Verify heads have different selections
    int different_count = 0;
    for (uint32_t h1 = 0; h1 < n_head - 1; h1++) {
        if (memory_per_head[h1] != memory_per_head[h1+1]) {
            different_count++;
        }
    }

    assert(different_count > n_head / 2);  // At least half should differ
}
```

**Target**: > 50% of adjacent heads have different memory sets

---

### Metric 10: Sorted Output for Coalesced Access

**Test**: Memory indices are sorted

```cpp
void test_sorted_indices() {
    kv.h2o_build_memory_set(0, S);

    int32_t * mem_idx_data = (int32_t *)kv.h2o_memory_indices[0]->data;

    for (uint32_t h = 0; h < n_head; h++) {
        // Extract valid indices
        std::vector<int32_t> indices;
        for (uint32_t m = 0; m < M; m++) {
            int32_t idx = mem_idx_data[h * M + m];
            if (idx >= 0) indices.push_back(idx);
        }

        // Check sorted
        for (size_t i = 1; i < indices.size(); i++) {
            assert(indices[i] > indices[i-1]);
        }
    }
}
```

**Target**: All indices in ascending order within each head

---

### Metric 11: Numerical Stability (Long Sequences)

**Test**: No overflow after many accumulations

```cpp
void test_numerical_stability() {
    // Simulate 16 chunks (16384 tokens)
    for (int chunk = 0; chunk < 16; chunk++) {
        std::vector<float> scores(n_head * S, 0.5f);

        if (chunk == 0) {
            kv.h2o_init_chunk_scores(0, 0, S, scores.data());
        } else {
            kv.h2o_accumulate_memory_scores(0, scores.data());
            kv.h2o_init_chunk_scores(0, chunk * S, S, scores.data());
        }

        kv.h2o_build_memory_set(0, (chunk + 1) * S);
        kv.h2o_next_chunk(S);
    }

    // Check all scores for inf/nan
    ggml_bf16_t * score_data = (ggml_bf16_t *)kv.h2o_scores[0]->data;
    for (size_t i = 0; i < n_head * kv_size; i++) {
        float score = ggml_bf16_to_fp32(score_data[i]);
        assert(!std::isnan(score) && !std::isinf(score));
        assert(score >= 0.0f && score < 1e6f);
    }
}
```

**Target**: No inf/nan after 16 chunks, scores < 1e6

---

## Performance Benchmarks

### Benchmark 1: Memory Allocation Time

```cpp
auto start = std::chrono::high_resolution_clock::now();
llama_kv_cache kv(..., LLAMA_SWA_TYPE_H2O, ..., 256, 256);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

// Target: < 100ms for full allocation
assert(duration.count() < 100);
```

**Target**: < 100ms allocation time

---

### Benchmark 2: Memory Set Construction Speed

```cpp
// Profile h2o_build_memory_set()
auto start = std::chrono::high_resolution_clock::now();
kv.h2o_build_memory_set(0, S);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

// Target: < 1ms per layer (32 layers = 32ms per chunk)
assert(duration.count() < 1000);
```

**Target**: < 1ms per layer, < 100ms for 32 layers

---

### Benchmark 3: Score Update Throughput

```cpp
std::vector<float> inter_scores(n_head * M, 0.01f);

auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 1000; i++) {
    kv.h2o_accumulate_memory_scores(0, inter_scores.data());
}
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

// Target: < 10μs per update
assert(duration.count() / 1000 < 10);
```

**Target**: < 10μs per score update call

---

## Critical Files Summary

| File | Lines to Modify | Description |
|------|----------------|-------------|
| `/root/workspace/Prism/src/llama-hparams.h` | 18-23 | Add LLAMA_SWA_TYPE_H2O enum |
| `/root/workspace/Prism/src/llama-kv-cache.h` | 96-109, 201+, 250+ | Constructor signature, public API, private fields |
| `/root/workspace/Prism/src/llama-kv-cache.cpp` | 17+, 170+, EOF | Constructor impl, tensor allocation, H2O methods |

**Total Estimated LOC**: ~800 lines (400 implementation + 400 tests)

---

## Success Criteria

✅ All 11 verification metrics pass
✅ All 3 performance benchmarks meet targets
✅ Memory overhead < 5%
✅ No crashes on edge cases (N=S, N=2S-1, N<L)
✅ Compilation succeeds on CPU and CUDA backends
✅ Integration tests pass with existing KV cache tests

---

## Next Steps After Task 1

1. **Task 2**: Implement intra-chunk causal attention (reuse existing flash_attn)
2. **Task 3**: Wire score tracking into attention computation graph
3. **Task 4**: Implement memory selection in batch processing loop
4. **Task 5**: Implement inter-chunk attention with online softmax fusion
5. **Task 6**: End-to-end pipeline integration
6. **Task 7**: CUDA kernel optimization
7. **Task 8**: Benchmarking and quality validation
