# Task 5.7: H2O Memory Management for GQA + MHA

## Objective
Modify the H2O memory collection/management logic to correctly support two attention head types:
- Traditional **MHA (Multi-Head Attention)**
- **GQA (Grouped Query Attention)**

The model type is determined during model loading/initialization, and **GQA is the default behavior**. MHA is treated as a degenerate GQA case (group size 1).

## Overview
This update aligns H2O memory data structures and score handling with **KV-heads** (n_head_kv) instead of Q-heads (n_head). For GQA models, attention column sums (computed per Q-head) are **reduced to KV-heads** before scoring and memory selection. For MHA, n_head == n_head_kv and the reduction is a no-op.

## Implementation Steps

### Step 1: Determine head type during model load (default GQA)
**Files**: `src/llama-hparams.h`, `src/llama-hparams.cpp`, `src/llama-model.cpp`

Add a head-type enum and store per-layer head type in hparams. Default to GQA and mark MHA when `n_head_kv == n_head`.

```cpp
// src/llama-hparams.h
enum llama_attn_head_type : uint8_t {
    LLAMA_ATTN_HEAD_GQA = 0,
    LLAMA_ATTN_HEAD_MHA = 1,
};

struct llama_hparams {
    ...
    std::array<uint8_t, LLAMA_MAX_LAYERS> attn_head_type_arr;
    ...
    llama_attn_head_type attn_head_type(uint32_t il) const;
    bool is_gqa(uint32_t il) const { return attn_head_type(il) == LLAMA_ATTN_HEAD_GQA; }
    uint32_t n_head_mem(uint32_t il) const {
        const uint32_t n_kv = n_head_kv(il);
        return n_kv > 0 ? n_kv : n_head(il);
    }
};
```

```cpp
// src/llama-hparams.cpp
llama_attn_head_type llama_hparams::attn_head_type(uint32_t il) const {
    if (il < n_layer) {
        return static_cast<llama_attn_head_type>(attn_head_type_arr[il]);
    }
    GGML_ABORT("fatal error");
}
```

```cpp
// src/llama-model.cpp (after n_head_kv_arr is loaded)
std::fill(hparams.attn_head_type_arr.begin(), hparams.attn_head_type_arr.end(),
          LLAMA_ATTN_HEAD_GQA); // default

for (uint32_t il = 0; il < hparams.n_layer; ++il) {
    const uint32_t n_head    = hparams.n_head_arr[il];
    const uint32_t n_head_kv = hparams.n_head_kv_arr[il];
    if (n_head_kv > 0 && n_head_kv == n_head) {
        hparams.attn_head_type_arr[il] = LLAMA_ATTN_HEAD_MHA;
    }
}
```

**Verification**:
- `cmake --build build -j12 --target llama-cli`

---

### Step 2: Use KV-head count for H2O tensors and loops (GQA default)
**File**: `src/llama-kv-cache.cpp`

Allocate H2O tensors using `n_head_mem` (KV-head count). Update all H2O loops to use KV-heads.

```cpp
// H2O tensor allocation
const uint32_t n_head_mem = hparams.n_head_mem(il);

ggml_tensor * scores = ggml_new_tensor_2d(ctx_h2o, GGML_TYPE_BF16, kv_size, n_head_mem);
...

ggml_tensor * mem_idx = ggml_new_tensor_2d(ctx_h2o, GGML_TYPE_I32, h2o_memory_size, n_head_mem);
...
```

```cpp
// In h2o_init_chunk_scores / h2o_accumulate_memory_scores / h2o_build_memory_set
const uint32_t n_head = hparams.n_head_mem(il);
```

**Verification**:
- `cmake --build build -j12 --target test-h2o-kv-cache`

---

### Step 3: Reduce Q-head colsum -> KV-head colsum before H2O scoring
**File**: `src/llama-context.cpp`

Add a helper to reduce per-Q-head colsum into KV-head space.

```cpp
static void h2o_reduce_colsum_to_kv_heads(
        const float * src,
        uint32_t n_cols,
        uint32_t n_head,
        uint32_t n_head_kv,
        float * dst) {
    if (n_head_kv == n_head) {
        std::memcpy(dst, src, n_head * n_cols * sizeof(float));
        return;
    }
    GGML_ASSERT(n_head_kv > 0);
    GGML_ASSERT(n_head % n_head_kv == 0);

    const uint32_t n_gqa = n_head / n_head_kv;
    std::memset(dst, 0, n_head_kv * n_cols * sizeof(float));

    for (uint32_t kv = 0; kv < n_head_kv; ++kv) {
        float * dst_head = dst + kv * n_cols;
        for (uint32_t g = 0; g < n_gqa; ++g) {
            const uint32_t qh = kv * n_gqa + g;
            const float * src_head = src + qh * n_cols;
            for (uint32_t c = 0; c < n_cols; ++c) {
                dst_head[c] += src_head[c];
            }
        }
    }
}
```

Use the helper before H2O updates in both phase-1 and phase-2:

```cpp
// Phase 1 (per-chunk colsum)
std::vector<float> chunk_src(n_head * chunk_len);
...
std::vector<float> chunk_colsum(hparams.n_head_mem(layer.il) * chunk_len);
h2o_reduce_colsum_to_kv_heads(
    chunk_src.data(), chunk_len, n_head, n_head_kv, chunk_colsum.data());
kv->h2o_init_chunk_scores(layer.il, chunk_start, chunk_len, chunk_colsum.data());
```

```cpp
// Phase 2 (inter colsum)
std::vector<float> inter_kv(hparams.n_head_mem(layer.il) * n_cols);
h2o_reduce_colsum_to_kv_heads(
    layer.data.data(), n_cols, n_head, n_head_kv, inter_kv.data());
kv->h2o_accumulate_memory_scores(il, inter_kv.data());
```

**Verification**:
- `cmake --build build -j12 --target test-h2o-edge-cases`

---

### Step 4: Tests (GQA coverage + existing updates)
**Files**: `tests/test-h2o-gqa-memory.cpp` (new), `tests/test-h2o-edge-cases.cpp`, `tests/test-h2o-kv-cache.cpp`, `tests/test-h2o-memory-selection.cpp`, `tests/CMakeLists.txt`

#### New test (verifiable code)

```cpp
// tests/test-h2o-gqa-memory.cpp
#include "ggml.h"
#include "llama-cpp.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama.h"
#include "get-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void require(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "H2O GQA memory test failed: %s\n", msg);
        std::exit(EXIT_FAILURE);
    }
}

int32_t find_first_kv_layer(const llama_hparams & hparams) {
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (hparams.has_kv(il)) {
            return static_cast<int32_t>(il);
        }
    }
    return -1;
}

bool has_token(const ggml_tensor * mem_idx, uint32_t head, uint32_t M, int32_t token) {
    const uint8_t * base = static_cast<const uint8_t *>(mem_idx->data);
    const int32_t * head_data = reinterpret_cast<const int32_t *>(base + head * mem_idx->nb[1]);
    for (uint32_t m = 0; m < M; ++m) {
        if (head_data[m] == token) {
            return true;
        }
    }
    return false;
}

void reduce_colsum_to_kv_heads(
        const float * src, uint32_t n_cols, uint32_t n_head, uint32_t n_head_kv, float * dst) {
    if (n_head_kv == n_head) {
        std::memcpy(dst, src, n_head * n_cols * sizeof(float));
        return;
    }
    require(n_head_kv > 0, "n_head_kv == 0");
    require(n_head % n_head_kv == 0, "n_head not divisible by n_head_kv");
    const uint32_t n_gqa = n_head / n_head_kv;
    std::memset(dst, 0, n_head_kv * n_cols * sizeof(float));
    for (uint32_t kv = 0; kv < n_head_kv; ++kv) {
        float * dst_head = dst + kv * n_cols;
        for (uint32_t g = 0; g < n_gqa; ++g) {
            const uint32_t qh = kv * n_gqa + g;
            const float * src_head = src + qh * n_cols;
            for (uint32_t c = 0; c < n_cols; ++c) {
                dst_head[c] += src_head[c];
            }
        }
    }
}
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "WARNING: failed to load model '%s', skipping\n", model_path);
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const llama_hparams & hparams = model->hparams;
    const int32_t il = find_first_kv_layer(hparams);
    if (il < 0) {
        fprintf(stderr, "WARNING: no KV layers, skipping\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }

    const uint32_t n_head    = hparams.n_head(il);
    const uint32_t n_head_kv = hparams.n_head_kv(il);
    if (n_head_kv == n_head) {
        fprintf(stderr, "WARNING: model is MHA (n_head_kv == n_head), skipping GQA test\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }
    require(n_head_kv > 0, "n_head_kv == 0");
    require(n_head % n_head_kv == 0, "invalid GQA ratio");

    const uint32_t kv_size   = 32;
    const uint32_t chunk_len = 8;
    const uint32_t h2o_local = 2;
    const uint32_t h2o_heavy = 2;

    llama_kv_cache kv(
        *model, GGML_TYPE_F16, GGML_TYPE_F16, true, false, true,
        kv_size, 1, 1, 0, LLAMA_SWA_TYPE_H2O, nullptr, nullptr, h2o_local, h2o_heavy);

    std::vector<float> q_colsum(n_head * chunk_len, 0.0f);
    const uint32_t n_gqa = n_head / n_head_kv;
    for (uint32_t kvh = 0; kvh < n_head_kv; ++kvh) {
        const uint32_t hot_token = (kvh * 2) % (chunk_len - h2o_local);
        for (uint32_t g = 0; g < n_gqa; ++g) {
            const uint32_t qh = kvh * n_gqa + g;
            for (uint32_t pos = 0; pos < chunk_len; ++pos) {
                q_colsum[qh * chunk_len + pos] = (pos == hot_token) ? 10.0f : 0.1f;
            }
        }
    }

    std::vector<float> kv_colsum(n_head_kv * chunk_len, 0.0f);
    reduce_colsum_to_kv_heads(q_colsum.data(), chunk_len, n_head, n_head_kv, kv_colsum.data());

    kv.h2o_init_chunk_scores(il, 0, chunk_len, kv_colsum.data());
    kv.h2o_build_memory_set(il, chunk_len);

    const ggml_tensor * mem_idx = kv.h2o_get_memory_indices_tensor(il);
    require(mem_idx != nullptr, "mem_idx missing");
    require(mem_idx->ne[1] == static_cast<int64_t>(n_head_kv), "mem_idx head count mismatch");

    const uint32_t M = kv.h2o_get_memory_size();
    const uint32_t local_start = chunk_len - h2o_local;

    for (uint32_t kvh = 0; kvh < n_head_kv; ++kvh) {
        for (uint32_t pos = local_start; pos < chunk_len; ++pos) {
            require(has_token(mem_idx, kvh, M, static_cast<int32_t>(pos)), "local token missing");
        }
        const int32_t hot_token = static_cast<int32_t>((kvh * 2) % (chunk_len - h2o_local));
        require(has_token(mem_idx, kvh, M, hot_token), "expected heavy token missing");
    }

    llama_backend_free();
    return EXIT_SUCCESS;
}
```

#### Update existing tests
- `tests/test-h2o-edge-cases.cpp`: remove GQA skip; assert `h2o_get_memory_indices_tensor(il)->ne[1] == hparams.n_head_kv(il)` after memory init.
- `tests/test-h2o-kv-cache.cpp`: replace `n_head` with `n_head_kv` (or `hparams.n_head_mem(il)`) for expected tensor shapes and colsum sizes.
- `tests/test-h2o-memory-selection.cpp`: same head-count adjustment in loops and expected sizes.
- `tests/CMakeLists.txt`: add new test target and include dirs.

```cmake
llama_build_and_test(test-h2o-gqa-memory.cpp)
target_include_directories(test-h2o-gqa-memory PRIVATE ${PROJECT_SOURCE_DIR}/src)
```

**Verification**:
- `cmake --build build -j12 --target test-h2o-gqa-memory test-h2o-edge-cases test-h2o-kv-cache test-h2o-memory-selection`
- `LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf ./build/bin/test-h2o-gqa-memory`
- `LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf ./build/bin/test-h2o-edge-cases`

## Notes
- All existing MHA behavior remains intact via `n_head_kv == n_head`.
- H2O memory indices and score tensors are aligned to KV-heads for both attention types.
- This plan does not address flash-attention integration (explicitly out of scope).
