# Task 5: Inter-Chunk Attention + Online Softmax Fusion

## Overview

Implement inter-chunk attention with online softmax fusion to complete the H2O chunked sparse attention system. This task adds the missing inter-chunk attention mechanism that fuses with intra-chunk attention using the online softmax algorithm.

## Status

**Dependencies**: Tasks 1-4 completed ✅
**Current**: Task 5 implementation
**Architecture**: Two-phase parallel-then-sequential processing

## Key Architecture

### Two-Phase Processing

**Phase 1 (Parallel)**: All chunks compute intra-attention simultaneously
- All chunks process Q(Cₖ) × K(Cₖ)ᵀ in parallel
- Store K,V in KV cache + attention outputs in temporary buffer
- Get partial results (each chunk only attends to itself)
- Initialize scores from intra-attention

**Phase 2 (Sequential)**: Each chunk updates via inter-attention + online softmax fusion
- Chunk 0: Use Phase 1 output directly (no history)
- Chunks 1,2,3...: Inter-attention with previous memory M_{c-1}
- Online softmax fusion to merge inter + intra results
- Score accumulation for memory tokens
- Build memory set for next chunk

### Why This Approach?

✅ Leverages GPU parallelism (Phase 1 processes all chunks simultaneously)
✅ Better throughput for prefill phase
✅ Cleaner separation: partial results → correction → final results

## Implementation Steps

### Step 1: Add Online Softmax State Structures

**File**: `src/llama-graph.h` (after line 582 in `llm_graph_result`)

```cpp
// Online softmax state for chunked attention fusion
struct h2o_online_softmax_state {
    ggml_tensor * m;      // [n_tokens, n_head] running max of logits
    ggml_tensor * l;      // [n_tokens, n_head] running sum exp(logit - m)
    ggml_tensor * o;      // [n_embd, n_tokens, n_head] running weighted sum
    bool initialized = false;
};

// Storage for Phase 1 outputs and Phase 2 fusion
struct h2o_phase_state {
    std::map<int, ggml_tensor *> intra_output;   // [il] → output tensor
    std::map<int, ggml_tensor *> intra_logits;   // [il] → logits before softmax
    std::map<int, h2o_online_softmax_state> online_state;  // [il] → state
};

h2o_phase_state h2o_phase;
```

### Step 2: Implement Inter-Chunk Attention

**File**: `src/llama-graph.cpp` (after `build_attn_h2o` ~line 1788)

Add `build_attn_inter_h2o` function to compute Q(C_c) × K(M_{c-1})^T with no causal mask.

**Function signature** in `src/llama-graph.h`:
```cpp
ggml_tensor * build_attn_inter_h2o(
    ggml_tensor * q, ggml_tensor * k_mem, ggml_tensor * v_mem,
    float kq_scale, int il,
    ggml_tensor ** inter_logits_out,
    ggml_tensor ** inter_weights_out) const;
```

### Step 3: Implement Online Softmax Fusion

**File**: `src/llama-graph.cpp`

Two helper functions:
1. `init_online_softmax_state_h2o`: Initialize (m, ℓ, o) state from inter-attention
2. `update_online_softmax_state_h2o`: Update state with intra-attention and finalize

**Function signatures** in `src/llama-graph.h`:
```cpp
void init_online_softmax_state_h2o(
    ggml_tensor * inter_logits, ggml_tensor * v_mem,
    h2o_online_softmax_state & state) const;

ggml_tensor * update_online_softmax_state_h2o(
    h2o_online_softmax_state & state,
    ggml_tensor * intra_logits, ggml_tensor * v_intra) const;
```

### Step 4: Modify build_attn_mha_h2o to Return Logits

**File**: `src/llama-graph.cpp` (line 1663-1741)

Add optional parameter `ggml_tensor ** attn_logits_out` to return logits before softmax.
Store logits after scaling and masking but before softmax application.

### Step 5: Add Phase Flag to Context

**File**: `src/llama-context.h`

Add field:
```cpp
int current_phase = 0;  // 0=normal, 1=phase1, 2=phase2
```

### Step 6: Implement Two-Phase Orchestration

**File**: `src/llama-context.cpp` (in `llama_decode_internal` around line 1400-1642)

Add two-phase processing logic:
1. **Phase 1 Loop**: Process all chunks with intra-attention in parallel
2. **Phase 2 Loop**: Sequentially update chunks 1+ with inter-attention + fusion

### Step 7: Modify Graph Building

**File**: `src/llama-graph.cpp` (around line 1933-1963)

Modify attention building section to check `current_phase`:
- Phase 1: Use existing `build_attn_h2o` (intra-only), store outputs and logits
- Phase 2: Use `build_attn_inter_h2o` + online softmax fusion for chunks 1+

## Verification Metrics

### Correctness
- [ ] Online softmax produces mathematically equivalent results (< 1e-5 error)
- [ ] Inter-attention computes correct attention over memory set
- [ ] Score accumulation increases scores monotonically
- [ ] Memory sets built correctly with Local + Heavy selection
- [ ] Cross-chunk propagation works

### Performance
- [ ] H2O throughput >= 1.5× baseline for N=4096
- [ ] Phase 1 parallelism reduces total time
- [ ] Memory overhead < 10% vs baseline
- [ ] No memory leaks

### Robustness
- [ ] No crashes on edge cases (N=S, N=2S-1, N<L)
- [ ] Numerical stability with extreme logits
- [ ] GPU memory stays within limits
- [ ] CUDA and CPU backends work

## Build and Test Commands

```bash
# Clean build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j12

# Run integration test
LLAMACPP_TEST_MODELFILE=/models/Qwen_Qwen3-1.7B-Q8_0.gguf \
./build/bin/llama-cli -m /models/Qwen_Qwen3-1.7B-Q8_0.gguf \
  -p "Hello world from H2O! " -n 0 \
  -b 4096 -ub 1024 --h2o-local 256 --h2o-heavy 256 -st

# Performance benchmark
./build/bin/llama-cli -m /models/Qwen_Qwen3-1.7B-Q8_0.gguf \
  -p "$(head -c 4000 /dev/urandom | base64)" -n 0 \
  -b 4096 -ub 1024 --h2o-local 256 --h2o-heavy 256 -st
```

## Critical Files

| File | Lines | Description |
|------|-------|-------------|
| `src/llama-graph.h` | +60 | State structures + function declarations |
| `src/llama-graph.cpp` | +300 | Inter-attention + online softmax + graph building |
| `src/llama-context.h` | +1 | Phase flag |
| `src/llama-context.cpp` | +150 | Two-phase orchestration |

**Total Estimated LOC**: ~800 lines

## Success Criteria

✅ End-to-end test completes without crashes
✅ 4096 tokens processed in 4 chunks
✅ Memory sets initialized for all layers
✅ Performance targets met (1.5× throughput)
✅ Numerical stability (no inf/nan)
✅ Mathematical equivalence with reference implementation

## References

- Plan: `/root/.claude/plans/vectorized-tinkering-planet.md`
- Whole Plan: `/root/workspace/Prism/tasks/whole_plan.md` (Section 1.2.5, modified)
- Tasks 1-4: `/root/workspace/Prism/tasks/task_{1,2,3,4}.md`
