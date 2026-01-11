# Chunked Sparse Attention for Prefill Optimization

## Terminology

| Term | Definition | llama.cpp Flag |
|------|------------|----------------|
| **request** | Single user prompt (we only support 1 concurrent request) | - |
| **logical batch** | Max tokens per `llama_decode()` call | `-b` / `n_batch` |
| **chunk** | Physical processing unit within a logical batch | `-ub` / `n_ubatch` |
| **chunk_size** | Size of one chunk (= `n_ubatch`) | `-ub` |
| **memory set** | Selected KV tokens (Local + Heavy) for inter-chunk attention | - |
| **Phase 1** | Parallel intra-chunk attention for all chunks in a logical batch | - |
| **Phase 2** | Sequential inter-chunk attention + fusion | - |

**Important**: "batch" in H2O context always means **logical batch** (`n_batch`), 
not concurrent request count. We assume single-request inference (batch=1 in the 
traditional sense).

## Implementation Plan v1.0

---

## 1. Problem Statement and Method Overview

### 1.1 Problem Definition

**Target Scenario**: Edge device LLM inference with batch=1, where prefill latency (TTFT) is the dominant bottleneck.

**Core Issue**: Standard causal self-attention exhibits $O(N^2)$ computational complexity. When combined with memory-constrained chunked prefill (processing tokens in blocks of size $S$), the observed throughput (tokens/s) degrades as prompt length increases:

$$\text{Total FLOPs} \propto \sum_{t=1}^{N} t = \frac{N(N+1)}{2} = O(N^2)$$

**Empirical Observation**: With chunk size $S=1024$, processing a 4096-token prompt requires attending to progressively larger KV caches (1024, 2048, 3072, 4096), causing average throughput to decrease monotonically.

### 1.2 Proposed Method: Block-Sparse Causal Attention Approximation

We replace full causal attention with a **block-sparse approximation** that maintains:
- **Intra-chunk**: Full causal attention within each chunk. In this architecture, given that our prefill phase no longer relies on sequential dependencies, attention calculation within each intra-chunk iteration is executed in parallel. For instance, a 4096-token prompt is segmented into four 1024-token chunks, and these four chunks initially undergo internal parallel prefill computation (thereby augmenting throughput).
  - **Block-Diagonal Causal Mask**: To enable parallel processing of all chunks in Phase 1, we use a block-diagonal causal mask. For each query position $i$, keys are only visible in $[\text{chunk\_start}(i), i]$ where $\text{chunk\_start}(i) = \lfloor i/S \rfloor \cdot S$. This ensures:
    - Within each chunk: standard causal masking (token at position $i$ can only attend to positions $\leq i$)
    - Across chunks: no attention (query in chunk $c$ cannot see keys from chunk $c' \neq c$ during Phase 1)
    - Result: All chunks can compute intra-attention simultaneously without cross-chunk dependencies
- **Inter-chunk**: Attention only to a bounded subset $\mathcal{M}_c$ of historical KV pairs

#### 1.2.1 Notation and llama.cpp Parameter Mapping

**Symbol Definitions**:
- $N$: Total prompt length (user input tokens)
- $B$: Logical batch size (`-b` / `n_batch`, default 2048) - max tokens per `llama_decode()` call
- $S$: Chunk size (`-ub` / `n_ubatch`, default 512) - physical processing unit
- $k_B = \lceil B/S \rceil$: Number of chunks per logical batch (e.g., 4 when $B=4096, S=1024$)
- $n_B = \lceil N/B \rceil$: Number of logical batches for full prompt
- $k = \lceil N/S \rceil$: Total number of chunks (global)
- $C_c$: Token indices in chunk $c$ (global indexing across all logical batches)
- $L$: Local window size (hyperparameter, default 256)
- $H$: Heavy hitter budget (hyperparameter, default 256)
- $M = L + H$: Total memory size (must satisfy **$M < S$**)

**llama.cpp Parameter Mapping**:

| Our Term | llama.cpp Flag | Default | Description |
|----------|----------------|---------|-------------|
| Chunk size ($S$) | `-ub` / `--ubatch-size` | 512 | Physical batch: tokens processed in one compute pass |
| Logical batch ($B$) | `-b` / `--batch-size` | 2048 | Max tokens per `llama_decode()` call |
| Local window ($L$) | `--h2o-local` | 256 | Recent tokens always retained |
| Heavy budget ($H$) | `--h2o-heavy` | 256 | Score-based token budget |

**Recommended Configuration** (for this plan's examples):
```bash
-b 4096 -ub 1024 --h2o-local 256 --h2o-heavy 256
# B=4096, S=1024 → k_B = 4 chunks per logical batch




#### 1.2.2 Core Idea: Block-Accumulated Attention Fusion

**Key Insight**: For each query token $i$ in chunk $C_c$, the target key set is:

$$\mathcal{K}(i) = \underbrace{\{j \in C_c : j \leq i\}}_{\text{intra-chunk (causal)}} \cup \underbrace{\mathcal{M}_{c-1}}_{\text{inter-chunk (memory)}}$$

Rather than computing attention over $\mathcal{K}(i)$ in a single pass, we perform **two separate attention computations** that are mathematically equivalent to a single attention over the combined key set:

| Pass | Query | Key Set | Computation Size | Mask |
|------|-------|---------|------------------|------|
| **Intra** | $Q(C_c)$ | $K(C_c), V(C_c)$ | $S \times S$ | Causal (triangular) |
| **Inter** | $Q(C_c)$ | $K(\mathcal{M}_{c-1}), V(\mathcal{M}_{c-1})$ | $S \times M$ | None (all visible) |

**Fusion via Online Softmax**: The two attention passes are **fused** using the online softmax technique. This ensures that the final output is mathematically identical to computing:

$$\text{Attn}(Q_i) = \text{softmax}\left(\frac{Q_i \cdot [K_{\text{intra}}; K_{\text{inter}}]^T}{\sqrt{d}}\right) \cdot [V_{\text{intra}}; V_{\text{inter}}]$$

The online softmax maintains running statistics $(m, \ell, o)$ that allow combining partial softmax results from multiple key blocks into an exact final result. This is the same principle used in FlashAttention for tiled computation, but here applied to fuse structurally different attention patterns (intra vs inter).

**Execution Schedule (Prefill)**:
- **Phase 1 (Parallel)**: All chunks compute **intra-attention** simultaneously:
  - $Q(C_c) \times K(C_c)^T$ with causal mask → store intra online-softmax state $(m,l,o)$
  - Initialize scores for tokens in $C_c$ from intra-attention column sums
- **Phase 2 (Sequential)**: For each chunk $C_c$ (where $c \geq 1$):
  1. Compute **inter-attention**: $Q(C_c) \times K(\mathcal{M}_{c-1})^T$ (no mask)
  2. **Update scores** for tokens in $\mathcal{M}_{c-1}$ using inter-attention weights
  3. **Fuse** inter state + cached intra $(m,l,o)$ via online softmax to get final output

#### 1.2.3 Memory Set Construction (H2O-like Selection)

The memory set $\mathcal{M}_c$ (used by chunk $C_{c+1}$) has a fixed size of $M = L + H$ tokens, with **$M < S$** (chunk size), composed of:

$$\mathcal{M}_c = \underbrace{\text{Local}_c}_{\text{forced retention}} \cup \underbrace{\text{Heavy}_c}_{\text{score-based selection}}$$

**Local Window** (size $L = 256$):
- Definition: The **tail $L$ tokens** of chunk $C_c$
- Formula: $\text{Local}_c = \{(c+1)S - L, \ldots, (c+1)S - 1\}$
- Rationale: Recent tokens are almost always relevant (temporal locality)

**Heavy Hitters** (size $H = 256$):
- Definition: Top-$H$ tokens by accumulated attention score from the **candidate set**
- Candidate set: $(\mathcal{M}_{c-1} \cup C_c) \setminus \text{Local}_c$
- This allows **cross-chunk propagation**: a token from $C_0$ that remains important can survive into $\mathcal{M}_1, \mathcal{M}_2, \ldots$

#### 1.2.4 Score Update Mechanism (Chunk-Level, Not Token-Level)

**Critical Difference from Standard H2O**: In decode-phase H2O, scores are updated after **every generated token**. In our prefill method, scores are **initialized once per chunk** from intra-attention, and **accumulated once per chunk** after the inter-attention pass.

**When Scores Update**:
- **Initialization** (Phase 1): After computing intra-attention for chunk $C_c$, initialize scores for tokens in $C_c$
- **Accumulation** (Phase 2): After computing inter-attention for chunk $C_c$ against memory $\mathcal{M}_{c-1}$

**What Gets Updated**:
- **Initialization**: Tokens in the current chunk $C_c$
- **Accumulation**: Only tokens in the current memory set $\mathcal{M}_{c-1}$

**Update Formula** (per-head, per-layer):
$$\text{score}(j) \mathrel{+}= \sum_{i \in C_c} A_{\text{inter}}(i, j), \quad \forall j \in \mathcal{M}_{c-1}$$

where $A_{\text{inter}}$ is the attention weight matrix from inter-attention.

**Interpretation**: Each inter-attention pass is a **voting event** where queries in $C_c$ vote for which memory tokens are useful. Tokens that receive strong attention accumulate higher scores and are more likely to be retained as heavy hitters.

#### 1.2.5 Complete Processing Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 1 (PARALLEL): INTRA-ATTN FOR ALL CHUNKS           │
├─────────────────────────────────────────────────────────────────────┤
│  • Split prompt into chunks C₀..Cₖ₋₁                                │
│  • For each chunk C_c in parallel:                                  │
│      - Compute intra(C_c) with causal mask                           │
│      - Store K/V + intra online-softmax state (m, l, o)               │
│      - Initialize scores from intra(C_c) column-sum                  │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  BUILD M₀ (READY FOR C₁ INTER-ATTN)                  │
├─────────────────────────────────────────────────────────────────────┤
│  • Local₀ = C₀[768:1024] (tail 256 tokens)                          │
│  • Heavy₀ = Top-256 from C₀[0:768] by intra scores                  │
│  • M₀ = Local₀ ∪ Heavy₀                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│          PHASE 2 (SEQUENTIAL): INTER-ATTN + FUSION (c ≥ 1)           │
├─────────────────────────────────────────────────────────────────────┤
│  For each chunk C_c in order (c = 1, 2, ...):                        │
│                                                                     │
│  Step 1: Inter-attention                                            │
│      • Q(C_c) × K(M_{c-1})ᵀ → A_inter [S × M]                       │
│                                                                     │
│  Step 2: Score Update (memory only)                                 │
│      • For each j ∈ M_{c-1}: score(j) += Σᵢ A_inter(i,j)            │
│                                                                     │
│  Step 3: Online Softmax Fusion                                      │
│      • Initialize state from inter-attention (m/l + o_inter)         │
│      • Fuse with cached intra state (m/l/o)                          │
│      • Fuse → final attention output for C_c                         │
│                                                                     │
│  Step 4: Build M_c for next chunk                                   │
│      • Local_c = tail of C_c (recent tokens)                        │
│      • Candidates = (M_{c-1} ∪ C_c[0:S-L])                          │
│      • Heavy_c = Top-H from Candidates by score                      │
│      • M_c = Local_c ∪ Heavy_c                                      │
│      • Drop non-selected tokens (keep only M_c for next step)       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    DECODE PHASE (AFTER PREFILL)                     │
├─────────────────────────────────────────────────────────────────────┤
│  H2O sparse prefill is inactive during decode; attention is full    │
│  causal and no H2O score/memory updates are applied                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ PREFILL → DECODE TRANSITION                                  │   │
│  │                                                              │   │
│  │ After final chunk Cₖ₋₁ (which may be partial):              │   │
│  │   1. Inter-attention: Q(Cₖ₋₁) × K(Mₖ₋₂)ᵀ                    │   │
│  │   2. Online softmax fusion with intra-attention              │   │
│  │   3. Store final chunk KV at positions [chunk_start, N)      │   │
│  │   4. Memory set updates are no longer consumed in decode      │   │
│  │                                                              │   │
│  │ KV cache state at decode start:                              │   │
│  │   • All N prompt positions have KV stored                    │   │
│  │   • H2O scoring/selection is frozen (not used in decode)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ DECODE EXECUTION (for each generated token tᵢ)              │   │
│  │                                                              │   │
│  │ Detection: ubatch.n_tokens == 1                              │   │
│  │                                                              │   │
│  │ For generated token at position N + i:                       │   │
│  │   1. Project Q, K, V for single token                        │   │
│  │   2. Apply RoPE with position = N + i                        │   │
│  │   3. **FULL attention**: Q × K[0:N+i]ᵀ (no H2O masking)      │   │
│  │   4. Store K(tᵢ), V(tᵢ) at position N + i in KV cache        │   │
│  │   5. No H2O score updates, no memory set changes             │   │
│  │                                                              │   │
│  │ Implementation: build_attn_h2o path may be used, but the mask │   │
│  │ collapses to standard causal attention when n_tokens <= S,   │   │
│  │ and score updates are skipped (ubatch.n_tokens == 1)          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Rationale:                                                         │
│      • Decode is O(N+t) per token, not the O(N²) bottleneck         │
│      • Full KV storage ensures maximum generation quality           │
│      • Simpler implementation - reuse existing decode path entirely │
│      • H2O overhead would slow decode with no meaningful benefit    │
└─────────────────────────────────────────────────────────────────────┘

#### 1.2.8 Final Chunk and Decode Transition Details

**Final Chunk Handling**:

The final chunk of prefill may not be a full `chunk_size` tokens:

```
Example: N = 3500, chunk_size = 1024

Chunk 0: positions [0, 1023]     - 1024 tokens (full)
Chunk 1: positions [1024, 2047]  - 1024 tokens (full)
Chunk 2: positions [2048, 3071]  - 1024 tokens (full)
Chunk 3: positions [3072, 3499]  - 428 tokens (partial, final)
```

**Processing the final chunk**:
1. Compute intra-attention for the 428-token partial chunk
2. Compute inter-attention against memory set M₂
3. Fuse via online softmax
4. Store KV for positions [3072, 3499]
5. **Skip** building M₃ - there's no next prefill chunk to use it
6. **Skip** advancing chunk state - prefill is complete

**KV Cache State at Decode Start**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Position:  0    1024   2048   3072   3500                       │
│            ├─────┼──────┼──────┼──────┤                         │
│ Content:   │ C₀  │  C₁  │  C₂  │ C₃   │  ← All KV stored        │
│            └─────┴──────┴──────┴──────┘                         │
│                                                                 │
│ During prefill, attention was sparse (via memory sets).         │
│ During decode, attention is FULL over [0, current_pos).         │
│                                                                 │
│ Decode tokens will be appended starting at position 3500:       │
│            ├─────┼──────┼──────┼──────┼─┬─┬─┬─┬─...             │
│            │ C₀  │  C₁  │  C₂  │ C₃   │t₁│t₂│t₃│t₄│             │
│            └─────┴──────┴──────┴──────┴─┴─┴─┴─┴─...             │
│                                        ↑                        │
│                                   Decode tokens                 │
│                                   (full attention,              │
│                                    full KV storage)             │
└─────────────────────────────────────────────────────────────────┘
```

**Why No H2O During Decode?**

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Tokens per call | N (thousands) | 1 |
| Complexity | O(N²) with chunking | O(N+t) per token |
| Bottleneck | Yes (TTFT) | No (TPS limited by other factors) |
| H2O benefit | High (quadratic → linear) | Minimal (already linear) |
| Quality risk | Acceptable (sparse approx) | Unacceptable (generation quality) |

**Implementation Simplicity**:

By using the standard decode path:
- No changes to existing decode logic
- No additional H2O state management during decode
- No risk of H2O bugs affecting generation
- Easy to verify correctness (decode unchanged from baseline)

```

#### 1.2.6 Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Local = Forced Recent** | Recent window tokens are always retained regardless of score |
| **Heavy = Future-Validated** | Heavy hitters are tokens that "future queries proved useful" |
| **Chunk-Level Updates** | Scores update once per chunk, not per token (efficiency) |
| **Cross-Chunk Propagation** | Important tokens can survive across multiple chunk boundaries |
| **Per-Head Independence** | Selection and scoring are independent per attention head per layer |

#### 1.2.7 Complexity Analysis

| Component | Full Attention | Block-Sparse (Ours) |
|-----------|---------------|---------------------|
| Intra-chunk (all chunks) | - | $k \cdot \frac{S(S+1)}{2} = O(NS)$ |
| Inter-chunk (chunks 1 to k-1) | - | $(k-1) \cdot S \cdot M = O(NM)$ |
| **Total** | $O(N^2)$ | $O(N(S + M))$ |

**Example** ($N=4096, S=1024, M=512$):

| Metric | Full Attention | Block-Sparse |
|--------|---------------|--------------|
| Intra dot products | - | $4 \times \frac{1024 \times 1025}{2} = 2,099,200$ |
| Inter dot products | - | $3 \times 1024 \times 512 = 1,572,864$ |
| **Total dot products** | $\frac{4096 \times 4097}{2} = 8,390,656$ | $3,672,064$ |
| **Reduction** | 1.0× | **2.3×** |

As $N$ grows, the benefit increases:
- $N=8192$: Full = 33.6M, Ours = 7.3M → **4.6× reduction**
- $N=16384$: Full = 134M, Ours = 14.7M → **9.1× reduction**

### 1.3 Key Engineering Constraints

1. **Per-Head, Per-Layer Independence**: H2O scoring and selection operate independently for each attention head in each layer
2. **Global RoPE Positions**: Position embeddings use absolute indices $\text{pos}(t) = t$, not chunk-relative
3. **Online Softmax Accumulation**: Numerically stable combination of intra-chunk and inter-chunk attention via $(m, \ell, o)$ state
4. **Memory Efficiency**: Avoid frequent allocation/deallocation; use pre-allocated buffers with index-based management

---

## 2. Implementation Plan

### Overview: Task Dependency Graph

```
[Task 1: Core Data Structures]
         │
         ▼
[Task 2: Intra-Chunk Attention] ──┐
         │                        │
         ▼                        │
[Task 3: Score Tracking]          │
         │                        │
         ▼                        │
[Task 4: Memory Selection]        │
         │                        │
         ▼                        ▼
[Task 5: Inter-Chunk Attention + Online Softmax]
         │
         ▼
[Task 6: Full Pipeline Integration]
         │
         ▼
[Task 7: CUDA Memory Optimization]
         │
         ▼
[Task 8: Benchmarking & Validation]
```

---

### Task 1: Core Data Structures and KV Cache Management

**Objective**: Design memory-efficient data structures for chunked KV cache with O(1) access patterns.

#### Subtasks

**1.1 KV Cache Buffer Design**
- Pre-allocate contiguous buffers for maximum sequence length
- Shape: `[num_layers, 2, num_heads, max_seq_len, head_dim]` (2 for K and V)
- Use index tracking rather than dynamic resizing

**1.2 Memory Set Index Structure**
- Per-layer, per-head index arrays for heavy hitter positions
- Shape: `[num_layers, num_heads, H]` for heavy hitter indices
- Local window: computed on-the-fly (no storage needed)

**1.3 Score Accumulator**
- Per-layer, per-head importance scores
- Shape: `[num_layers, num_heads, max_seq_len]`
- Initialize to zero; accumulate during attention

**1.4 Chunk Metadata**
- Track current chunk index, processed token count
- Boundary indices for each chunk

#### Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Memory layout correctness | Contiguous, cache-friendly | `torch.is_contiguous()`, stride analysis |
| Access pattern | O(1) index lookup | Profile with `torch.profiler` |
| Memory overhead | < 5% additional vs baseline | Compare peak memory usage |
| Index bounds | No out-of-bounds access | Unit tests with edge cases (N=S, N=2S-1, etc.) |

#### Deliverables
```python
class ChunkedKVCache:
    """Pre-allocated KV cache with chunked access patterns."""
    
class MemorySetTracker:
    """Tracks heavy hitter indices and scores per head/layer."""
    
class ChunkManager:
    """Manages chunk boundaries and iteration."""
```

---

### Task 2: Intra-Chunk Causal Attention

**Objective**: Implement standard causal attention within a single chunk, with correct RoPE positions.

#### Subtasks

**2.1 RoPE Application with Global Positions**
- Input: chunk tokens with local indices $[0, S)$
- Apply RoPE using global positions $[cS, (c+1)S)$
- Verify position encoding continuity across chunk boundaries

**2.2 Causal Mask Construction**
- Standard lower-triangular mask within chunk
- Shape: `[S, S]` (reusable across chunks)

**2.3 Attention Computation**
- Standard scaled dot-product attention
- Output: attention output + attention weights (for score tracking)
- Use FlashAttention-style implementation if available

**2.4 KV Cache Population**
- Write K, V to pre-allocated buffer at correct global positions
- Verify no buffer overflow

#### Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Numerical correctness | Match reference impl | Compare vs `torch.nn.functional.scaled_dot_product_attention` |
| RoPE position correctness | Global positions | Unit test: chunk 1 position 0 should use global pos $S$ |
| Causal mask correctness | No future leakage | Gradient check: $\partial \text{out}_i / \partial \text{in}_j = 0$ for $j > i$ |
| KV cache write | Correct positions | Read back and compare |

#### Deliverables
```python
def apply_rope_global(q, k, chunk_idx, chunk_size, rope_freqs):
    """Apply RoPE with global absolute positions."""
    
def intra_chunk_attention(q, k, v, causal_mask) -> Tuple[Tensor, Tensor]:
    """Returns (output, attention_weights)."""
    
def populate_kv_cache(cache, k, v, chunk_idx, chunk_size):
    """Write K,V to pre-allocated cache at correct positions."""
```

---

### Task 3: Attention Score Tracking (H2O Statistics)

**Objective**: Track per-token importance scores for heavy hitter selection, with chunk-level (not token-level) updates.

#### Key Design: Chunk-Level Score Updates

**Difference from Standard H2O**: In decode-phase H2O, scores update after every generated token. In our prefill method:
- Scores update **once per chunk** after inter-attention
- Only **memory tokens** receive score updates from inter-attention
- **Current chunk tokens** receive initial scores from intra-attention

#### Implementation Notes (current code)
- H2O **prefill** activates only when the prompt spans **multiple chunks** (`n_tokens_all > n_ubatch`). Single‑chunk sequences use standard full attention.
- **Phase 1** uses `n_batch` (logical batch) with a **block‑diagonal causal mask** to process all chunks in parallel and caches intra online‑softmax state (`o/l/m`).
- After Phase‑1 colsum readback, scores are initialized for **all chunks** and **M₀** is built from chunk 0.
- **Phase 2** runs per‑chunk (`n_ubatch`). Inter‑attention is used only when `ubatch.pos[0] > 0` and memory is initialized; then scores accumulate, **M_c** is rebuilt, and `h2o_next_chunk()` advances the chunk state.

#### Subtasks

**3.1 Score Storage Structure**
- Shape: `[num_layers, num_heads, max_seq_len]`
- Initialize to zero
- Per-head, per-layer independence

**3.2 Initial Score Assignment (from Intra-Attention)**

After computing intra-attention for chunk $C_c$:
```python
# attention_weights: [num_heads, S, S] (after softmax, with causal mask)
# Column sum: how much attention each key position received
intra_scores = attention_weights.sum(dim=-2)  # [num_heads, S]
scores[:, :, chunk_start:chunk_end] = intra_scores
```

This provides initial scores for newly processed tokens, used as a proxy for "how useful this token is" when building the first memory set that includes them.

**3.3 Score Accumulation (from Inter-Attention)**

After computing inter-attention for chunk $C_c$ against memory $\mathcal{M}_{c-1}$:
```python
# inter_weights: [num_heads, S, M] (after softmax)
# Column sum: how much the current chunk attended to each memory token
score_delta = inter_weights.sum(dim=-2)  # [num_heads, M]

# Update only memory token scores
for head in range(num_heads):
    for m, token_idx in enumerate(memory_indices[head]):
        scores[layer, head, token_idx] += score_delta[head, m]
```

**Interpretation**: Each inter-attention is a **voting event**. Tokens that receive strong attention from new queries accumulate higher scores and are more likely to survive as heavy hitters.

**3.4 Score Lookup for Heavy Hitter Selection**

When building $\mathcal{M}_c$, look up scores for all candidates:
```python
candidate_scores = scores[layer, head, candidate_indices]
heavy_indices = topk(candidate_scores, k=H).indices
```

#### Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Initial score correctness | Column sum of intra-attention | Manual computation on small example |
| Update correctness | Column sum of inter-attention | Manual computation on small example |
| Index alignment | Scores at correct global positions | Unit test: chunk 1 scores start at index $S$ |
| Monotonic accumulation | Scores never decrease | `assert (new_scores >= old_scores).all()` |
| Numerical stability | No overflow/underflow | Check for inf/nan after long sequences |
| Per-head independence | Different heads have different scores | Compare scores across heads |
| Update timing | Scores update once per chunk | Log update counts, verify = num_chunks - 1 |

#### Deliverables
```python
class ScoreTracker:
    """
    Manages per-token importance scores for heavy hitter selection.
    
    Attributes:
        scores: Tensor[num_layers, num_heads, max_seq_len]
    
    Implementation Note (llama.cpp):
        In C++ implementation, scores are stored as per-layer tensors:
        - h2o_scores[il]: Tensor[kv_size, n_head] (BF16)
        - Access pattern: score_data[pos * n_head + head] or score_data[head * kv_size + pos]
        The conceptual shape [num_layers, num_heads, max_seq_len] is achieved via
        vector<ggml_tensor*> h2o_scores with one [kv_size, n_head] tensor per layer.
    """
    
    def __init__(self, num_layers: int, num_heads: int, max_seq_len: int):
        """Initialize score buffer to zeros."""
    
    def set_initial_scores(
        self,
        layer_idx: int,
        chunk_start: int,
        chunk_end: int,
        intra_attention_weights: Tensor,  # [num_heads, S, S]
    ):
        """Set initial scores for new chunk from intra-attention column sums."""
    
    def accumulate_scores(
        self,
        layer_idx: int,
        memory_indices: Tensor,  # [num_heads, M]
        inter_attention_weights: Tensor,  # [num_heads, S, M]
    ):
        """Accumulate scores for memory tokens from inter-attention column sums."""
    
    def get_scores(
        self,
        layer_idx: int,
        indices: Tensor,  # [num_heads, num_candidates] or List[Set[int]]
    ) -> Tensor:
        """Look up scores for candidate indices."""
```

---

### Task 4: Memory Set Selection (H2O-like Selection)

**Objective**: Build memory set $\mathcal{M}_c$ for chunk $C_{c+1}$ using Local + Heavy hitter selection with cross-chunk propagation.

#### Key Design: Cross-Chunk Propagation

The heavy hitter candidate set is:
$$\text{Candidates} = (\mathcal{M}_{c-1} \cup C_c) \setminus \text{Local}_c$$

This means tokens from earlier chunks (e.g., $C_0$) can survive into $\mathcal{M}_1, \mathcal{M}_2, \ldots$ if they continue receiving high attention scores.

#### Implementation Notes (current code)
- `h2o_build_memory_set()` is called **for each layer in Phase 1** to build **M₀** after intra colsum is available.
- **Critical**: The memory set must be built for **every layer** before setting `h2o_memory_initialized = true`. Since transformers are multi-layer architectures, each layer has its own `h2o_memory_indices` tensor. The global `h2o_memory_initialized` flag gates inter-attention for all layers, so it must only be set after all layers complete their memory set construction.
- In **Phase 2**, after each chunk's inter‑attention, scores accumulate and **M_c** is rebuilt; `h2o_next_chunk()` advances the chunk boundary.
- Local window uses `chunk_end - L` clamped to 0; candidates are `(prev_memory ∪ current_chunk) \ local`.

#### Subtasks

**4.1 Local Window Extraction**
- Definition: Tail $L$ tokens of the current chunk
- Indices: $\{(c+1)S - L, \ldots, (c+1)S - 1\}$
- Edge case: Chunk 0's local window is just $\{S - L, \ldots, S - 1\}$
- Edge case: Partial final chunk where chunk length < $L$

**4.2 Candidate Set Construction**
- Start with previous memory set $\mathcal{M}_{c-1}$
- Add current chunk tokens (excluding local window): $C_c \setminus \text{Local}_c$
- For chunk 0: candidates = $C_0 \setminus \text{Local}_0 = \{0, \ldots, S - L - 1\}$

**4.3 Heavy Hitter Selection (Per-Head, Per-Layer)**
- For each (layer, head) pair:
  - Get scores for all candidate tokens
  - Select top-$H$ by accumulated score
- Important: Selection is **independent** per head—different heads may select different tokens

**4.4 Memory Set Assembly**
- Combine: $\mathcal{M}_c = \text{Local}_c \cup \text{Heavy}_c$
- Total size: $M = L + H$ (with $M < S$)
- Sort indices for coalesced memory access (important for CUDA performance)

**4.5 KV Gather Operation**
- Input: Full KV cache `[num_layers, 2, num_heads, max_seq_len, head_dim]`
- Output: Memory KV `[num_layers, 2, num_heads, M, head_dim]`
- Per-head indexing: Each head gathers from its own selected indices

#### Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Local correctness | Exactly tail $L$ tokens | `assert local_indices == range(chunk_end - L, chunk_end)` |
| Heavy selection | Top-$H$ by score | Manual sort comparison |
| Cross-chunk propagation | Old tokens can survive | Unit test: token from $C_0$ with high score appears in $\mathcal{M}_2$ |
| Index bounds | All indices in $[0, \text{current\_end})$ | Assert checks |
| No overlap | $\text{Local} \cap \text{Heavy} = \emptyset$ | Set intersection check |
| Per-head independence | Different heads can select different tokens | Verify head 0 and head 1 have different heavy sets |
| Sorted output | Indices in ascending order | `torch.all(idx[1:] > idx[:-1])` |
| Memory size | Exactly $M = L + H$ | `assert len(memory_indices) == M` |

#### Deliverables
```python
def get_local_window_indices(
    chunk_end: int,
    local_window_size: int,
) -> Tensor:
    """Return indices for local window (tail L tokens of current chunk)."""

def get_heavy_hitter_candidates(
    prev_memory_indices: Tensor,  # [num_heads, M] or None for chunk 0
    chunk_start: int,
    chunk_end: int,
    local_indices: Tensor,
) -> List[Set[int]]:
    """
    Return candidate sets per head.
    Candidates = (prev_memory ∪ current_chunk) \ local
    """

def select_heavy_hitters(
    scores: Tensor,           # [num_heads, max_seq_len]
    candidates: List[Set[int]],  # per head
    heavy_hitter_budget: int,
) -> Tensor:                  # [num_heads, H]
    """Select top-H indices per head from candidates by score."""

def build_memory_set(
    prev_memory: Optional[Tensor],  # [num_layers, num_heads, M] or None
    chunk_start: int,
    chunk_end: int,
    scores: Tensor,           # [num_layers, num_heads, max_seq_len]
    local_window_size: int,
    heavy_hitter_budget: int,
) -> Tensor:                  # [num_layers, num_heads, M]
    """
    Build memory set for next chunk.
    
    Returns sorted indices per layer per head.
    """

def gather_memory_kv(
    kv_cache: Tensor,         # [num_layers, 2, num_heads, max_seq_len, head_dim]
    memory_indices: Tensor,   # [num_layers, num_heads, M]
                              # Implementation Note: In llama.cpp, stored as per-layer tensors:
                              # h2o_memory_indices[il]: Tensor[M, n_head] (I32)
                              # Access: mem_idx_data[m * n_head + head] or mem_idx_data[head * M + m]
) -> Tuple[Tensor, Tensor]:   # K: [num_layers, num_heads, M, head_dim], V: same
    """
    Gather K, V for memory set.
    
    Note: Uses advanced indexing; indices should be sorted for better cache performance.
    """
```

---

### Task 5: Attention Fusion with Online Softmax

**Objective**: Fuse inter-chunk and intra-chunk attention using numerically stable online softmax to produce mathematically equivalent output to single-pass attention over the combined key set.

#### Background: Why Online Softmax?

For query $q_i$, we want:
$$\text{Attn}(q_i) = \text{softmax}\left(\frac{q_i [K_{\text{inter}}; K_{\text{intra}}]^T}{\sqrt{d}}\right) [V_{\text{inter}}; V_{\text{intra}}]$$

But we compute inter-attention and intra-attention separately. Online softmax lets us combine partial results exactly.

#### Subtasks

**5.1 Online Softmax State Definition**
- State: $(m, \ell, o)$ per query position, per head
  - $m \in \mathbb{R}$: running max of logits (for numerical stability)
  - $\ell \in \mathbb{R}$: running sum of $\exp(\text{logit} - m)$
  - $o \in \mathbb{R}^{d}$: running weighted sum of values

**5.2 Processing Order (Critical)**

For chunks $c \geq 1$, the order is:
1. **Inter-attention first**: $Q(C_c) \times K(\mathcal{M}_{c-1})^T$ → initialize online softmax state
2. **Intra-attention second**: $Q(C_c) \times K(C_c)^T$ with causal mask → update online softmax state
3. **Finalize**: Compute $o / \ell$ to get final output

This order allows score updates for memory tokens to happen before building the next memory set.

**Implementation note (llama.cpp)**:
- Phase 1 computes intra logits once and exports cached $(m, \ell, o)$ for the full prompt.
- Phase 2 does **not** recompute intra logits; it fuses inter state with cached intra $(m, \ell, o)$ using max/scale factors.

**5.3 State Initialization from Inter-Attention**
```python
# After computing inter-attention logits [num_heads, chunk_len, M]
inter_logits = Q @ K_mem.T / sqrt(d)  # [H, S, M]

# Initialize state
m = inter_logits.max(dim=-1)  # [H, S]
exp_logits = exp(inter_logits - m.unsqueeze(-1))  # [H, S, M]
l = exp_logits.sum(dim=-1)  # [H, S]
o = einsum('hsm,hmd->hsd', exp_logits, V_mem)  # [H, S, d]
```
Implementation detail: the code may reuse the already-computed inter output as `o` and only uses `inter_logits` for `m` and `l`.

**5.4 State Update from Intra-Attention**
```python
# After computing intra-attention logits [num_heads, chunk_len, chunk_len]
intra_logits = Q @ K.T / sqrt(d)  # [H, S, S]
intra_logits = apply_causal_mask(intra_logits)  # -inf for future positions

# Block statistics
m_new = intra_logits.max(dim=-1)  # [H, S]
exp_logits_new = exp(intra_logits - m_new.unsqueeze(-1))  # [H, S, S]
l_new = exp_logits_new.sum(dim=-1)  # [H, S]
o_new = einsum('hss,hsd->hsd', exp_logits_new, V)  # [H, S, d]

# Merge with existing state
m_merged = maximum(m, m_new)
correction_old = exp(m - m_merged)
correction_new = exp(m_new - m_merged)

l = l * correction_old + l_new * correction_new
o = o * correction_old.unsqueeze(-1) + o_new * correction_new.unsqueeze(-1)
m = m_merged
```

**Phase-2 fusion with cached intra (actual code path)**:
```python
m_total = maximum(m_inter, m_intra)
scale_inter = exp(m_inter - m_total)
scale_intra = exp(m_intra - m_total)
l_total = l_inter * scale_inter + l_intra * scale_intra
o_total = (o_inter * l_inter * scale_inter + o_intra * l_intra * scale_intra) / l_total
```

**5.5 Finalization**
```python
output = o / l.unsqueeze(-1)  # [H, S, d]
```

**5.6 Special Case: Chunk 0**
- No inter-attention (no memory yet)
- Only intra-attention with causal mask
- Standard softmax (no online accumulation needed)

#### Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Mathematical equivalence | Exact match to reference | Compare vs `softmax(cat([inter_logits, intra_logits]))` on small example |
| Numerical stability | No inf/nan | Stress test with logits in range [-100, 100] |
| Causal correctness | No future leakage in intra | Gradient check: $\partial \text{out}_i / \partial K_j = 0$ for $j > i$ |
| Output shape | `[num_heads, chunk_len, head_dim]` | Shape assertion |
| Order independence | Same result regardless of inter/intra order | Unit test both orders produce same output |

#### Deliverables
```python
@dataclass
class OnlineSoftmaxState:
    m: Tensor  # [num_heads, seq_len] - running max
    l: Tensor  # [num_heads, seq_len] - running denominator
    o: Tensor  # [num_heads, seq_len, head_dim] - running numerator

def init_online_softmax_from_inter(
    inter_logits: Tensor,  # [H, S, M]
    V_mem: Tensor,         # [H, M, d]
) -> OnlineSoftmaxState:
    """Initialize online softmax state from inter-chunk attention."""

def update_online_softmax_with_intra(
    state: OnlineSoftmaxState,
    intra_logits: Tensor,  # [H, S, S] with causal mask applied
    V: Tensor,             # [H, S, d]
) -> OnlineSoftmaxState:
    """Update state with intra-chunk attention contribution."""

def finalize_online_softmax(state: OnlineSoftmaxState) -> Tensor:
    """Return final attention output: o / l."""

def fused_chunk_attention(
    Q: Tensor,           # [H, S, d]
    K: Tensor,           # [H, S, d] - current chunk
    V: Tensor,           # [H, S, d] - current chunk
    K_mem: Tensor,       # [H, M, d] - memory
    V_mem: Tensor,       # [H, M, d] - memory
    causal_mask: Tensor, # [S, S]
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns (output, inter_weights, intra_weights).
    
    inter_weights needed for score updates.
    intra_weights needed for current chunk score initialization.
    """
```

---

### Task 6: Full Pipeline Integration

**Objective**: Integrate all components into a complete chunked prefill pipeline.

#### Subtasks

**6.1 Single-Layer Chunked Attention**
- Orchestrate: project Q,K,V → intra-chunk attn → score tracking → memory selection → inter-chunk attn → online softmax merge
- Handle layer-specific KV cache and score buffers

**6.2 Multi-Layer Forward Pass**
- Sequential layer processing (standard)
- Ensure KV cache and scores are properly indexed by layer

**6.3 Chunk Iteration Loop**
- Process chunks sequentially: 0, 1, 2, ..., k-1
- Chunk 0: intra-chunk only (no history)
- Chunks 1+: intra-chunk + inter-chunk

**6.4 Edge Case Handling**
- Final partial chunk (when $N \mod S \neq 0$)
- Very short sequences ($N < S$)
- Single chunk ($k = 1$)

#### Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| End-to-end correctness | Reasonable output quality | Perplexity on validation set |
| No gradient accumulation bugs | Fresh state per forward | Compare two identical inputs |
| Edge case handling | No crashes | Unit tests for all edge cases |
| Memory not leaking | Constant peak memory | Profile across multiple batches |

#### Deliverables
```python
class ChunkedSparseAttentionLayer(nn.Module):
    """Single transformer layer with chunked sparse attention."""
    
class ChunkedSparseTransformer(nn.Module):
    """Full transformer with chunked sparse prefill."""
    
def chunked_prefill(
    model: ChunkedSparseTransformer,
    input_ids: Tensor,
    chunk_size: int = 1024,
    local_window: int = 256,
    heavy_hitter_budget: int = 256,
) -> Tensor:
    """Main entry point for chunked sparse prefill."""
```

---

### Task 7: CUDA Memory Optimization

**Objective**: Optimize memory access patterns for GPU efficiency.

#### Subtasks

**7.1 Coalesced Memory Access for KV Gather**
- Sort memory set indices before gather
- Consider page-level grouping (e.g., 16-token pages)
- Profile gather throughput vs random access baseline

**7.2 Buffer Reuse and Pre-allocation**
- Allocate all buffers at initialization
- Use views and index-based access, not dynamic allocation
- Implement buffer pool for temporary tensors

**7.3 Memory Layout Optimization**
- Analyze access patterns with `torch.profiler`
- Consider alternative layouts: `[seq_len, num_heads, head_dim]` vs `[num_heads, seq_len, head_dim]`
- Benchmark different layouts

**7.4 Kernel Fusion Opportunities**
- Fuse RoPE + attention score computation
- Fuse score accumulation with attention
- Consider custom CUDA kernels for hot paths

#### Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Memory bandwidth utilization | > 60% of theoretical | `nvidia-smi` + roofline analysis |
| No unnecessary allocations | 0 allocs in hot path | `torch.cuda.memory_stats()` |
| Cache hit rate | > 80% L2 cache | Nsight Compute profiling |
| Gather efficiency | > 50% of contiguous read | Benchmark vs baseline |

#### Deliverables
```python
class OptimizedKVCache:
    """Memory-optimized KV cache with coalesced access patterns."""
    
def sorted_gather(cache: Tensor, indices: Tensor) -> Tensor:
    """Gather with sorted indices for better memory access."""
    
# Optional: Custom CUDA kernels
# fused_rope_attention_kernel.cu
# score_accumulate_kernel.cu
```

---

### Task 8: Benchmarking and Quality Validation

**Objective**: Comprehensive evaluation of performance and output quality.

#### Subtasks

**8.1 Performance Benchmarks**
- Metric: Prefill throughput (tokens/second) vs sequence length
- Baseline: Standard chunked prefill (no sparsity)
- Configurations: $N \in \{1024, 2048, 4096, 8192, 16384\}$
- Hardware: Target edge device (Jetson Orin) + development GPU

**8.2 Memory Benchmarks**
- Peak memory usage vs sequence length
- Memory breakdown: KV cache, scores, temporary buffers

**8.3 Quality Evaluation**
- Perplexity on held-out data (e.g., WikiText-2, C4)
- Compare: full attention vs sparse attention
- Ablation: vary $L$, $H$ parameters

**8.4 Attention Pattern Analysis**
- Visualize which tokens are selected as heavy hitters
- Verify attention sink behavior (early tokens)
- Layer-wise and head-wise analysis

#### Verification Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Throughput improvement | > 1.5× at N=4096 | Wall-clock timing |
| Perplexity degradation | < 5% relative | Eval on benchmark |
| Memory reduction | Controlled by $L+H$ | Profile memory |
| Heavy hitter stability | Consistent across runs | Jaccard similarity of selected indices |

#### Deliverables
```python
def benchmark_throughput(model, seq_lengths, num_runs=10) -> pd.DataFrame:
    """Benchmark prefill throughput across sequence lengths."""
    
def evaluate_perplexity(model, dataset, chunk_config) -> float:
    """Evaluate perplexity on dataset."""
    
def visualize_attention_patterns(model, input_ids, chunk_config):
    """Generate attention pattern visualizations."""
    
# benchmark_results/
#   throughput_comparison.png
#   memory_profile.png
#   perplexity_ablation.csv
#   attention_heatmaps/
```

---

## 3. Implementation Schedule and Dependencies

### Recommended Order

| Phase | Tasks | Est. Effort | Dependencies |
|-------|-------|-------------|--------------|
| **Phase 1: Foundation** | Task 1 | 2-3 days | None |
| **Phase 2: Intra-Chunk** | Task 2, 3 | 3-4 days | Task 1 |
| **Phase 3: Selection** | Task 4 | 2-3 days | Task 1, 3 |
| **Phase 4: Inter-Chunk** | Task 5 | 3-4 days | Task 2, 4 |
| **Phase 5: Integration** | Task 6 | 2-3 days | Tasks 1-5 |
| **Phase 6: Optimization** | Task 7 | 3-5 days | Task 6 |
| **Phase 7: Validation** | Task 8 | 2-3 days | Task 6 |

### Critical Path
```
Task 1 → Task 2 → Task 5 → Task 6 → Task 8
           ↓
         Task 3 → Task 4
                    ↓
                  Task 5
```

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Online softmax numerical issues | Extensive unit tests with extreme values |
| Heavy hitter selection suboptimal | Ablation study in Task 8; fallback to larger $H$ |
| CUDA memory bandwidth bottleneck | Profile early (Task 7); consider page-level selection |
| Quality degradation | Monitor perplexity throughout; add sink token handling if needed |

---

## Appendix A: Reference Implementation Pseudocode

**Note**: The production code uses a two-phase schedule (Phase 1 parallel intra for all chunks, Phase 2 sequential inter + fusion) and caches per-layer $(m, \ell, o)$ for intra. The pseudocode below is a conceptual single-pass view.

```python
def chunked_sparse_prefill(input_ids, model, config):
    """
    Main algorithm pseudocode.
    
    config: {
        chunk_size: S = 1024,
        local_window: L = 256,
        heavy_hitter_budget: H = 256,
        memory_size: M = L + H (M < S)
    }
    """
    N = len(input_ids)
    S = config.chunk_size
    L = config.local_window
    H = config.heavy_hitter_budget
    num_chunks = ceil(N / S)
    
    # Pre-allocate buffers (avoid dynamic allocation)
    kv_cache = allocate_kv_cache(model.num_layers, model.num_heads, N, model.head_dim)
    scores = zeros(model.num_layers, model.num_heads, N)  # importance scores
    
    hidden = model.embed(input_ids)
    
    # ========== PROCESS CHUNK 0 (initialization, no history) ==========
    chunk_hidden = hidden[0:S]
    for layer_idx, layer in enumerate(model.layers):
        Q, K, V = layer.project_qkv(chunk_hidden)
        Q, K = apply_rope(Q, K, positions=range(0, S))
        
        # Store in KV cache
        kv_cache[layer_idx, :, :, 0:S] = stack([K, V])
        
        # Intra-attention only (no memory yet)
        output, attn_weights = intra_chunk_attention(Q, K, V, causal_mask=True)
        
        # Initialize scores from intra-attention (proxy for future usefulness)
        # Column-sum: how much attention each key received
        scores[layer_idx, :, 0:S] = attn_weights.sum(dim=-2)  # sum over queries
        
        chunk_hidden = layer.output_proj(output) + chunk_hidden
        chunk_hidden = layer.ffn(chunk_hidden)
    
    hidden[0:S] = chunk_hidden
    
    # Build M₀ for chunk 1
    # Local₀ = tail L tokens of C₀
    # Heavy₀ = top-H from C₀[0:S-L] by score
    memory_indices = build_initial_memory(scores, S, L, H)  # per layer, per head
    
    # ========== PROCESS CHUNKS 1, 2, ..., k-1 ==========
    for chunk_idx in range(1, num_chunks):
        chunk_start = chunk_idx * S
        chunk_end = min((chunk_idx + 1) * S, N)
        chunk_len = chunk_end - chunk_start
        chunk_hidden = hidden[chunk_start:chunk_end]
        
        for layer_idx, layer in enumerate(model.layers):
            Q, K, V = layer.project_qkv(chunk_hidden)
            Q, K = apply_rope(Q, K, positions=range(chunk_start, chunk_end))
            
            # Store current chunk's KV in cache
            kv_cache[layer_idx, :, :, chunk_start:chunk_end] = stack([K, V])
            
            # ===== Step 1: Inter-attention (chunk queries → memory keys) =====
            # Gather memory KV (512 tokens)
            mem_idx = memory_indices[layer_idx]  # [num_heads, M]
            K_mem, V_mem = gather_kv(kv_cache[layer_idx], mem_idx)
            
            # Inter-attention: [chunk_len × M], no causal mask
            inter_logits = einsum('nhd,nhmd->nhm', Q, K_mem) / sqrt(head_dim)
            inter_weights = softmax(inter_logits, dim=-1)
            inter_output = einsum('nhm,nhmd->nhd', inter_weights, V_mem)
            
            # Initialize online softmax state from inter-attention
            state = init_online_softmax(inter_output, inter_logits)
            
            # ===== Step 2: Score Update (only memory tokens) =====
            # Each memory token j gets score += sum over queries of attention to j
            score_delta = inter_weights.sum(dim=-2)  # [num_heads, M]
            for h in range(num_heads):
                for m, j in enumerate(mem_idx[h]):
                    scores[layer_idx, h, j] += score_delta[h, m]
            
            # ===== Step 3: Intra-attention (chunk queries → chunk keys, causal) =====
            intra_logits = einsum('nhqd,nhkd->nhqk', Q, K) / sqrt(head_dim)
            intra_logits = apply_causal_mask(intra_logits)
            
            # Update online softmax state and fuse
            state = update_online_softmax(state, intra_logits, V)
            final_output = finalize_online_softmax(state)
            
            # Also accumulate intra-attention scores for current chunk
            intra_weights = softmax(intra_logits, dim=-1)
            scores[layer_idx, :, chunk_start:chunk_end] += intra_weights.sum(dim=-2)
            
            # Output projection and residual
            chunk_hidden = layer.output_proj(final_output) + chunk_hidden
            chunk_hidden = layer.ffn(chunk_hidden)
        
        hidden[chunk_start:chunk_end] = chunk_hidden
        
        # ===== Step 4: Build Mₖ for next chunk =====
        # Local = tail L tokens of current chunk
        local_indices = range(chunk_end - L, chunk_end)
        
        # Candidates = (previous memory ∪ current chunk) \ local
        # Heavy = top-H from candidates by score
        memory_indices = build_memory(
            prev_memory=memory_indices,
            current_chunk_range=(chunk_start, chunk_end),
            local_indices=local_indices,
            scores=scores,
            H=H
        )
    
    return hidden, kv_cache


def build_memory(prev_memory, current_chunk_range, local_indices, scores, H):
    """
    Build memory set for next chunk.
    
    Memory = Local (forced) ∪ Heavy (top-H by score)
    
    Candidates for Heavy = (prev_memory ∪ current_chunk) \ local
    This allows cross-chunk propagation of important tokens.
    """
    # Implementation per layer, per head
    new_memory = []
    for layer_idx in range(num_layers):
        layer_memory = []
        for head_idx in range(num_heads):
            # Candidate set (excluding local window)
            candidates = set(prev_memory[layer_idx][head_idx])
            candidates.update(range(*current_chunk_range))
            candidates -= set(local_indices)
            
            # Select top-H by score
            candidate_scores = [(j, scores[layer_idx, head_idx, j]) for j in candidates]
            candidate_scores.sort(key=lambda x: -x[1])  # descending
            heavy = [j for j, _ in candidate_scores[:H]]
            
            # Memory = local + heavy, sorted for coalesced access
            memory = sorted(list(local_indices) + heavy)
            layer_memory.append(memory)
        new_memory.append(layer_memory)
    
    return new_memory
```

---

## Appendix B: Key Equations Summary

### Sparse Key Set Definition
For query token $i$ in chunk $C_c$:
$$\mathcal{K}(i) = \{j \in C_c : j \leq i\} \cup \mathcal{M}_{c-1}$$

### Memory Set Construction
$$\mathcal{M}_c = \text{Local}_c \cup \text{Heavy}_c$$
$$\text{Local}_c = \{(c+1)S - L, \ldots, (c+1)S - 1\}$$
$$\text{Heavy}_c = \text{TopK}_H\left(\text{scores}[j] : j \in (\mathcal{M}_{c-1} \cup C_c) \setminus \text{Local}_c\right)$$

### Score Update Formulas

**Initial scores (from intra-attention, chunk $C_c$)**:
$$\text{score}(j) = \sum_{i \in C_c, i \geq j} A_{\text{intra}}(i, j), \quad \forall j \in C_c$$

**Score accumulation (from inter-attention, chunk $C_c$ → memory $\mathcal{M}_{c-1}$)**:
$$\text{score}(j) \mathrel{+}= \sum_{i \in C_c} A_{\text{inter}}(i, j), \quad \forall j \in \mathcal{M}_{c-1}$$

### Online Softmax Fusion

**State**: $(m, \ell, o)$ where $m$ = running max, $\ell$ = running denominator, $o$ = running numerator

**Initialize from inter-attention**:
$$m = \max_j s_{\text{inter},j}$$
$$\ell = \sum_j \exp(s_{\text{inter},j} - m)$$
$$o = \sum_j \exp(s_{\text{inter},j} - m) \cdot V_j$$

**Update with intra-attention**:
$$m' = \max(m, \max_j s_{\text{intra},j})$$
$$\ell' = \ell \cdot e^{m - m'} + \sum_j \exp(s_{\text{intra},j} - m')$$
$$o' = o \cdot e^{m - m'} + \sum_j \exp(s_{\text{intra},j} - m') \cdot V_j$$

**Finalize**:
$$\text{output} = o' / \ell'$$

### Complexity Analysis
$$C_{\text{full}} = O(N^2)$$
$$C_{\text{ours}} = O(N(S + M)) = O(N(S + L + H))$$

### Dimension Reference

| Symbol | Meaning | Default Value |
|--------|---------|---------------|
| $N$ | Total sequence length | 4096 |
| $S$ | Chunk size | 1024 |
| $L$ | Local window size | 256 |
| $H$ | Heavy hitter budget | 256 |
| $M$ | Memory size ($L + H$) | 512 |
| $k$ | Number of chunks ($\lceil N/S \rceil$) | 4 |

# Addition note
+ 说明：llama.cpp 提供 -b（n_batch）和 -ub（n_ubatch）。为与示例对齐，运行时使用 -b=4096、-ub=1024。
+ 原生 llama.cpp prefill 会把 -b 拆成多个 -ub **顺序**执行。
+ H2O 两阶段中：Phase 1 使用 block‑diagonal mask 一次性处理 n_batch（并行 intra），Phase 2 再按 n_ubatch 顺序处理。
    * **-b 逻辑批大小**默认 2048；示例中设为 4096
    * -ub 物理批大小 默认 512；原生/Phase 2 都按 -ub 顺序处理
