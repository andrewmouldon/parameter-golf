# BankLinear: Compositional Weight Sharing via Learned Basis Mixtures

## Overview

Modern transformers allocate a unique set of weights to every layer, leading to significant redundancy across depth. Many layers learn structurally similar transformations, yet parameters are not shared.

**BankLinear** replaces explicit per-layer weight matrices with **compositions over a shared bank of learned basis matrices**. Each layer constructs its weights dynamically using learned mixing coefficients, enabling parameter sharing across depth while preserving per-layer specialization.

---
## Method

Instead of storing a weight matrix per layer, BankLinear defines a shared bank of basis matrices:

- `B_i ∈ R^{d_out × d_in}`: shared basis matrices

Each layer constructs its weights through a **factorized mixing mechanism**:

W^(l)[o, :] = Σ_i α_global_i^(l) · α_channel_i^(l, o) · B_i[o, :]

where:

- `α_global^(l) ∈ R^{bank_size}`: global mixing coefficients for layer `l`
- `α_channel^(l) ∈ R^{bank_size × d_out}`: channel-wise modulation

---

## Interpretation

This decomposition separates weight construction into two roles:

- **Global selection:**  
  `α_global` determines which basis elements are active at a given layer

- **Channel-wise modulation:**  
  `α_channel` adjusts contributions per output feature

This allows BankLinear to:

- share structure across layers
- specialize per layer
- adapt per output channel

without storing independent weight matrices.

---

## Initialization

Initialization is critical for stable performance.

We use a **depth-aware initialization** for the global mixing coefficients:
- early, middle, and late layers are biased toward different basis elements
- transitions between them are smooth

This provides an initial division of labor across depth while retaining full flexibility during training.

Without this structure, performance degrades significantly.

---

## Integration

BankLinear is applied to attention projections:

- Query, Key, and Value projections are constructed from the shared bank
- Each layer uses its own mixing coefficients

This enables parameter sharing across depth while maintaining expressivity where it matters most.

---

## Results

All runs are trained for 10k steps under identical settings across three seeds.  
BankLinear replaces QKV projections, and saved parameters are reinvested into a larger MLP (2.65× vs 2.00× baseline).

| Model | Seed | Pre-quant BPB ↓ | Post-quant BPB ↓ | Size (bytes) |
|-------|------|----------------:|-----------------:|-------------:|
| Baseline | 1337 | 1.2262 | 1.2328 | 15861272 |
| BankLinear | 1337 | **1.2210** | **1.2274** | 15766158 |
| Baseline | 42 | 1.2276 | 1.2343 | 15856563 |
| BankLinear | 42 | **1.2208** | **1.2269** | 15839677 |
| Baseline | 2025 | 1.2253 | 1.2321 | 15853892 |
| BankLinear | 2025 | **1.2204** | **1.2267** | 15829397 |
| **Average (Baseline)** | — | 1.2264 | 1.2331 | 15857242 |
| **Average (BankLinear)** | — | **1.2207** | **1.2270** | 15811744 |

---

## Additional Experiments

- **LoRA-style adapters:** Allocating saved parameters to layer-specific adapters was less effective than increasing MLP capacity.
- **Output projections:** Applying BankLinear to projections that write directly into the residual stream significantly degraded performance. We hypothesize this is due to the sensitivity of these layers: errors introduced here directly affect the residual stream and accumulate across depth.
- **Random projections:** We experimented with augmenting the shared bank with fixed random projection matrices. Without depth-aware initialization, this provided a performance benefit. However, once depth-aware initialization is used, we consistently observe degraded performance when including random projections. 
---

