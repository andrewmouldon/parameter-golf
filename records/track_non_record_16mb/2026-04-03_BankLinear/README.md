# BankLinear: Compositional Weight Sharing via Learned Basis Mixtures

## Overview

Modern transformers allocate a unique set of weights to every layer, leading to significant redundancy across depth. Many layers learn structurally similar transformations, yet parameters are not shared.

**BankLinear** replaces explicit per-layer weight matrices with **compositions over a shared bank of learned basis matrices**. Each layer constructs its weights dynamically using learned mixing coefficients, enabling parameter sharing across depth while preserving per-layer specialization.

The mixing mechanism is factorized into global and channel-wise terms. The global coefficient provides an easy optimization handle for tuning the overall influence of each basis at a given layer, while the channel-wise coefficient gives finer-grained control over how each output feature specializes its use of the shared bank.

## Method

BankLinear defines a shared bank of learned basis matrices:

- `B_i ∈ R^{d_out × d_in}`: shared basis matrices

Each layer constructs its effective weight matrix using a factorized mixing mechanism:

```text
W^(l)[o, :] = Σ_i α_global_i^(l) · α_channel_i^(l, o) · B_i[o, :]
```

where:

- `α_global^(l) ∈ R^{bank_size}`: global mixing coefficients for layer `l`
- `α_channel^(l) ∈ R^{bank_size × d_out}`: channel-wise modulation coefficients

This separates weight construction into two roles:

- **Global selection:** `α_global` tunes the overall contribution of each basis element for a given layer
- **Channel-wise modulation:** `α_channel` adjusts basis contributions per output feature

Together, these allow BankLinear to share structure across layers while preserving both layer-level and output-channel-level specialization.

---

## Pseudocode

```python
class BankLinear:
    # shared learned basis matrices
    bank = learned_tensor(bank_size, d_out, d_in)

    # per-layer mixing coefficients
    alpha_global = learned_matrix(num_layers, bank_size)
    alpha_channel = learned_tensor(num_layers, bank_size, d_out)

    def forward(x, layer_id):
        # compose this layer's weight matrix from the shared bank
        W = einsum(
            "b,bo,boi->oi",
            alpha_global[layer_id],      # global basis strength
            alpha_channel[layer_id],     # per-output-channel modulation
            bank                         # shared basis matrices
        )

        return linear(x, W)
```

---

## Initialization

Initialization is important for stable performance.

This PR uses a **depth-aware initialization** for the global mixing coefficients:

- early, middle, and late layers are biased toward different basis elements
- transitions between these regions are smooth
- the basis bank starts with an initial division of labor across depth while remaining fully learnable

This initialization was important for getting consistent gains. Fixed random projections were removed, since they degraded performance once depth-aware initialization was used.

---

## Setup

- Fixed 10k training steps
- Same architecture and training setup as the naive baseline
- BankLinear applied to QKV projections
- 3 learnable bases compose the bank
- These bases are mixed to compose 9 total layers
- Saved parameters are reinvested into a larger MLP
  - Baseline MLP expansion: 2.00×
  - BankLinear MLP expansion: 2.65×

---

## Results

All runs use identical settings across three seeds, building off of the original naive baseline.

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

On average, BankLinear improves both pre-quant and post-quant BPB while staying within the same size budget.

---

## Throughput

BankLinear introduces additional overhead from dynamic weight composition and the larger MLP.

In the current implementation, training is approximately **1.25× slower** than the baseline.

---

## Additional Experiments

Several related variants were tested:

- **LoRA-style adapters:** Allocating saved parameters to layer-specific adapters was less effective than increasing MLP capacity.
- **Output projections:** Applying BankLinear to projections that write directly into the residual stream significantly degraded performance. One likely reason is that errors introduced there directly affect the residual stream and accumulate across depth.
- **Random projections:** Fixed random projection matrices were helpful before depth-aware initialization, but consistently degraded performance once depth-aware initialization was used.
- **Head-wise bank construction:** Structuring the shared bank at the attention-head level consistently underperformed the standard formulation.

---

## Related Work Note

After developing this submission, I found that the core idea is closely related to **MASA (Matrix Atom Sharing in Attention)**, which also represents attention projection matrices as combinations of shared matrix atoms.

BankLinear differs in a few small-model-specific choices:

- it uses explicit depth-aware initialization for the mixing coefficients
- it adds channel-wise modulation on top of global basis mixing
- it focuses on reallocating saved projection parameters into a larger MLP under the 16MB budget

I did not know about MASA when developing this submission, but the similarity is useful evidence that cross-layer matrix sharing is a plausible direction rather than just a small model challenge-specific trick.
