# Experiment 02: L1 sparsity sweep on `L, R`

## 1. Five evaluation targets

| Target | How this approach addresses it |
|--------|--------------------------------|
| **Short** — few components for high fidelity | Indirectly. L1 on `L,R` shrinks within-component pixel support, not the number of components. But sparser components tend to specialize, which can effectively shorten the dictionary at fixed fidelity. |
| **Interpretable** — each component readable as a discrete feature | **Primary target.** L1 pushes pixel patterns toward localized supports — the classic dictionary-learning prior. Components should look more like strokes or edge detectors than dense mixtures. |
| **Fidelity** — cosine(B, B̂) + accuracy preserved | Trade-off axis. Higher α erodes fidelity. The sweep `α ∈ {0.001, 0.01, 0.1}` is meant to map the curve, not pick a single point. |
| **Sharing** — features shared across classes encoded once | Inherited from CP. L1 on `L,R` doesn't change the cross-class structure (would need L1 on `D` for that). |
| **Orthogonality** — within-class non-orthogonality allowed | Inherited from CP. L1 actually *helps* here — sparse components with disjoint support are naturally non-orthogonal in the right way (no artifact negatives). |

## 2. Intervention level(s)

- [ ] Level 1 — Decomposition family. Same CP as baseline.
- [ ] Level 2 — Parameterization. Raw real-valued, same as baseline.
- [x] **Level 3 — Loss function.** Add L1 penalty: `loss = (1 − cos(B, B̂)) + α · (‖L‖₁ + ‖R‖₁)`. Sweep α.

## 3. Hypothesis

As α increases from 0.001 → 0.1, components transition from dense baseline-like patterns → spatially localized stroke/edge detectors → eventually too sparse to reconstruct `B` well (fidelity collapse). The interesting sweet spot is the largest α at which cosine ≥ 0.9 and accuracy stays within ~2% of baseline.

**Caveat (per `METHOD_REFERENCE.md` §7):** cosine loss is scale-invariant, so the *effective* L1 strength differs from L1-with-Frobenius literature numbers. The sweep characterizes our regime; absolute α values aren't directly comparable across loss choices.

## 4. Method

```python
for alpha in ALPHAS_L1:  # [0.001, 0.01, 0.1]
    fit_decomposition(model, alpha_l1=alpha, symmetric=False, nonneg=False)
```

Three runs, each 200 steps, fresh seed per α. Each run produces its own figure and metrics row in the summary DataFrame.

## 5. Status

- [x] Designed
- [ ] Implemented in `experiments/main_experiments.ipynb`
- [ ] Run
- **Figure**: `figures/fig_l1_0.001.png`, `figures/fig_l1_0.01.png`, `figures/fig_l1_0.1.png`

`Status: designed`

## 6. Results / Notes

To be filled after running. Per α: cosine, accuracy, mean nonzero fraction of `L+R`, qualitative interpretability read. Pick the best α to feed Experiments 3 and 4.
