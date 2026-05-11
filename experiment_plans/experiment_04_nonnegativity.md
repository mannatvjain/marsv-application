# Experiment 04: Non-negativity (squared parameterization) + L1

## 1. Five evaluation targets

| Target | How this approach addresses it |
|--------|--------------------------------|
| **Short** — few components for high fidelity | Likely worsened. Forcing non-negativity removes the model's ability to use cancellation, so more components may be needed to express the same `B`. Trade-off accepted in exchange for the interpretability gain. |
| **Interpretable** — each component readable as a discrete feature | **Primary target.** Non-negative components are *additive parts* — no red/blue cancellation, no XOR-style edge detectors. Components should look like positive masks (a stroke, a region) rather than contrast patterns. NMF-style parts decomposition for tensors. |
| **Fidelity** — cosine(B, B̂) + accuracy preserved | Will degrade some. Non-negativity is a hard constraint; the achievable cosine is bounded by the best non-negative approximation of `B`. We accept this drop if the components become clearly more nameable. |
| **Sharing** — features shared across classes encoded once | Inherited from CP. Non-negativity also forces `D` ≥ 0, so a component can only *push toward* classes, never *push away*. This actually sharpens class concentration. |
| **Orthogonality** — within-class non-orthogonality allowed | Inherited from CP, and arguably improved: non-negative sparse vectors with disjoint supports are naturally non-orthogonal in the right way (no negative artifact lobes). |

## 2. Intervention level(s)

- [ ] Level 1 — Decomposition family. Same CP. (Could combine with symmetric — flagged as future work.)
- [x] **Level 2 — Parameterization.** Squared parameterization: `L_eff = L²`, `R_eff = R²` (and optionally `D_eff = D²`) before computing reconstruction. Smooth alternative to softplus; no clipping needed.
- [x] **Level 3 — Loss function.** Carry L1 from Experiment 2. L1 on the *raw* `L,R` is fine because squaring preserves zeros (zero in → zero out).

## 3. Hypothesis

Components become parts-based: positive masks of stroke regions, no negative lobes. Top-k visualization should be readable as "this component is the bottom curve of a 6/8/0", "this is the top stem of a 4/7", etc. Fidelity drops more than in Exp. 2 or 3 (maybe cosine 0.85–0.90 vs. >0.95). The interpretability gain is the payoff.

## 4. Method

```python
fit_decomposition(model, alpha_l1=BEST_ALPHA, symmetric=False, nonneg=True)
```

Inside `fit_decomposition`, when `nonneg=True`: monkey-patch the `Sparse` reconstruction so that the effective factors are `L²`, `R²`. Critical: non-negativity must apply to the *reconstruction*, not just the penalty (per §10 of `METHOD_REFERENCE.md`).

## 5. Status

- [x] Designed
- [ ] Implemented in `experiments/main_experiments.ipynb`
- [ ] Run
- **Figure**: `figures/fig_nonneg_l1.png`

`Status: designed`

## 6. Results / Notes

To be filled. Specifically check: are `L+R` plots all-positive-pixels, do they look like single-stroke masks, and is `D` showing sharp class concentration?
