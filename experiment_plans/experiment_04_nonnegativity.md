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
- [x] Implemented in `main_experiments.ipynb`
- [x] Run
- **Figure**: `figures/fig_nonneg_l1.png`

`Status: run`

## 6. Results / Notes

At α = 0.1 (carried from Exp 2): cosine 0.9842, sparse_acc 0.9663. Fidelity drops further than Exp 3 — 1.0004 → 0.9842 — exactly as predicted, since hard non-negativity removes the model's ability to express contrast features via cancellation. Accuracy holds within 0.2pp of asymmetric. Implementation note (caught and fixed mid-task): the squared parameterization must apply to the *reconstruction* used for the loss, not just to the L1 penalty term — an earlier helper version squared only the penalty and silently no-op'd the constraint. Now: `L_eff = L²`, `R_eff = R²` go through a manual einsum + symmetrize, and the squared values are baked back into `sparse.left/right` post-fit so `decompose()` / `evaluate()` see the actually-positive atoms. The `L − R` row of the visualization is meaningless here (both factors positive) and should be dropped for the report.

## 7. Failure modes / where this won't fully solve the problem

- **Bilinear MLPs may genuinely need negative interactions.** `B` was learned without a non-negativity constraint; nothing about its training suggests an additive-parts decomposition exists. Edge detectors are inherently contrast features ("bright minus dark") and cannot be expressed as `L² · R²`. Squaring imposes an NMF-style ontology on a model that wasn't trained to obey it — fidelity may drop sharply below METHOD_REFERENCE §9's cosine > 0.9 success threshold.
- **The `L−R` negative-pattern column loses its meaning.** With `L_eff = L²`, `R_eff = R²`, the visualization `L−R` no longer represents what it did in Exp 1–3. The figure either needs a relabeling or that column should be dropped.
- **If `D` is also non-negative, "push away from class c" is gone.** The paper notes negative eigenvalues are interpretable (a feature *inhibits* a class). Forcing `D ≥ 0` removes that channel; we'd be measuring only excitatory features.
- **No `D` sparsity → sharing still invisible.** Same blind spot as Exp 2.
- **Squared parameterization has zero-derivative basins.** Once a column hits `L_r ≈ 0`, gradients vanish and it can't recover. Expect dead atoms early; effective rank may collapse below the nominal `R`.
