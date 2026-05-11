# Experiment 03: Symmetric CP (`L = R`) + L1

## 1. Five evaluation targets

| Target | How this approach addresses it |
|--------|--------------------------------|
| **Short** — few components for high fidelity | **Improved.** The asymmetric CP wastes rank by potentially learning two copies of each true component (one with (L,R), one with (R,L) swap). Tying `L = R` collapses these duplicates, so the same fidelity should be reachable with effectively half the rank. |
| **Interpretable** — each component readable as a discrete feature | **Improved.** Removing the `L/R` swap ambiguity means each component is unambiguously *one* feature, not "feature plus its swap." Combined with L1, components should be cleaner. |
| **Fidelity** — cosine(B, B̂) + accuracy preserved | Should match Experiment 2 at the chosen α — `B` is symmetric in (i,j) after symmetrization (§8 of `METHOD_REFERENCE.md`), so symmetric CP is the right family. Asymmetric CP can't exploit anything symmetric CP can't. |
| **Sharing** — features shared across classes encoded once | Inherited from CP. Symmetry tie is orthogonal to cross-class sharing. |
| **Orthogonality** — within-class non-orthogonality allowed | Inherited from CP. L1 still allows non-orthogonal sparse components. |

## 2. Intervention level(s)

- [x] **Level 1 — Decomposition family.** Switch from asymmetric CP to symmetric CP: `B[c,i,j] ≈ Σ_r D[c,r] · V[i,r] · V[j,r]` with a single factor matrix `V` instead of separate `L, R`.
- [x] **Level 2 — Parameterization.** Implementation: copy `L` into `R` before each forward pass (hack-style tie), or use a single `nn.Parameter` for `V`. Halves parameter count.
- [x] **Level 3 — Loss function.** Carry the best α from Experiment 2.

## 3. Hypothesis

At the same α, symmetric CP matches asymmetric CP on fidelity but produces visibly cleaner components (no L/R swap-pair duplicates in the top-k). Effective rank — measured by importance-weighted component count — should drop noticeably.

## 4. Method

```python
fit_decomposition(model, alpha_l1=BEST_ALPHA, symmetric=True, nonneg=False)
```

Inside `fit_decomposition`, when `symmetric=True`: either init a single `V` parameter or hard-copy `L → R` at the start of each forward. Reconstruction loss + L1 unchanged.

## 5. Status

- [x] Designed
- [ ] Implemented in `experiments/main_experiments.ipynb`
- [ ] Run
- **Figure**: `figures/fig_symmetric_l1.png`

`Status: designed`

## 6. Results / Notes

Compare directly against Exp. 2's best-α run. Look for: same cosine, same accuracy, fewer duplicate-looking components in the top-8 visualization.
