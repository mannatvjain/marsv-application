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
- [x] Implemented in `main_experiments.ipynb`
- [x] Run
- **Figure**: `figures/fig_symmetric_l1.png`

`Status: run`

## 6. Results / Notes

At α = 0.1 (carried from Exp 2): cosine 0.9944, sparse_acc 0.9657. Fidelity drops 6e-4 vs. asymmetric (1.0004 → 0.9944) and accuracy drops 0.2pp (0.9677 → 0.9657) — symmetric CP is, as expected, the right family for `B` and pays an essentially-free cost. Qualitatively the `L+R` row in the top-8 visualization shows no `(L, R) ↔ (R, L)` swap-pair duplicates, which is the whole point. The `L − R` row collapses to numerical noise (`L = R` by construction) and is uninformative — should be dropped when this figure goes into the report.

## 7. Failure modes / where this won't fully solve the problem

- **`L = R` collapses the L±R duality.** The identity `ab = ¼(a+b)² − ¼(a−b)²` is what gives each component a *positive pattern* (`L+R`) and a *negative pattern* (`L−R`). With `L = R`, `L−R = 0` — the negative pattern vanishes. Every component becomes a single positive template. If real bilinear features have a "this MINUS that" structure (contrast detectors), we lose the ability to represent them as one component.
- **Restricts expressivity if features aren't self-symmetric.** Symmetric CP can only represent components where left and right input patterns are identical. Asymmetric features of the form "pattern A interacts with pattern B ≠ A" get forced into sums of self-symmetric pieces — possibly *inflating* effective rank rather than reducing it.
- **Inherits Exp 2's L1-on-cosine degeneracy.** This experiment stacks Level-1/2 changes on top of the Level-3 problem from Exp 2; it doesn't fix the scale-invariance issue.
- **Removes swap ambiguity but not CP non-uniqueness.** Symmetric CP decompositions are still non-unique (sign and rotation freedom). Seed variance remains.
