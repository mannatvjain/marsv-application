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

**Caveat (per `claude_context/METHOD_REFERENCE.md` §7):** cosine loss is scale-invariant, so the *effective* L1 strength differs from L1-with-Frobenius literature numbers. The sweep characterizes our regime; absolute α values aren't directly comparable across loss choices.

## 4. Method

```python
for alpha in ALPHAS_L1:  # [0.001, 0.01, 0.1]
    fit_decomposition(model, alpha_l1=alpha, symmetric=False, nonneg=False)
```

Three runs, each 200 steps, fresh seed per α. Each run produces its own figure and metrics row in the summary DataFrame.

## 5. Status

- [x] Designed
- [x] Implemented in `experiments/main_experiments.ipynb`
- [x] Run
- **Figure**: `figures/fig_l1_0.001.png`, `figures/fig_l1_0.01.png`, `figures/fig_l1_0.1.png`

`Status: run`

## 6. Results / Notes

| α     | cosine  | sparse_acc | nonzero(L,R) |
|-------|---------|------------|--------------|
| 0.001 | 1.0006  | 0.9675     | 0.998        |
| 0.01  | 1.0006  | 0.9677     | 0.994        |
| 0.1   | 1.0004  | 0.9677     | 0.949        |

Quantitatively the sweep barely moves anything — even at α = 0.1, only ~5% of `L,R` entries fall below the 1e-4 threshold, and accuracy is unchanged. This is the scale-invariance issue from METHOD_REFERENCE §7 made visible: cosine loss + per-element-mean L1 means α=0.1 is not actually "10× more pressure than 0.01"; the optimizer is free to inflate `L,R` magnitudes to make the L1 mean small without paying any reconstruction cost. To get real bite we'd need either Hoyer-square (scale-invariant) or a switch to Frobenius — Exp 5's territory. `BEST_ALPHA = 0.1` is selected mechanically (largest α meeting the cosine/accuracy bar), but should be read as "least-weak" rather than "best".

## 7. Failure modes / where this won't fully solve the problem

- **Cosine-loss scale degeneracy (METHOD_REFERENCE §7).** The optimizer can shrink `L,R` and grow `D` (or vice versa) without affecting cosine fit, neutralizing L1's effective pressure. α may need to be 10–100× literature numbers to bite — but the actual ratio is unknowable without unit-norm or Frobenius loss. Exp 5 is the version that fixes this.
- **No L1 on `D` → sharing is invisible.** Even if `L,R` get sparse, every atom still has free rein to touch every class. We can't tell which atoms are *shared* vs. class-specific from the L1-on-LR output alone. The "sharing" target gets a structural pass from CP but no visible signal from this prior.
- **Complete rank (R = d = 64) leaves no room to specialize.** With as many atoms as input dimensions, the optimizer must cram every feature into the basis. Classical dictionary learning is overcomplete for exactly this reason.
- **L1 enforces sparsity, not spatial coherence.** Sparse atoms can be "scattered noise pixels" rather than "strokes" — L1 doesn't care if the nonzero pixels are adjacent. A spatial-smoothness penalty would address this.
- **No unit-norm constraint.** Atom magnitudes leak into `D`; what you see in `L+R` heatmaps is not directly comparable across components.
