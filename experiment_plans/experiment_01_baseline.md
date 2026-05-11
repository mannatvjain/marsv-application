# Experiment 01: Baseline (pure CP, cosine loss)

## 1. Five evaluation targets

| Target | How this approach addresses it |
|--------|--------------------------------|
| **Short** — few components for high fidelity | Indirectly. Rank is fixed at 64; no rank-pruning prior. The empirical importance ranking from `decompose()` shows how many components carry meaningful weight. |
| **Interpretable** — each component readable as a discrete feature | Not addressed. No prior shapes components toward localization or partness. Components are expected to be dense, tangled mixtures. |
| **Fidelity** — cosine(B, B̂) + accuracy preserved | Directly optimized. `loss = 1 − cos(B, B̂)`. Expect cosine ≥ 0.95 and accuracy within ~1% of the trained MNIST classifier. |
| **Sharing** — features shared across classes encoded once | Addressed structurally by CP. Each component has a `D_r` vector spanning all 10 classes, so a stroke useful for {1,4,7} can be one component instead of three duplicates. |
| **Orthogonality** — within-class non-orthogonality allowed | Addressed structurally by CP. No constraint requires `L_r ⊥ L_{r'}`; components can overlap freely in input space. |

## 2. Intervention level(s)

- [x] **Level 1 — Decomposition family.** Switch from per-class eigendecomposition (paper baseline) to joint CP across all classes. This is the fundamental shift; everything else builds on it.
- [ ] **Level 2 — Parameterization.** Raw real-valued tensors (`nn.Parameter`).
- [ ] **Level 3 — Loss function.** Pure cosine loss, no regularizers.

## 3. Hypothesis

CP alone resolves the **sharing** and **orthogonality** failure modes by construction. But without any prior shaping the components, the optimizer will land on a minimum that reconstructs `B` accurately while producing dense, hard-to-name components. This is the "sharing-only" result — necessary as a control to show that the priors in Experiments 2–4 are doing real work, not just rediscovering CP's structural wins.

## 4. Method

```python
sparse = Sparse.from_config(rank=64).to(device)
optimizer = torch.optim.Muon(sparse.parameters(), lr=0.02, momentum=0.95)
scheduler = CosineAnnealingLR(optimizer, T_max=200)
for _ in range(200):
    loss = 1 - sparse.similarity(model)
    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
```

Single call: `fit_decomposition(model, alpha_l1=0.0, symmetric=False, nonneg=False)`. Seed fixed before fit.

## 5. Status

- [x] Designed
- [x] Implemented in `experiments/main_experiments.ipynb`
- [ ] Run
- **Figure**: `figures/fig_baseline.png`

`Status: implemented`

Note: cell exists and runs, but `kaleido` isn't installed in `marsv` env so `fig.write_image(...)` is currently no-op. Install `kaleido` (or screenshot the inline output to `figures/fig_baseline.png`) and bump status to `run`.

## 6. Results / Notes

To be filled after the figure is exported. Capture: cosine similarity, test accuracy, qualitative read of the top-8 components (`L+R`, `L−R`, `D`).
