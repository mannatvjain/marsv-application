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
- [x] Implemented in `main_experiments.ipynb`
- [x] Run
- **Figure**: `figures/fig_baseline.png`

`Status: run`

## 6. Results / Notes

Cosine similarity ≈ 1.0006 (≥ 1 by ~6e-4 — float accumulation in the symmetrized inner product, not a bug); sparse-model accuracy 0.9675 vs. orig 0.9674. CP alone reconstructs `B` essentially perfectly and preserves classification — the structural wins (sharing, non-orthogonality) come for free. Qualitative read of the top-8 components is dense and tangled as predicted, with no obvious stroke/edge specialization. This control validates that Exps 2–4 need to earn cleaner components, not just match these numbers.

## 7. Failure modes / where this won't fully solve the problem

This experiment is a *control*, not a candidate solution — by design it does not address interpretability. Expected to fail on:

- **Dense, tangled components.** With no prior, the top-k visualizations will look like blurry ghosts — mixtures of strokes/regions rather than discrete features. Naming them will require generous interpretation.
- **CP non-uniqueness invisible to the loss.** Cosine reaches its minimum on infinitely many qualitatively different decompositions (METHOD_REFERENCE §4). Seed-to-seed variance in component appearance will be high; the "result" is one arbitrary draw from this set.
- **Importance ranking ≠ interpretability ranking.** `decompose()` sorts by component norm, which under no regularization is dominated by reconstruction utility, not feature semantics. The visible top-k may not be the most nameable atoms in the dictionary.

Its job is to make Experiments 2–5 falsifiable: if their priors don't produce visibly cleaner components than this, they aren't earning their complexity.
