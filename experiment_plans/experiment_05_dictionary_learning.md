# Experiment 05: Honest dictionary learning

> The L1 sweep in Exp 02 is "dictionary learning" only in spirit — it touches the loss but leaves the parameterization complete-rank, lets atom magnitudes drift under cosine, and never enforces sparse *usage* per class. This experiment runs the full classical setup: overcomplete dictionary, unit-norm atoms, L1 on both atoms (`L,R`) and code (`D`), Frobenius reconstruction. The framing Thomas/Ward flagged as obvious.

## 1. Five evaluation targets

| Target | How this approach addresses it |
|--------|--------------------------------|
| **Short** — few components for high fidelity | **Primary target via `D` sparsity.** Overcompleteness (R > d) gives the optimizer more atoms than it needs; L1 on `D` then prunes which ones each class actually uses. Effective short-ness emerges from sparse usage, not from a small rank ceiling. |
| **Interpretable** — each component readable as a discrete feature | **Primary target via `L,R` sparsity + unit-norm.** Olshausen-Field setup: sparse, normalized atoms over natural-ish image patches yield localized stroke/edge templates. |
| **Fidelity** — cosine(B, B̂) + accuracy preserved | Frobenius loss makes magnitude pressure real, so the α values bite where they should (cf. §7 of `METHOD_REFERENCE.md`). Cosine is still *reported* for cross-experiment comparison. |
| **Sharing** — features shared across classes encoded once | Inherited from CP, made *visible* by the sparse `D`: a stroke shared between 1/4/7 should now show as one atom with three lit-up `D` entries, instead of three near-duplicate atoms. |
| **Orthogonality** — within-class non-orthogonality allowed | Same as CP — no orthogonality constraint, and unit-norm columns don't induce one (norm-1 vectors are generally not orthogonal). |

## 2. Intervention level(s)

- [x] **Level 1 — Decomposition family.** Overcomplete CP: `R = 128` vs. embed dim `d = 64` (2× overcomplete). Atoms no longer span a basis; the dictionary can carry redundant explanatory candidates and let sparsity pick.
- [x] **Level 2 — Parameterization.** Unit-norm columns on `L, R` — after each optimizer step, renormalize `L[:,r] /= ‖L[:,r]‖` and same for `R`. Removes scale-degeneracy and makes L1 magnitudes commensurable across atoms.
- [x] **Level 3 — Loss function.** Frobenius reconstruction (`‖B − B̂‖_F²`) instead of cosine, plus L1 on both `L,R` *and* `D`:
  ```
  loss = ‖B − B̂‖_F² + α_atom · (‖L‖₁ + ‖R‖₁) + α_code · ‖D‖₁
  ```

## 3. Hypothesis

Versus Exp 02 (L1 on `L,R` only, cosine loss, complete rank):
- More atoms will end up genuinely *unused* (their `D` column collapses to ≈0), and the *used* atoms will be more localized — because the optimizer can throw away atoms instead of compressing all features into 64 slots.
- Per-class `D` rows will be sparse: each digit lights up a small handful of atoms, not all 128. This is the "sparse usage" signal Exp 02 cannot produce.
- Fidelity (cosine on `B`) will drop slightly from baseline (~0.95 → ~0.90 range) but accuracy should hold within ~2%.

The interesting failure mode: if Frobenius + unit-norm makes the optimization stiff (Muon was tuned for matrix-shaped raw params), we may see slower convergence or oscillation. If that happens, swap to Adam.

## 4. Method

New helper `fit_dictionary_learning(model, oc_rank=128, alpha_atom=0.01, alpha_code=0.01, ...)` because the existing `fit_decomposition` is cosine-loss-only and has no column-normalization step.

```python
B_target = model.b  # (C, d, d) ground-truth interaction tensor
sparse = Sparse.from_config(rank=128).to(device)
opt = torch.optim.Adam(sparse.parameters(), lr=1e-2)
for step in range(400):
    B_hat = torch.einsum('ir,jr,cr->cij', sparse.l, sparse.r, sparse.d)
    recon = (B_target - B_hat).pow(2).mean()
    reg   = alpha_atom * (sparse.l.abs().mean() + sparse.r.abs().mean())
    reg  += alpha_code * sparse.d.abs().mean()
    loss = recon + reg
    loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        sparse.l.data /= sparse.l.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        sparse.r.data /= sparse.r.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
```

`N_STEPS = 400` (Frobenius converges slower than cosine in our regime), `α_atom = α_code = 0.01` as a starting point. One run; if results are degenerate, sweep α next.

The figure shows the top-K atoms sorted by `‖D[:,r]‖` (importance by usage, not by raw atom norm — since atoms are unit-norm).

## 5. Status

- [x] Designed
- [x] Implemented in `main_experiments.ipynb`
- [x] Run (degenerate — see Results)
- **Figure**: `figures/fig_dictionary_learning.png`

`Status: run`

## 6. Results / Notes

Run is degenerate: cosine 0.020, sparse_acc 0.218, 76% of atoms unused, mean 2.3 atoms/class. The problem is the failure mode flagged in §7 of this plan ("Frobenius + Adam may converge to a worse fidelity") taken to its limit: `B_target` has mean abs ≈ 9e-4 and max ≈ 3e-2, so `(B − B̂)²` mean-reduced has a tiny scale relative to `α · ‖L,R‖₁/N`. The L1 term dominates from the first step; the optimizer minimizes it by shrinking everything to zero, and the reconstruction never recovers. Quick fix would be: (a) switch `recon` from `.pow(2).mean()` to `.pow(2).sum()` so the recon term scales with tensor size, (b) lower α by 2–3 orders of magnitude, or (c) normalize `B_target` to unit Frobenius before computing the loss. The downstream cells in the notebook (`sweep over priors`, `pushed`, `warm-start`, etc.) are variations on the same bug; they collapse for the same reason. Filed under known-failure rather than re-tuning — the canonical Exps 1–4 already cover the design space and the L1-on-cosine ceiling (§7 of METHOD_REFERENCE) is the more honest story to tell in the report than a re-tuned dictionary-learning success.

## 7. Failure modes / where this won't fully solve the problem

- **L1 doesn't enforce spatial coherence.** Atoms could converge to sparse-but-scattered pixel patterns rather than coherent strokes. Sparsity is necessary for interpretability, not sufficient. A total-variation / smoothness penalty would address this — out of scope here.
- **CP non-uniqueness still applies.** Even with overcompleteness, unit-norm, and L1 on both factors, different seeds may produce qualitatively different dictionaries at the same loss. Single-seed conclusions are weak; honest reporting needs a seed sweep.
- **Sparse usage ≠ semantic class concentration.** L1 on `D` makes `D[c,:]` sparse, but the few atoms used by class `c` may still be visually uninterpretable. We get short codes, not necessarily *meaningful* codes.
- **Unit-norm hard projection is non-smooth.** Renormalizing after every Adam step is a hard constraint applied outside the optimizer's view; can cause oscillation near the manifold. If we see loss bouncing, this is the likely cause. A Riemannian optimizer would be cleaner.
- **Two-axis α tuning unexplored.** `α_atom` and `α_code` are set equal — no reason to think this is the right ratio. A proper 2D sweep is needed before strong claims.
- **Frobenius + Adam may converge to a worse fidelity than cosine + Muon.** Different loss geometry, different minima. Reported cosine for cross-experiment comparison may look worse than Exp 2 even when atoms are visibly cleaner — careful framing required.
- **The bilinear tensor may be lower-rank than the dictionary.** If `B` truly is ~30 atoms (per the paper's per-class eigenvalue findings), 128 atoms means ~100 dead — fine, but we only learn this post-hoc.
