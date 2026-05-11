# Method Reference — Decomposing Bilinear MLP Weights into Human Concepts

Standalone context for the experimental work in this repo. The companion files are `CLAUDE.md` (project setup, conventions, agent protocol), `PLAN.md` (task checklist), and `LOG.md` (session changelog). This file is the *theory + experimental design* reference — read it before touching `main_experiments.ipynb`.

Math is written in Unicode / fenced code blocks rather than LaTeX, since Claude Code does not render LaTeX. The final report can use LaTeX freely.

---

## 1. The task

Decompose the bilinear interaction tensor `B` of a trained bilinear MLP into components that are both **short** (few components needed for high fidelity) and **interpretable** (each component readable as a discrete feature).

The bilinear MLP architecture replaces the standard MLP with the form:

```
h   = (W_L x) ⊙ (W_R x)
y   = W_P h
```

where `W_L, W_R` are "up-projection" weights and `W_P` is a "down-projection" / unembedding. The output for class `c` is:

```
y_c = Σ_{i,j} B[c,i,j] · x_i · x_j  =  xᵀ B_c x
```

where `B` is the third-order **interaction tensor**:

```
B[c,i,j] = Σ_h W_P[c,h] · W_L[h,i] · W_R[h,j]
```

This tensor fully describes the layer's computation. For an MNIST classifier in embedding space, `B` has shape `(10, d, d)` where `d` is the embedding dimension.

## 2. The paper this builds on

Pearce, Dooms, Rigg, Oramas, Sharkey — "Bilinear MLPs enable weight-based mechanistic interpretability" (ICLR 2025).

Their method: for each output class `c`, take the slice `B_c`, symmetrize it (the antisymmetric part contributes zero to the bilinear form), and apply eigendecomposition:

```
B_c = Σ_i λ_i · v_i v_iᵀ
```

The output becomes:

```
y_c = Σ_i λ_i · (v_iᵀ x)²
```

Each eigenvector `v_i` is a "feature direction." Its activation is the squared projection of the input onto that direction. Positive `λ_i` pushes prediction toward class `c`; negative pushes away.

Key empirical findings:
- Eigenvalue spectra are sharply low-rank. ~10 eigenvectors per class are sufficient for >99% accuracy retention.
- Top eigenvectors are visually interpretable on MNIST — stroke detectors, edge detectors, sometimes prototype-like.
- Top eigenvectors are consistent across training runs (cosine ~0.8–0.9).

## 3. The failure modes being addressed

Per-class eigendecomposition has two limitations explicitly flagged by the paper:

**Orthogonality (within-class).** Eigenvectors of a symmetric matrix are orthogonal by the spectral theorem. Real visual features in input space are *not* orthogonal — a "vertical stroke" and a "left-side curve" can have positive dot product because they overlap in pixel space. Eigendecomposition forces these features into orthogonal combinations, producing "superposed" eigenvectors that look like rotated mixtures rather than clean feature detectors. Artifact negative regions appear in the eigenvectors to satisfy orthogonality, even when the underlying real features have no negative components.

**Sharing (across-class).** The eigendecomposition runs separately per class. Features shared across multiple classes (e.g., a vertical stroke useful for digits 1, 4, 7) get duplicated as similar-looking eigenvectors in three separate per-class bases. This inflates the apparent component count and hides the model's true economy.

The paper's discussion explicitly states: "Applying sparse dictionary learning approaches to decompose the bilinear tensor may be a promising way to relax the orthogonality constraint and find interpretable features from model weights." This is the open direction the task asks us to explore.

## 4. The tensor decomposition framework

Instead of decomposing each class slice `B_c` separately, we decompose the full tensor `B` at once into a CP (canonical polyadic) form:

```
B[c,i,j]  ≈  Σ_{r=1..R}  L[i,r] · R[j,r] · D[c,r]
```

Each component `r` has three vectors:
- `L_r` — left input pattern (length `d`)
- `R_r` — right input pattern (length `d`)
- `D_r` — output weights across classes (length `C`)

Identity used for visualization:

```
ab = ¼(a+b)² − ¼(a−b)²
```

So each neuron contributes a positive pattern `L_r + R_r` and a negative pattern `L_r − R_r`. Visualizing `L ± R` as 28×28 images shows the input templates each component responds to.

This formulation addresses both failure modes structurally:
- **Sharing** is solved because each component's `D_r` vector contributes across all classes simultaneously. One component can encode a feature used by multiple digits.
- **Orthogonality** is solved because no constraint requires `L_r ⊥ L_{r'}`. Components can overlap in input space.

What is *not* solved by this formulation alone: making individual components interpretable. Without additional structure, the optimizer finds a decomposition that reconstructs `B` but produces dense, tangled components. The work of the task is adding priors that shape the components toward interpretability.

## 5. The skeleton notebook (baseline approach)

The skeleton (`bilinear-decomposition/exercises/0_decomposition.ipynb`) provides:

A `Sparse` class (imported as `from image.sparse import Model as Sparse`) with learnable parameters `L, R, D`, parameterized as `nn.Parameter` tensors. The class exposes:
- `similarity(model)` — cosine similarity between the reconstructed tensor and the original model's `B`.
- `decompose()` — returns `L+R`, `L-R`, `D`, and a sigma importance ranking, sorted by component magnitude.
- A forward pass that uses `L, R, D` as a new bilinear layer for classification accuracy comparison.

The training loop:

```python
sparse = Sparse.from_config(rank=64).to(device)
optimizer = torch.optim.Muon(sparse.parameters(), lr=0.02, momentum=0.95)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

for _ in range(200):
    loss = 1 - sparse.similarity(model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

200 steps of Muon optimizer with cosine learning rate annealing. No regularization. No constraints on `L, R, D`. The loss is just `1 − cos(B, B̂)`.

This is the baseline. The empty exercises cell at the end is where modifications go.

## 6. The three levels of intervention

Modifications to the baseline can be made at three levels:

**Level 1: Decomposition family.** The mathematical form of the decomposition itself. CP is the skeleton's choice. Alternatives include Tucker (more flexible but harder to interpret), BTD (Block Term Decomposition — each component is rank-`L` instead of rank-1), tensor train, symmetric CP (tied `L = R`), overcomplete dictionary (`R > d`).

**Level 2: Parameterization.** How `L, R, D` are represented. The skeleton uses raw real-valued tensors. Alternatives include:
- Non-negativity via squared parameterization (`L_eff = L²`) or softplus (`L_eff = softplus(L)`). Forces components to be additive parts.
- Unit-norm columns. After each step, normalize each column of `L, R` to have norm 1. Prevents scale degeneracy.
- TopK on `D`. `D[c,:] = TopK_K(D̃[c,:])` enforces that each class touches at most `K` components.
- Convolutional structure. Parameterize `L_r` as a kernel placed at a position, for translation equivariance.
- Symmetric tie. `L = R`, halving parameters and respecting the actual symmetry of `B`.

**Level 3: Loss function.** Regularizers added to the reconstruction loss. The skeleton uses pure cosine similarity. Alternatives include:
- L1 sparsity on `L, R`:  `α · (‖L‖₁ + ‖R‖₁)`. Spatial localization.
- L1 sparsity on `D`:  `β · ‖D‖₁`. Class specialization.
- Hoyer-square:  `γ · Σ_r (‖L[:,r]‖₁ / ‖L[:,r]‖₂)²`. Scale-invariant sparsity, compatible with cosine loss.
- Spatial smoothness: penalize `Σ_{i,j} (L[i+1,j,r] − L[i,j,r])²` on reshaped `L_r`. Coherent regions.
- Distinctness:  `δ · Σ_{r≠r'} ⟨L_r, L_{r'}⟩²`. Soft orthogonality preventing duplicates.
- Group lasso on columns:  `λ · Σ_r √(‖L_r‖² + ‖R_r‖² + ‖D_r‖²)`. Whole-component pruning; automatic rank selection.
- Switching from cosine to Frobenius:  `‖B − B̂‖_F²`. Necessary to make standard L1 meaningful, since cosine is scale-invariant.

## 7. Sparsity caveat: scale invariance

A common pitfall: standard L1 regularization is degenerate under a cosine similarity loss. Cosine is scale-invariant — scaling `L, R, D` by any constant does not change the cosine. The L1 penalty pushes magnitudes to zero, but the cosine fit doesn't care, so the optimizer can drive all magnitudes to zero without affecting the fit term. In practice the gradients still work, but the regularization strength is not what it appears to be.

Two fixes:
1. Switch to Frobenius loss. Then magnitude matters, and L1 has meaningful pressure against it.
2. Use scale-invariant priors. Hoyer-square (`‖x‖₁² / ‖x‖₂²`) measures sparsity as a ratio and is scale-invariant. TopK is also scale-invariant. These compose with cosine cleanly.

For pragmatic experimentation L1 with cosine still produces visible sparsity effects, but the alpha values are not directly comparable to L1-with-Frobenius literature numbers.

## 8. Symmetry of B

`B` is symmetric in its two input modes (`B[c,i,j] = B[c,j,i]`) after symmetrization. An unconstrained CP decomposition wastes rank by potentially learning two copies of each component — one with `(L, R)` and one with `(R, L)`. Both reconstruct the same symmetric tensor, but neither alone is the "true" feature.

Tying `L = R` forces symmetric CP:  `B[c,i,j] ≈ Σ_r D[c,r] · V[i,r] · V[j,r]`. This:
- Halves the parameter count.
- Removes the `L/R` swap ambiguity.
- Recovers the per-class eigendecomposition as a special case if you also force `D` to be one-hot per class.

The asymmetric formulation in the skeleton is more general but less efficient on the symmetric tensor.

## 9. Evaluation criteria

Two quantitative anchors:
1. **Reconstruction fidelity.** Cosine similarity between `B` and reconstructed `B̂`. Higher is better.
2. **Behavioral preservation.** Classification accuracy of the decomposed model on MNIST test set. Should be close to the original model's accuracy. Large gaps indicate the decomposition is destroying useful structure.

Two qualitative criteria:
3. **Component locality and nameability.** Visual inspection of `L+R` and `L−R` patterns. Do they look like recognizable features (strokes, edges, regions)? Can a human name what each component detects?
4. **Class concentration.** Visual inspection of `D_r` bars. Does each component contribute to a small number of semantically related classes, or spread across many?

There is no single "interpretability score." The combination of these criteria is the basis for the comparison.

## 10. Experimental design

Four experiments, each modifying the baseline at one or more of the three levels. **The full plan for each lives in its own file under `experiment_plans/`** — see the template (`experiment_plans/experiment_template.md`) for the required structure (5 targets × 3 intervention levels). Summarized here:

| # | Plan | One-line summary | Levels touched |
|---|------|------------------|----------------|
| 1 | `experiment_plans/experiment_01_baseline.md` | Pure CP, cosine loss, no priors. Sharing-only result | L1 |
| 2 | `experiment_plans/experiment_02_l1_sweep.md` | L1 on `L,R`, sweep `α ∈ {0.001, 0.01, 0.1}` for spatial localization | L3 |
| 3 | `experiment_plans/experiment_03_symmetric_cp.md` | Tie `L = R` + best α from Exp. 2; remove L/R swap ambiguity | L1 + L2 + L3 |
| 4 | `experiment_plans/experiment_04_nonnegativity.md` | Squared parameterization `L_eff = L²` + L1; parts-based components | L2 + L3 |

When proposing a *new* experiment, copy `experiment_plans/experiment_template.md` to `experiment_plans/experiment_NN_<slug>.md` and fill it in *before* writing notebook cells. This is the design artifact the mentors will read — the notebook is just the execution.

Implementation notes (kept here because they cut across experiments):
- For symmetric CP (#3), cleanest is a single parameter matrix; hack-style is to copy `L → R` before each forward pass.
- For non-negativity (#4), the squaring must apply to the *reconstruction*, not just the penalty — modify the `Sparse` forward or monkey-patch the similarity computation.

## 11. Notebook structure

Cells in order (in `main_experiments.ipynb`):

1. Imports and device setup.
2. Configuration block: `RANK`, `N_STEPS`, `LR`, `SEED`, `K_VIS`, `ALPHAS_L1`. Single point of control.
3. Plotly renderer fix (one-line workaround for inline rendering).
4. Data loading and MNIST classifier training.
5. Helper functions: `fit_decomposition(model, **kwargs)`, `evaluate(sparse, model, test)`, `visualize_decomposition(sparse, title, save_path)`. Refactored so each experiment is a one-line call.
6. Storage list `results` for collecting metrics and figure paths.
7. Markdown + experiment 1 (baseline).
8. Markdown + experiment 2 (L1 sweep, looped over alphas).
9. Markdown + experiment 3 (symmetric CP).
10. Markdown + experiment 4 (non-negativity).
11. Summary cell: pandas DataFrame of all metrics, exported to CSV.
12. Optional: side-by-side comparison figure of the best result from each experiment.

## 12. Practices applied

- Single config block at the top. One place to change parameters.
- Parameterized fit function — methods are called via `fit_decomposition(..., alpha_l1=..., symmetric=..., nonneg=...)` rather than copy-pasted cells.
- Consistent result schema: each experiment appends `{"name": ..., "sparse": ..., "metrics": ...}` to a shared list.
- Markdown hypothesis cell before each experiment.
- Seeds set before each fit for reproducibility.
- Figures saved to disk with descriptive names (`fig_baseline.png`, `fig_l1_0.01.png`, etc.). Requires `kaleido` — if not installed, pass `save_path=None` and rely on inline display.
- Quantitative metrics printed in addition to figures.

Practices deliberately not applied (time-bounded scope):
- Caching to disk (each fit is ~30 seconds; not worth the infrastructure).
- W&B logging.
- jupytext / nbstripout (not needed for a one-off submission).
- Module extraction to `.py` files (helpers stay in the notebook).

## 13. What success looks like

A successful submission shows:
- Reconstruction fidelity (cosine similarity to `B`) preserved at >0.9 across all experiments.
- Classification accuracy preserved within a few percent of the baseline MNIST classifier.
- Visible qualitative difference between baseline components (dense, scattered, less interpretable) and prior-modified components (localized, nameable, class-specialized).
- Clean reporting of the trade-off curve: as priors get stronger, reconstruction degrades but interpretability improves. The interesting question is where the sweet spot is.

Per the original paper's discussion, the explicit research target is to produce a decomposition with:
- A single shared, non-orthogonal dictionary of ~30 atoms across all classes.
- Each class lighting up at most `K ≈ 5` atoms.
- Atoms readable as parts/strokes rather than orthogonal blends.

The skeleton's CP formulation is the starting point; the experiments here test whether sparsity priors and parameterization changes get closer to this target than the unconstrained baseline.

## 14. Implementation gotchas

- The `Sparse` class parameters are likely named `sparse.l`, `sparse.r`, `sparse.d`. If experiments fail with AttributeError, check `list(dict(sparse.named_parameters()).keys())` to confirm.
- Muon is a less common optimizer designed for matrix-shaped parameters. The skeleton uses it; we keep it. Substituting Adam may change convergence but should not alter the conclusions.
- `decompose()` returns components sorted and normalized by importance. The visualization shows the top `k=8`.
- `Sparse.from_config(rank=R)` initializes with random factors. The seed is set globally before each fit for reproducibility.
- The MNIST classifier itself is trained with Gaussian noise augmentation (`std=0.4`) per the tutorial 1 setup. Without this, the model overfits and the interaction tensor's eigenvectors look noisy (the same effect transfers to CP decompositions).
- All experiments use the same trained MNIST classifier. We do not re-train the underlying model between experiments — only re-fit the decomposition on top of the fixed `B` tensor.
- Plotly figure image export uses the `kaleido` engine. If `pip install kaleido` hasn't run, `fig.write_image(...)` raises `ValueError`. Pass `save_path=None` (or guard the call) to fall back to inline display.

## 15. Future directions

If extending beyond the time-budgeted scope:
- Sweep rank `R` at fixed prior strength to characterize the rank-fidelity Pareto frontier.
- Combine non-negativity, sparsity, and symmetric tying in one experiment.
- Implement Hoyer-square or group-lasso priors for scale-invariant sparsity.
- Try BTD (rank-`(L, L, 1)`) for translation-equivariant features.
- Implement convolutional CP for translation invariance.
- Compare against eigendecomposition baseline (per-class) on the same fidelity-vs-interpretability axes.
- Apply the same framework to a deeper bilinear network or to SAE feature dictionaries on a language model.
