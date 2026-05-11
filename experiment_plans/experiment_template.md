# Experiment NN: <name>

> Template for experiment plans. Copy to `experiment_NN_<slug>.md`, fill in all sections, and update `Status` as work progresses. The maintenance check (`scripts/check_experiments_figures.py`) requires every experiment with `Status: run` to have its `Figure` file present on disk under `figures/`.

## 1. Five evaluation targets

How this experiment's decomposition approach addresses each of the five targets the project is judged on. Be specific — say *which mechanism* in the approach delivers the target, or write "no change vs. baseline" if it doesn't.

| Target | How this approach addresses it |
|--------|--------------------------------|
| **Short** — few components needed for high fidelity | |
| **Interpretable** — each component readable as a discrete feature (stroke, edge, region) | |
| **Fidelity** — cosine(B, B̂) and downstream classification accuracy preserved | |
| **Sharing** — features shared across classes are encoded once, not duplicated per-class | |
| **Orthogonality** — within-class components are *not* forced orthogonal; real overlapping features are allowed | |

## 2. Intervention level(s)

Mark which of the three levels (per `METHOD_REFERENCE.md` §6) this experiment touches. An experiment can hit more than one.

- [ ] **Level 1 — Decomposition family.** Mathematical form of the decomposition itself (CP, Tucker, BTD, symmetric CP, overcomplete dictionary, etc.).
- [ ] **Level 2 — Parameterization.** How `L, R, D` are represented (raw real, squared/non-neg, unit-norm columns, TopK on `D`, convolutional, symmetric tie, etc.).
- [ ] **Level 3 — Loss function.** Reconstruction loss + regularizers (cosine vs. Frobenius, L1 on `L,R` or `D`, Hoyer-square, spatial smoothness, distinctness, group lasso, etc.).

For each box checked, describe the specific change in 1–2 sentences.

## 3. Hypothesis

What we expect to see and *why*. Tie the prediction to the targets above.

## 4. Method

Concrete implementation: which fields of `fit_decomposition(...)` change, what new code (if any), what the loss looks like, training details (rank, steps, lr, seed).

## 5. Status

- [ ] Designed
- [ ] Implemented in `experiments/<notebook>.ipynb`
- [ ] Run
- **Figure**: `figures/<filename>.png`  *(must exist on disk if Status=Run; checked by `scripts/check_experiments_figures.py`)*

`Status:` one of `designed | implemented | run`. The maintenance check parses this line.

## 6. Results / Notes

Filled in after running. Quantitative metrics (cosine sim, accuracy, sparsity stats), qualitative observations (do components look interpretable?), surprises, follow-ups.
