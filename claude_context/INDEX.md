# Index

Paths are repo-root-relative. This file lives in `claude_context/`.

## Documentation

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions, architecture, agent protocol (root-level — Claude Code entry point) |
| `claude_context/INDEX.md` | This file — map of the repository |
| `claude_context/PLAN.md` | Task checklist with progress tracking |
| `claude_context/LOG.md` | Reverse-chronological session changelog |
| `claude_context/METHOD_REFERENCE.md` | Theory + experimental design: bilinear MLPs, CP decomposition, the five experiments. Read before touching `experiments/main_experiments.ipynb` |
| `claude_context/task-decomposing-weights.pdf` | The MARS V applicant prompt (Dooms & Gauderis) |

## Code

**Notebook naming cheat-sheet — internalize before working:**
- **`experiments/main_experiments.ipynb`** — *our* primary notebook. Canonical Exps 1–5.
- **`experiments/additional_priors.ipynb`** — additional prior variants beyond Exps 1–5 (smoothness, distinctness, group-lasso, asymmetric-L1, warm-start, low-rank, non-negative).
- **`bilinear-decomposition/tutorials/0_introduction.ipynb`** — referred to as **"0-tutorial"**. The first tutorial: bilinear MLP basics + eigendecomposition motivation.
- **`bilinear-decomposition/tutorials/1_image.ipynb`** — referred to as **"1-images"** (the *other* tutorial). MNIST setup + decomposition workflow on images.
- **`bilinear-decomposition/exercises/0_decomposition.ipynb`** — referred to as **"0-exps"**. Thomas Dooms's exercise skeleton — the scaffolding the prompt asks us to extend. (Thomas Dooms is one of our two mentors.)

| File | Purpose |
|------|---------|
| `experiments/main_experiments.ipynb` | **Primary notebook.** MNIST classifier + Exp 1–5 (baseline, L1 sweep, symmetric CP, non-negativity, dictionary learning) |
| `experiments/additional_priors.ipynb` | Additional decomposition experiments beyond the canonical Exps 1–5: spatial-smoothness, distinctness, group-lasso, asymmetric-L1 (atoms vs. code), warm-start from the dictionary-learning init, and low-rank / non-negative variants. |
| `experiments/tensor-decomposition.ipynb` | Side scratch: tiny `toy.Model` exploration of the symmetric interaction tensor |
| `environment.yml` | Conda env spec (`marsv`) for running the notebooks |
| `experiment_plans/` | One `.md` per experiment + `experiment_template.md`. **All future experiment plans go here.** Template forces 5 targets (short, interpretable, fidelity, sharing, orthogonality) + 3 intervention levels (decomposition family, parameterization, loss) |
| `experiment_plans/experiment_01_baseline.md` | Pure CP, cosine loss, no priors — sharing-only result |
| `experiment_plans/experiment_02_l1_sweep.md` | L1 on `L,R`, sweep α ∈ {0.001, 0.01, 0.1} |
| `experiment_plans/experiment_03_symmetric_cp.md` | Symmetric CP (`L=R`) + best α |
| `experiment_plans/experiment_04_nonnegativity.md` | Squared parameterization (`L_eff=L²`) + L1 |
| `experiment_plans/experiment_05_dictionary_learning.md` | Overcomplete CP + unit-norm columns + Frobenius loss + L1 on `L,R,D` |
| `figures/` | Exported plots (`fig_<experiment>.png`). Referenced from `experiment_plans/`. |
| `canonical_results.csv` | Summary table written by `experiments/main_experiments.ipynb` (one row per canonical experiment). |
| `scripts/check_experiments_figures.py` | Maintenance check: every plan with `Status=run` must have its figure on disk. Wired into the Stop hook. |
| `.claude/settings.json` | Claude Code settings — Stop hook runs the maintenance check at end of every session. |
| `bilinear-decomposition/` | Upstream repo (read-only). Tutorials, exercise skeleton, and `src/` modules imported by our notebooks |
| `bilinear-decomposition/tutorials/0_introduction.ipynb` | "0-tutorial" — required reading per the prompt |
| `bilinear-decomposition/tutorials/1_image.ipynb` | "1-images" — the other required tutorial |
| `bilinear-decomposition/exercises/0_decomposition.ipynb` | "0-exps" — Thomas's exercise skeleton, the thing we're extending |

## Data

| Path | Purpose |
|------|---------|
| `data/MNIST/` | MNIST blob downloaded by `MNIST(train=...)` — gitignored |
| `data/mnist_bilinear.pt` | Cached MNIST classifier weights — shared between notebooks, gitignored |
