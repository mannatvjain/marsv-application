# Index

## Documentation
| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions, architecture, agent protocol |
| `INDEX.md` | This file — map of the repository |
| `PLAN.md` | Task checklist with progress tracking |
| `LOG.md` | Reverse-chronological session changelog |
| `METHOD_REFERENCE.md` | Theory + experimental design: bilinear MLPs, CP decomposition, the four experiments. Read before touching `main_experiments.ipynb` |
| `task-decomposing-weights.pdf` | The MARS V applicant prompt (Dooms & Gauderis) |

## Code

**Notebook naming cheat-sheet — internalize before working:**
- **`main_experiments.ipynb`** (repo root) — *our* primary notebook, where all real work happens. The four CP decomposition experiments live here.
- **`bilinear-decomposition/tutorials/0_introduction.ipynb`** — referred to as **"0-tutorial"**. The first tutorial: bilinear MLP basics + eigendecomposition motivation.
- **`bilinear-decomposition/tutorials/1_image.ipynb`** — referred to as **"1-images"** (the *other* tutorial). MNIST setup + decomposition workflow on images.
- **`bilinear-decomposition/exercises/0_decomposition.ipynb`** — referred to as **"0-exps"**. Thomas Dooms's exercise skeleton — the scaffolding the prompt asks us to extend. (Thomas Dooms is one of our two mentors.)

| File | Purpose |
|------|---------|
| `main_experiments.ipynb` | **Primary notebook.** MNIST classifier + the four CP decomposition experiments |
| `tensor-decomposition.ipynb` | Side scratch: tiny `toy.Model` exploration of the symmetric interaction tensor |
| `environment.yml` | Conda env spec (`marsv`) for running the notebook |
| `experiment_plans/` | One `.md` per experiment + `experiment_template.md`. **All future experiment plans go here.** Template forces 5 targets (short, interpretable, fidelity, sharing, orthogonality) + 3 intervention levels (decomposition family, parameterization, loss) |
| `experiment_plans/experiment_01_baseline.md` | Pure CP, cosine loss, no priors — sharing-only result |
| `experiment_plans/experiment_02_l1_sweep.md` | L1 on `L,R`, sweep α ∈ {0.001, 0.01, 0.1} |
| `experiment_plans/experiment_03_symmetric_cp.md` | Symmetric CP (`L=R`) + best α |
| `experiment_plans/experiment_04_nonnegativity.md` | Squared parameterization (`L_eff=L²`) + L1 |
| `figures/` | Exported plots (`fig_<experiment>.png`). Referenced from `experiment_plans/`. |
| `scripts/check_experiments_figures.py` | Maintenance check: every plan with `Status=run` must have its figure on disk. Wired into the Stop hook. |
| `.claude/settings.json` | Claude Code settings — Stop hook runs the maintenance check at end of every session. |
| `bilinear-decomposition/` | Upstream repo (read-only). Tutorials, exercise skeleton, and `src/` modules imported by our notebook |
| `bilinear-decomposition/tutorials/0_introduction.ipynb` | "0-tutorial" — required reading per the prompt |
| `bilinear-decomposition/tutorials/1_image.ipynb` | "1-images" — the other required tutorial |
| `bilinear-decomposition/exercises/0_decomposition.ipynb` | "0-exps" — Thomas's exercise skeleton, the thing we're extending |

## Data
| Path | Purpose |
|------|---------|
| `data/MNIST/` | MNIST blob downloaded by `MNIST(train=...)` — gitignored |
