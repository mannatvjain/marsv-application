# Plan

Applying to the **Goodfire stream** of MARS V only. Mentors: Thomas Dooms & Ward Gauderis.

See `METHOD_REFERENCE.md` for the full theory and experimental design. This file is just the punch list.

## Phase 1: Setup (~1h)
- [x] Clone `bilinear-decomposition` upstream repo
- [x] Convert task brief from docx to PDF
- [x] Scaffold project docs (CLAUDE/INDEX/PLAN/LOG)
- [x] Create conda env `marsv` from `environment.yml` and install upstream as editable
- [x] Read `tutorials/0_introduction.ipynb` — bilinear MLP basics, eigendecomposition motivation
- [x] Read `tutorials/1_image.ipynb` — MNIST setup + decomposition workflow
- [x] Read `exercises/0_decomposition.ipynb` — understand the skeleton we're extending

## Phase 2: Experiment (1–2h)

Four-experiment design captured in `experiment_plans/` (one `.md` per experiment, copied from `experiment_plans/experiment_template.md`). Each plan addresses the five targets (short, interpretable, fidelity, sharing, orthogonality) and the three intervention levels (decomposition family / parameterization / loss). See `METHOD_REFERENCE.md` for the underlying theory.

- [x] Write up the theory + experimental design (`METHOD_REFERENCE.md` + per-experiment plans in `experiment_plans/`)
- [x] Build the notebook scaffold: imports, config, MNIST classifier, parameterized `fit_decomposition` helper, `evaluate`, `visualize_decomposition`
- [x] **Experiment 1: Baseline.** Pure CP, cosine loss, no priors. *Implemented; figure not yet saved (kaleido missing).* Plan: `experiment_plans/experiment_01_baseline.md`.
- [ ] **Experiment 2: L1 sparsity sweep on `L, R`.** `α ∈ {0.001, 0.01, 0.1}`. Plan: `experiment_plans/experiment_02_l1_sweep.md`.
- [ ] **Experiment 3: Symmetric CP + L1.** Tie `L = R`, combine with best `α` from Exp. 2. Plan: `experiment_plans/experiment_03_symmetric_cp.md`.
- [ ] **Experiment 4: Non-negativity + L1.** Squared parameterization `L_eff = L²`, additive parts representation. Plan: `experiment_plans/experiment_04_nonnegativity.md`.
- [ ] Summary table: pandas DataFrame of metrics across experiments; export CSV.
- [ ] Side-by-side comparison figure of the best result from each experiment.
- [ ] Capture screenshots → save into `figures/` with the names listed in each plan, then bump `Status: run` in the plan and re-run `scripts/check_experiments_figures.py`.

## Phase 3: Report
- [ ] Write up motivation, method, results, conclusions (concise — mentors emphasized brevity)
- [ ] Include link to this repo
- [ ] Submit to MARS V Goodfire stream

## Known issues / follow-ups
- [ ] `kaleido` is not installed in `marsv` env; `fig.write_image(...)` is wrapped in try/except as a workaround. Install `kaleido` if PNG export is needed for the report — without it, no `figures/fig_*.png` get written and `scripts/check_experiments_figures.py` will block any plan from being marked `Status: run`.
