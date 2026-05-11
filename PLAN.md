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
- [x] **Experiment 1: Baseline.** Pure CP, cosine loss, no priors. cos=1.0006, acc=0.9675. Plan: `experiment_plans/experiment_01_baseline.md`.
- [x] **Experiment 2: L1 sparsity sweep on `L, R`.** Run at `α ∈ {0.001, 0.01, 0.1}`; all preserve cos≈1 and acc≈0.967, but L1 barely bites (scale-invariance, METHOD_REFERENCE §7). Plan: `experiment_plans/experiment_02_l1_sweep.md`.
- [x] **Experiment 3: Symmetric CP + L1.** Tied `L = R` at α=0.1: cos=0.9944, acc=0.9657. Plan: `experiment_plans/experiment_03_symmetric_cp.md`.
- [x] **Experiment 4: Non-negativity + L1.** Squared parameterization at α=0.1: cos=0.9842, acc=0.9663. Plan: `experiment_plans/experiment_04_nonnegativity.md`.
- [x] **Experiment 5: Dictionary learning.** Degenerate (cos=0.02, recon term too small vs L1 mean term). Honest failure documented. Plan: `experiment_plans/experiment_05_dictionary_learning.md`.
- [x] Summary table: `canonical_results` DataFrame at end of notebook; exported to `canonical_results.csv`.
- [ ] Side-by-side comparison figure of the best result from each experiment (for the report).

## Phase 3: Report
- [ ] Write up motivation, method, results, conclusions (concise — mentors emphasized brevity)
- [ ] Include link to this repo
- [ ] Submit to MARS V Goodfire stream

## Known issues / follow-ups
- [x] `kaleido` is not installed in `marsv` env. **Resolved:** `pip install kaleido` done; figures now write to disk. All five plans now show `Status: run` and the maintenance check passes (5 run / 5 total).
- [ ] **Cell `imports` had `device = "cpu:0"` which breaks `torch.load(map_location="cpu:0")` in current torch.** Fixed to `device = "cpu"`. Worth a note in a `*_REFERENCE.md` to save the next session debugging an MNIST-classifier-at-8%-accuracy red herring.
- [ ] The cosine similarity reported for Exps 1–3 is slightly > 1 (e.g. 1.0006), exceeding the sanity-test's `1.0 + 1e-5` tolerance but only triggered there for random init. Probably float accumulation in the symmetrized inner product; worth confirming with a tighter test using a trained `sparse` before quoting these numbers in the report.
- [x] The exploratory cells (dict learning sweep / pushed / warm-start / low-rank / nonneg-low-rank / symmetric) — **resolved**: moved out of `main_experiments.ipynb`. The canonical notebook is now Exps 1→2 (×3 α cells) →3→4→5. The additional prior variants (smoothness, distinctness, group-lasso, asymmetric-L1, warm-start, low-rank, non-negative) live in `additional_priors.ipynb`, and Exp 5 is kept in the canonical set as the headline honest failure (cos=0.02).
