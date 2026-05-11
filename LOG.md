# Log

Reverse-chronological. Each session appends what changed, what's unfinished, what to pick up next.

### 2026-05-11 (cleanup pass — notebook reorder + history rewrite)
- Committed the dirty working state as a pre-cleanup snapshot, then rebuilt `main_experiments.ipynb` from scratch in canonical order: imports → config → MNIST → helpers → sanity → Exp 1 → Exp 2 (one cell per α: 0.001 / 0.01 / 0.1) → BEST_ALPHA picker → Exp 3 → Exp 4 → Exp 5 → summary. 25 cells total. Ran end-to-end against the cached MNIST classifier; all cells exit 0 and `canonical_results.csv` reproduces the documented numbers (baseline cos=1.0006, L1 sweep cos≈1.0006 with nonzero_frac 0.998/0.994/0.949, symmetric cos=0.9944, nonneg cos=0.9842, Exp 5 cos=0.02 honest failure).
- Deleted the six exploratory cells from `main_experiments.ipynb` (priors sweep, pushed asymmetric, warm-start, low-rank, nonneg-low-rank, exploratory symmetric). They were all variants of the Frobenius+L1 collapse and are preserved in `additional_priors.ipynb`. Stripped narrator/AI-style comments, tightened the `fit_decomposition` docstring (long prose cross-refs → none; kept one inline comment on the squared-param bake-back).
- **History rewrite**: removed the `Co-Authored-By: Claude` trailer from every commit on the branch (user pushed back: Claude is a tool, not an author). New hashes: `aa009ee` (unchanged), `bb6449d`, `2deb812`. Saved a global memory so future sessions don't re-add the trailer.
- Maintenance check still passes (5 run / 5 total).
- **Unfinished**: still no side-by-side comparison figure for the report, and the cosine>1 sanity-test tightening is open. No remote yet — user wants to push to `github.com/mannatvjain/marsv-application` (public) once they confirm.
- **Next session should**: confirm with the user, then `gh repo create mannatvjain/marsv-application --public --source=. --push`, then start the writeup.

### 2026-05-11 (later — canonical Exps 2–4 added & run)
- Read all five experiment plans, METHOD_REFERENCE, INDEX/PLAN/LOG. Existing notebook had Exp 1 baseline (working: cos≈1, acc=0.967) plus a stack of exploratory Frobenius+L1 dictionary-learning variants that all collapsed for the same reason (`recon.pow(2).mean()` produces a signal so small relative to per-element-mean L1 that the L1 term dominates from step 1).
- Fixed `fit_decomposition` so `nonneg=True` applies the squared parameterization to the *reconstruction*, not just to the L1 penalty (METHOD_REFERENCE §10). Previous version silently no-op'd the constraint — dormant bug since nonneg had never actually been run end-to-end before this session.
- Added canonical Exps 2 (L1 sweep over `ALPHAS_L1`), 3 (symmetric `L = R` at best-α), 4 (nonneg via `L_eff = L²` at best-α), and a `canonical_results` summary DataFrame cell at the bottom of `main_experiments.ipynb`. Removed four empty-cell cruft entries. Preserved the user's exploratory cells.
- Installed `kaleido` into the `marsv` env. All five figures now write to `figures/` on disk; maintenance check now reports 5 run / 5 total. Bumped each plan's Status to `run` and wrote real Results / Notes sections.
- **Caught a sharp footgun**: cell `imports` had `device = "cpu:0"`. PyTorch tensor placement accepts that string but `torch.load(map_location="cpu:0")` does not — the cached MNIST model fails to load, the cell raises, and (because nbconvert ran with `allow_errors=True`) every downstream cell silently runs against a random-init `model` with `orig_acc ≈ 0.08`. First nbconvert pass produced garbage results that looked plausible until I checked `orig_acc`. Fixed to `device = "cpu"`. Worth a `*_REFERENCE.md` note next session.
- Canonical results: baseline cos=1.0006 acc=0.9675 | L1 α=0.001 cos=1.0006 nz=0.998 | α=0.01 cos=1.0006 nz=0.994 | α=0.1 cos=1.0004 nz=0.949 | symmetric+L1 cos=0.9944 acc=0.9657 | nonneg+L1 cos=0.9842 acc=0.9663. Exp 5 still degenerate (cos=0.02) — left as honest failure.
- **Unfinished**: the side-by-side comparison figure for the report; tightening the cosine-sim sanity test to assert on a *trained* `sparse` (current test only checks random init); deciding whether to re-tune or remove the exploratory cells before submission.
- **Next session should**: build the comparison figure, then start the actual MARS V report writeup. The notebook now has everything needed.

### 2026-05-11 (same session — experiment_plans + figures + maintenance check)
- Authored the contents of `experiment_plans/` (template + four filled-in plans for Exps 1–4) and `scripts/check_experiments_figures.py` (the maintenance check). Created `figures/` (just `.gitkeep` for now) and `.claude/settings.json` (Stop hook running the check). Note: an earlier entry below misattributed this infra to "the user adding it in parallel" — it was authored in this session at the user's direction.
- Each experiment plan addresses the five evaluation targets (short, interpretable, fidelity, sharing, orthogonality) and tags which of the three intervention levels (decomposition family / parameterization / loss) it touches.
- Honest accounting: previously Exp. 1 was marked "run" in PLAN.md but the figure was never saved (kaleido isn't installed in `marsv`, so `fig.write_image` silently no-ops). Downgraded its plan to `Status: implemented`; check now passes (0 run / 4 plans).
- `METHOD_REFERENCE.md` §10 condensed to a pointer table — full per-experiment design now lives in `experiment_plans/`.
- `CLAUDE.md` "Before starting work" rewritten to require every session to read the three upstream notebooks (0-tutorial, 1-images, 0-exps), `main_experiments.ipynb`, plus `LOG.md`, `PLAN.md`, and `experiment_plans/`. Added a Conventions bullet documenting the experiment-plan rule and the Stop-hook maintenance check.
- **Unfinished**: same as below — Exps 2–4 not yet implemented or run; kaleido still not installed (unblocker for `Status: run` on anything).
- **Next session should**: install kaleido (`pip install kaleido` inside `marsv`), run Exp. 1 → save `figures/fig_baseline.png` → bump its plan to `Status: run`. Then implement Exps 2–4 per their plans, saving figures with the names already listed in each plan.

### 2026-05-11 (later session — naming cleanup)
- Short session. No code changes; clarified notebook naming in docs so future sessions don't confuse the four notebooks at play.
- Updated `INDEX.md`: added a "Notebook naming cheat-sheet" above the Code table mapping the user's shorthand ("0-tutorial", "1-images", "0-exps") to paths, and flagging `main_experiments.ipynb` as our primary notebook (vs. Thomas's exercise skeleton).
- Updated `CLAUDE.md`: added a "Notebook naming" subsection under Architecture covering the same mapping, including the convention that "Thomas's code" refers to `exercises/0_decomposition.ipynb`.
- User added new infra in parallel: `experiment_plans/` (one `.md` per experiment + `experiment_template.md`), `figures/` (PNG output dir), `scripts/check_experiments_figures.py` (status=run plans must have figures on disk), `.claude/settings.json` (Stop hook runs the check). CLAUDE.md was updated by the user to reflect this; I brought `INDEX.md` into alignment.
- **Unfinished**: unchanged from earlier today — Experiments 2–4 and the summary/comparison cells.
- **Next session should**: pick up Experiment 2 (L1 sweep over `ALPHAS_L1`) in `main_experiments.ipynb`.

### 2026-05-11
- Read the three required notebooks (`tutorials/0_introduction`, `tutorials/1_image`, `exercises/0_decomposition`); marked them done in `PLAN.md`.
- Wrote `METHOD_REFERENCE.md`: standalone theory + experimental-design doc covering the bilinear MLP, the paper's per-class eigendecomposition, the two failure modes (orthogonality within-class, sharing across-class), the CP tensor decomposition framework, the three intervention levels, the sparsity-with-cosine-loss caveat, the four experiments, and implementation gotchas. Math written in Unicode / fenced code blocks rather than LaTeX since Claude Code doesn't render LaTeX.
- Built notebook scaffold in `main_experiments.ipynb`: imports + renderer fix → config block → MNIST classifier training → parameterized `fit_decomposition` / `evaluate` / `visualize_decomposition` helpers → Experiment 1 (baseline) cell. Each remaining experiment is a one-line call to `fit_decomposition` with different flags.
- `tensor-decomposition.ipynb` is now just side scratch (the toy-model exploration); the real work has moved to `main_experiments.ipynb`. Updated `INDEX.md` and `CLAUDE.md` accordingly.
- Wrapped `fig.write_image(...)` in try/except — `kaleido` isn't installed in the `marsv` env, so the baseline cell previously crashed at the save step. Noted as a follow-up in `PLAN.md`.
- Initial git commit captures the scaffolded state.
- **Unfinished**: Experiments 2–4 (L1 sweep, symmetric CP, non-negativity) and the summary/comparison cells are not yet in the notebook.
- **Next session should**: add Experiment 2 (L1 sweep over `ALPHAS_L1`), then 3 (symmetric tie), then 4 (squared parameterization), then the summary DataFrame + comparison figure. After running, capture screenshots for the report.

### 2026-05-10
- Scaffolded project from `~/Developer/agent-scaffold` (CLAUDE/INDEX/PLAN/LOG).
- Pulled the MARS V applicant brief out of `~/Downloads/Decomposing Weights into Human Concepts.docx`, converted it to PDF (textutil docx→html, then Chrome headless html→pdf — `cupsfilter` on macOS has no html/rtf→pdf path), saved as `task-decomposing-weights.pdf`. Original docx removed.
- Cloned upstream [`tdooms/bilinear-decomposition`](https://github.com/tdooms/bilinear-decomposition) in-tree (not a submodule) so the notebook can import from `src/` directly.
- Recorded that we are applying to the **Goodfire stream only** in CLAUDE.md and PLAN.md.
- Created `environment.yml` and a conda env `marsv` (Python 3.12). Upstream uses `uv` + a `cu126` torch index for non-darwin; on this Mac we install plain `torch`/`torchvision` from conda-forge/PyPI and install the upstream package editable so its `src/` layout (image, language, sae, shared, toy) is importable.
- Filled out CLAUDE.md, INDEX.md, PLAN.md with project-specific content; deleted the scaffold's bootstrap "File Guide" section.
- Unfinished: notebooks unread; experiment not started.
- Next session should: `conda activate marsv`, then walk `tutorials/0_introduction.ipynb` and `tutorials/1_image.ipynb` before touching `tensor-decomposition.ipynb`.
