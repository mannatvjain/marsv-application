# Log

Reverse-chronological. Each session appends what changed, what's unfinished, what to pick up next.

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
