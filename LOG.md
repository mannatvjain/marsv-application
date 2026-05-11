# Log

Reverse-chronological. Each session appends what changed, what's unfinished, what to pick up next.

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
