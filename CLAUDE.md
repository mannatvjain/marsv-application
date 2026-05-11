# MARS V Applicant Task — Decomposing Weights into Human Concepts

Application task for the MARS V (Cambridge AI Safety Hub) program — **Goodfire stream only**. Mentors: Thomas Dooms & Ward Gauderis. Deliverable is a concise report with screenshots + thoughts/conclusions; code link encouraged but mentors likely won't read it.

## Stack
- **Language**: Python 3.12 (Jupyter notebooks)
- **Env manager**: conda (env name `marsv`, see `environment.yml`)
- **Key dependencies**: PyTorch, torchvision, einops, transformers, jaxtyping, kornia, plotly, nnsight, datasets, wandb, accelerate, bidict, ipykernel
- **Reference repo**: [tdooms/bilinear-decomposition](https://github.com/tdooms/bilinear-decomposition) — cloned in-tree, installed editable into the conda env so `from image import Model, MNIST` etc. resolve to its `src/`

## Architecture

```
marsv-application/
├── task-decomposing-weights.pdf        # the prompt (converted from MARS docx)
├── main_experiments.ipynb              # primary notebook: MNIST + the four CP decomposition experiments
├── tensor-decomposition.ipynb          # side scratch: toy.Model exploration of the symmetric interaction tensor
├── METHOD_REFERENCE.md                 # theory + experimental design — read before editing the notebook
├── experiment_plans/                   # ALL future experiment plans go here. One .md per experiment + experiment_template.md
├── figures/                            # exported plots (fig_<experiment>.png). Referenced from experiment_plans/
├── scripts/check_experiments_figures.py  # maintenance check: every plan with Status=run must have its figure on disk
├── .claude/settings.json               # Stop hook runs the maintenance check at end of every session
├── environment.yml                     # conda env spec
├── bilinear-decomposition/             # upstream repo (read-only). Installed -e into env.
│   ├── tutorials/  0_introduction, 1_image, 2_language
│   ├── exercises/  0_decomposition.ipynb (skeleton — required reading)
│   └── src/image, src/language, src/sae, src/shared, src/toy
└── CLAUDE.md, INDEX.md, PLAN.md, LOG.md
```

Flow: read tutorials 0+1 → understand bilinear MLP eigendecomposition → identify why orthogonality of eigenvectors yields "superposed" features → propose & implement a non-orthogonal tensor decomposition (sparsity is *one* prior, not the only one) → run on MNIST → screenshot interpretable components → write up.

### Notebook naming (memorize this — easy to confuse)
- **Our primary notebook** = `main_experiments.ipynb` at the repo root. All real work lives here.
- **"0-tutorial"** = `bilinear-decomposition/tutorials/0_introduction.ipynb` (the first tutorial — bilinear MLP basics).
- **"1-images"** = `bilinear-decomposition/tutorials/1_image.ipynb` (the *other* tutorial — MNIST workflow).
- **"0-exps"** = `bilinear-decomposition/exercises/0_decomposition.ipynb` — **Thomas's code**, the exercise skeleton we're extending. (Thomas Dooms is one of our two mentors.)

When the user says "the tutorial" they mean 0-tutorial; "the other tutorial" or "the image one" is 1-images; "Thomas's code" or "the exercise" is 0-exps.

## Conventions
- **No LaTeX in chat.** Claude Code does not render LaTeX, so `$...$`, `\mathbf{}`, etc. show up as raw source. Use Unicode (ρ, σ, π, ·, ², ∑, ∫, ∂) or code blocks with ASCII math instead. (LaTeX is fine inside Anki/Obsidian cards and the final report — those render it.)
- Application stream: **Goodfire only.** Don't tailor framing toward any other stream.
- Time budget per the prompt: ~1h setup (tutorials), 1–2h experimenting. Don't over-engineer; mentors care about thought process and proposed experiments, not polish.
- Work happens in `main_experiments.ipynb` at the repo root. Treat `bilinear-decomposition/` as read-only upstream — don't edit, just import.
- **Experiment plans live in `experiment_plans/`.** Every new experiment idea — before any code — gets a markdown plan copied from `experiment_template.md`. Template forces you to articulate (a) how the approach addresses each of the five targets [short, interpretable, fidelity, sharing, orthogonality] and (b) which of the three intervention levels [decomposition family / parameterization / loss] it touches. Plans are the design artifact mentors will read; the notebook is just the execution.
- **Figures live in `figures/`.** Each experiment's plan references a `figures/fig_<name>.png`. When you mark a plan `Status: run`, the figure must exist on disk.
- **Maintenance check.** `scripts/check_experiments_figures.py` walks every plan and asserts that `Status: run` ↔ figure-exists. It runs automatically as a Stop hook (see `.claude/settings.json`) at the end of every session — if it fails, fix the drift before handoff (either save the figure or downgrade the status). Run manually anytime: `python3 scripts/check_experiments_figures.py`.

## Dev commands
```bash
# one-time env setup
git clone https://github.com/tdooms/bilinear-decomposition.git  # gitignored — bootstrap fresh checkouts
conda env create -f environment.yml
conda activate marsv
# (the -e install of ./bilinear-decomposition is included in environment.yml)

# run the notebook
conda activate marsv && jupyter lab main_experiments.ipynb
```

Upstream's `pyproject.toml` pins a `pytorch-cu126` index for non-darwin platforms; on this Mac (Apple Silicon, no CUDA) we install plain torch from PyPI. If you move to a CUDA box, switch to upstream's `uv sync` flow inside `bilinear-decomposition/` instead, and set `device="cuda"` in the notebook.

## Setup
Scaffolded from `~/Developer/agent-scaffold`. The upstream `bilinear-decomposition` repo lives in-tree (not a submodule); installed editable so notebook imports work without sys.path hacks.

---

## Agent Protocol

You are one of many short-lived Claude sessions working on this project. The user relies on Claude to write code — knowledge must transfer between sessions via docs, not memory. You will not remember prior conversations. The docs are your memory.

### Before starting work — required reading

**Project state (every session, no exceptions):**
1. `INDEX.md` — repo layout.
2. `PLAN.md` — what's done, what's next.
3. `LOG.md` (latest entry) — where the previous session left off.
4. Any `*_REFERENCE.md` files listed in `INDEX.md` relevant to your task. Don't guess at APIs / platform behavior — it's documented for a reason.
5. `task-decomposing-weights.pdf` — the MARS V prompt, if you haven't internalized it.

**Substantive context (every session — these are short, read them all):**
6. `bilinear-decomposition/tutorials/0_introduction.ipynb` ("0-tutorial") — Thomas's bilinear MLP basics + per-class eigendecomposition. The math everything else extends.
7. `bilinear-decomposition/tutorials/1_image.ipynb` ("1-images") — MNIST classifier setup, interaction tensor `B`, the visualization workflow we copy.
8. `bilinear-decomposition/exercises/0_decomposition.ipynb` ("0-exps", a.k.a. "Thomas's code") — the skeleton CP decomposition we're extending. Skim the `Sparse` class and the training loop.
9. `main_experiments.ipynb` — *our* primary notebook. See its current state before adding cells.
10. `experiment_plans/` — every existing experiment plan. New experiments go here as `experiment_NN_<slug>.md`, copied from `experiment_template.md`. The template lists the five evaluation targets (short, interpretable, fidelity, sharing, orthogonality) and the three intervention levels (decomposition family, parameterization, loss) — every plan must address both.

Don't skip items 6–10 because "you read them last session." You didn't — you're a new session.

### During work

#### Planning (required for non-trivial tasks)
Before making changes, write a short ASCII plan and show it to the user:

```
+-------------------------------------+
| Task: <short description>           |
+-------------------------------------+
| 1. <step>                           |
| 2. <step>                           |
|    - <substep>                      |
| 3. <step>                           |
+-------------------------------------+
```

Wait for confirmation before proceeding. Keep plans concise.

#### Recap (required after completing each action)
After completing work, show an ASCII recap:

```
+-------------------------------------+
| Recap: <short description>          |
+-------------------------------------+
| Files edited:                       |
|  * path/to/file                     |
|    - <what changed>                 |
| Insights saved:                     |
|  > REFERENCE_DOC.md                 |
|    - <what was documented>          |
+-------------------------------------+
```

#### Documenting new knowledge
When you learn something non-obvious — a platform gotcha, an API quirk, a pattern that works — write it to the appropriate `*_REFERENCE.md` file immediately. If no reference doc exists for that domain yet, create one and add it to `INDEX.md`. This is how you pass knowledge to the next session.

### Handoff (end of every conversation)
When the user runs `/close` or the conversation is ending, complete this checklist:
1. Update any `*_REFERENCE.md` files with patterns learned this session.
2. Update `PLAN.md` — mark completed items `[x]`, add new items if the plan changed.
3. Append a dated entry to `LOG.md`: what changed, what's unfinished, what the next session should pick up.
4. Update `INDEX.md` if any files were added or removed.
5. If anything is half-finished, note it clearly in `LOG.md` so the next agent doesn't have to guess.
