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
├── environment.yml                     # conda env spec
├── bilinear-decomposition/             # upstream repo (read-only). Installed -e into env.
│   ├── tutorials/  0_introduction, 1_image, 2_language
│   ├── exercises/  0_decomposition.ipynb (skeleton — required reading)
│   └── src/image, src/language, src/sae, src/shared, src/toy
└── CLAUDE.md, INDEX.md, PLAN.md, LOG.md
```

Flow: read tutorials 0+1 → understand bilinear MLP eigendecomposition → identify why orthogonality of eigenvectors yields "superposed" features → propose & implement a non-orthogonal tensor decomposition (sparsity is *one* prior, not the only one) → run on MNIST → screenshot interpretable components → write up.

## Conventions
- **No LaTeX in chat.** Claude Code does not render LaTeX, so `$...$`, `\mathbf{}`, etc. show up as raw source. Use Unicode (ρ, σ, π, ·, ², ∑, ∫, ∂) or code blocks with ASCII math instead. (LaTeX is fine inside Anki/Obsidian cards and the final report — those render it.)
- Application stream: **Goodfire only.** Don't tailor framing toward any other stream.
- Time budget per the prompt: ~1h setup (tutorials), 1–2h experimenting. Don't over-engineer; mentors care about thought process and proposed experiments, not polish.
- Work happens in `main_experiments.ipynb` at the repo root. Treat `bilinear-decomposition/` as read-only upstream — don't edit, just import.

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

### Before starting work
1. Read `INDEX.md` to understand the repo layout.
2. Read `PLAN.md` to see what's done and what's next.
3. Read `LOG.md` (latest entry) to understand where the last session left off.
4. Read any `*_REFERENCE.md` files listed in `INDEX.md` before writing code that touches those domains. Do not guess at APIs or platform behavior — it's documented there for a reason.
5. Read `task-decomposing-weights.pdf` if you have not internalized the prompt.

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
