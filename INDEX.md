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
| File | Purpose |
|------|---------|
| `main_experiments.ipynb` | Primary notebook: MNIST classifier + the four CP decomposition experiments |
| `tensor-decomposition.ipynb` | Side scratch: tiny `toy.Model` exploration of the symmetric interaction tensor |
| `environment.yml` | Conda env spec (`marsv`) for running the notebook |
| `bilinear-decomposition/` | Upstream repo (read-only). Tutorials, exercise skeleton, and `src/` modules imported by our notebook |
| `bilinear-decomposition/tutorials/0_introduction.ipynb` | Required reading per the prompt |
| `bilinear-decomposition/tutorials/1_image.ipynb` | Required reading per the prompt |
| `bilinear-decomposition/exercises/0_decomposition.ipynb` | Skeleton the prompt asks us to extend |

## Data
| Path | Purpose |
|------|---------|
| `data/MNIST/` | MNIST blob downloaded by `MNIST(train=...)` — gitignored |
