# Japan Hydro Pipeline

DiffHydro/DiffRoute-based hydrological modeling for Japan. Ported from
`japan-hydro-pipeline` (git-tracked files only, same set as `git ls-files`
there) — `environment.yml` lives in `../environment/hydro/isambard/`.

```
hydro/
├── exp_helpers.py, Analysis.ipynb, compute_interpolation_weights.ipynb
├── modified_code/       # same 3 files, kept separately as in the source repo
├── exploration/         # data_exploration.ipynb
├── pipeline/            # clean, testable rebase of Analysis.ipynb -- see pipeline/README.md
└── scripts/isambard/    # vscode_tunnel.sh (sbatch + code tunnel)
```

`pipeline/` splits `Analysis.ipynb`'s notebook-global logic into a pure
data-loading/alignment module (tested here, against real data) and a thin
`diffhydro`/`xtensor`-wrapping layer (implemented against the actual
library source, but not runnable anywhere outside Isambard) — plus
separate `evaluate` (real forcing + discharge, NSE) and `predict` (any
forcing, no ground truth, e.g. HiRO-ACE-driven) CLI entry points. See
`pipeline/README.md` for the full design writeup and exactly what's
verified vs. still needs an Isambard run.

`modified_code/` duplicates `Analysis.ipynb` and
`compute_interpolation_weights.ipynb` with different content (`exp_helpers.py`
is identical to the root copy) — both were already tracked as separate
files in the source repo, so kept as-is rather than guessing which one is
authoritative.

**Not carried over:**
- `data/` — was already `.gitignore`d in the source repo (large `.zarr`/`.pkl`/`.nc`)
- `results/default.pt` (3.5 MB) — was untracked in the source repo (never `git add`ed); a trained checkpoint, add deliberately if it should be versioned or fetched separately

Depends on editable installs of sibling repos (`DiffHydro`, `DiffRoute`,
`xtensor`) — see `../environment/hydro/README.md`.
