# Environment

One subfolder per pipeline component, since each has its own dependency
graph (`hiroace/` needs a GPU-container torch build, `hydro/` needs conda +
editable installs of DiffHydro/DiffRoute). Don't share venvs across
components unless their dependencies actually match.

```
environment/
├── hiroace/isambard/   # container + venv
└── hydro/isambard/     # conda env
```

Inside each component, `<site>/` holds the concrete setup for one HPC
(container image, wheel index, module system — all site-specific). The
component's own README explains the pattern; a new site is a new sibling
folder, not an edit to an existing one.
