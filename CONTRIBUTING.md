# Contributing to Quire

Thanks for your interest. Quire is a small project with a clear shape: a Python engine
(SymPy, NumPy, SciPy), a module system, and a plain-JavaScript worksheet UI. Most
contributions fall into one of four kinds.

## Ways to contribute

- **A bug or a wrong answer.** Open an issue with the worksheet lines that reproduce it
  and what you expected. A wrong symbolic result is a bug: the benchmark in `bench/`
  exists so that regressions are caught.
- **A module.** Domain knowledge is the most valuable thing you can add. A module is a
  directory `modules/<name>/module.py` with a `register(api)` function; see
  `modules/mechanics/module.py` for a plain one and `modules/quantum/module.py` for one
  that adds a plot kind. Keep names unique across modules (`Registry.conflicts()` is
  tested), give every function a `signature`, `doc`, `category` and an `example`, accept
  quantities with units where the physics has them, and add a test with a few
  hand-checkable numbers.
- **A worksheet example.** Files in `examples/` are opened from the Examples menu. A good
  example is short, reads like a lesson, and exercises what it shows.
- **Documentation.** The README is the manual; keep it in step with the code.

Questions and ideas go to [Discussions](https://github.com/thepacket/quire/discussions).

## Working on the code

```bash
uv venv && uv pip install -e '.[dev]'        # or: python -m venv .venv && pip install -e '.[dev]'
.venv/bin/python -m quire                    # runs at http://127.0.0.1:8765
set -o pipefail; .venv/bin/python -m pytest -q tests | tail -3
```

The suite takes about a minute. Run it before every pull request; a pull request
that changes symbolic behaviour should also keep `bench/problems.py` and
`bench/hard.py` green (`python -m bench.run`).

Conventions that keep the worksheet language coherent:

- Definitions bind at parse time, top to bottom. Nothing runs in the browser but
  rendering, the arithmetic preview and the slider fast path.
- Unit symbols of one letter are units only directly after a number; longer names are
  always units. Never call `Abs` on a unit-bearing expression; convert quantity by quantity.
- Errors should say what to write instead. A message that names the fix
  ("write `3*m` for three times yours") beats a stack trace.
- The UI stays dependency-free apart from KaTeX and the bundled Plotly. No build step.

## Pull requests

Keep them focused: one module, one bug, one feature. Describe what changed and how you
checked it. Add or update tests. By contributing you agree that your work is released
under the MIT License that covers the project.
