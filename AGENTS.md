# Agent guide — `iivs-lib`

Guidance for AI coding assistants working on this project.
`CLAUDE.md` loads this file via the `@AGENTS.md` import.

## Project

- Package: `iivs/`
- Python:  3.13+
- Kind:    distributable library (`uv_build` backend)

## Toolchain

The Astral toolchain — keep it unless there is a clear reason to change:

- `uv`   — environment, locking, running
- `ruff` — linting + formatting
- `ty`   — type checking
- `pytest` — testing
- `pytest-cov` — coverage measurement and threshold gate

## Commands

```bash
uv sync --group dev      # create/refresh the environment
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type-check
uv run pytest            # run tests (coverage included by default)
uv run pytest --no-cov   # skip coverage for quick iteration
```

Coverage is measured by `pytest-cov` against `iivs/` with
branch tracking; the `fail_under` gate lives in `pyproject.toml`
(`0` = measure only — raise it once you have a baseline).


## Conventions

- Keep code fully typed — `ty` runs with `error-on-warning`.
- Fix `ruff` findings rather than suppressing them, unless there is a
  clear, commented reason.
- Tests live in `tests/` and may use bare `assert` (ruff `S101` is
  waived there).
- Mirror the package layout under `tests/`: `iivs/sub/mod.py`
  is tested by `tests/sub/test_mod.py`. Keep `__init__.py` markers in
  test subpackages (matches the `INP` ruff rule).
- Test layout: flat module-level `def test_*` functions by default;
  reach for a plain `class TestX:` (grouping only, no inheritance) to
  organize a large or multi-feature surface. Don't mix the two styles
  within one file.
- Not every source file needs a dedicated test file — types-only
  modules, re-export `__init__.py` markers, and details covered through
  a public-facing module are intentional exceptions.
- Cross-module test helpers live in `tests/<pkg>/helpers.py`; shared
  fixtures and per-package config go in `conftest.py` (see the fixtures
  note below).
- Test quality: assertions check concrete return values AND side
  effects, not merely "no exception raised"; error paths use
  `pytest.raises(..., match=...)`; verify numbers against independently
  computed values (not the implementation's own output); make timing /
  IO deterministic (an injected clock, fault injection) rather than
  flaky. A good test fails on *subtle* breakage, not just obvious
  breakage. When a contract is "one batched call per source", verify it
  with a spy, not only by the result.
- Name modules after their primary concept — usually a **singular**
  noun (`header.py`, `sequence.py`, `timestamp.py`), even when the
  module manages many instances (a sequence module is still
  `sequence.py`). Reserve **plural** names for modules that are a flat
  collection of co-equal peers with no single dominant concept
  (`constants.py`, `exceptions.py`, `utils.py`). The module name need
  not mirror a data file it handles (`timestamp.py` reads
  `timestamps.txt`).
- `__all__` declares **that module's own** public API, and nothing more.
  A package's `__init__.py` re-exporting a submodule's names is a
  curation judgement, never an obligation: `numpy` declares
  `linalg.solve` in `numpy.linalg.__all__` and offers it from there
  alone. So "name X is in a submodule's `__all__` but not in its
  package's" is not by itself a finding — do not open it as one. What
  *is* a finding: a public signature naming a type no caller can reach.
  Three placements, each deliberate, all present here:
  - **package `__all__`** — the headline surface (`KoalaTimelapse`
    would be, were `iivs.dhm.data` a hub; `PhaseGroup` is).
  - **module `__all__` only** — supported, reached by module path
    (`validate_ndim`, `resolve_height_scale`).
  - **public name, undeclared** — reachable, no promise
    (`PhaseImageView`: `to_image` returns the exported abstract
    `PhaseImageSequence`, so the concrete view is an internal detail).
  Note `__all__` gates only `from x import *`; an undeclared public name
  imports fine either way. When a package deliberately omits something,
  say why at the omission, as `iivs.dhm.analysis` does for its Torch
  twins and `iivs.common.data` for the composable validators.
- Fixtures live in `tests/conftest.py` when you need them — modern
  default, applies to `tests/` only. Use a root `conftest.py` only for
  `pytest_plugins` declarations, doctest fixtures shared with source
  files, or project-wide collection hooks.
- `ty` has no plugin system; rely on standard typing (PEP 681
  `dataclass_transform`, `.pyi` stubs), not type-checker plugins.
- Suppress `ty` errors with `# ty: ignore[<error-name>]` using `ty`'s
  own error names (e.g. `invalid-argument-type`), not mypy/pyright
  codes. Always include the specific code rather than bare
  `# ty: ignore` — bare suppressions can mask future regressions.

## Python style

`ruff` enforces most of this — run `uv run ruff check --fix` rather than
applying it by hand.

- Every module *with annotations* starts with `from __future__ import
  annotations` (ruff isort `required-imports`). `__init__.py` package
  markers are exempt (per-file-ignore `I002`): whether an empty marker or
  a pure re-export hub, they carry no annotations, so the import would be
  a no-op. `constants.py` and other bare-value modules keep it under the
  blanket rule (the no-op line costs less than a per-file exception, and
  guards a future annotation edit).
- Use builtin generics — `list`, `dict`, `tuple`, `type` — never
  `typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Type` (ruff `UP`).
- Imports are grouped and sorted: standard library, third party, first
  party, then a trailing `if TYPE_CHECKING:` block (grouped the same
  way). Within a group `import X` precedes `from X import Y`; entries are
  alphabetical.
- Docstrings are optional — write them where they clarify intent, not
  mechanically. "Mechanically" targets two habits to avoid: comments
  (or docstrings) that merely restate the code, and a base class whose
  docstring explains itself in terms of its specific subclasses —
  except a *closed* hierarchy's base, which may name its subclasses as
  a deliberate family map. Even then, a shared, technique-agnostic
  layer never names a downstream consumer in its docstrings: an
  `iivs.common` docstring must not mention a specific technique or
  modality (`dhm`, `phase`, `intensity`, `hologram`) — `common` cannot
  depend on what depends on it. It is *not* a licence to leave a
  consumed method bare. When written, document *intent and contracts, not
  mechanism*:
  - Lead with a one-line summary — a declarative noun phrase for
    classes ("An ordered, read-only view over ..."), an imperative
    verb phrase for functions and methods ("Yield successive windows
    from `items`."). Two kinds take a noun phrase instead: a property
    getter ("The reporting unit ...") and a boolean-returning *method*,
    which leads with "Whether ..." ("Whether `path` exists."). A
    boolean *function* stays imperative ("Test whether a path exists.").
  - A concrete public method a caller consumes must be self-explainable
    from its own docstring and signature — never lean on an inherited
    parent docstring. Abstract base methods document only the generic
    contract and never name a specific subclass.
  - Surface what callers cannot infer from the signature alone:
    invariants, edge cases, what subclasses must override, policy
    trade-offs. Skip restating what the code already shows.
  - *Contracts, not mechanism* — litmus: would a caller's behaviour
    change if the line were false? Cut prose that only says how the code
    is wired: delegation ("a thin wrapper over X", "the free-function
    alias for X"), the methods a base provides or a subclass inherits,
    inheritance wiring ("the .bin codec over X", "inherited from X"), and
    storage / API internals ("held internally as an np.memmap",
    "numpy.load(allow_pickle=False)", "staged to a temp file and moved
    into place"). Keep the guarantee those deliver ("written atomically",
    "never loaded whole", "a pickled array is rejected"), relationship
    notes that carry behaviour ("the `.npy` twin of X"), and what a
    subclass must override. A closed hierarchy's base may still name its
    subclasses (the family map above); a technique-agnostic layer may not.
  - Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
    sections (`Args:`, `Returns:`, `Yields:`, `Raises:`,
    `Type Parameters:`); omit types from `Args:` since the signature
    already carries them. Custom sections (`Example:`, `Truth table:`,
    and ad-hoc labels) are welcome when they clarify a real pitfall or
    pattern.
  - Add an `Args:` / `Returns:` block only for what the summary and
    signature cannot already convey. When they make the behaviour
    obvious (a no-arg getter, a self-evident one-liner), the summary
    *is* the whole docstring; a `Returns:` that merely restates it is
    the mechanical habit above. Document an edge case shared across a
    family once on the class.
  - Reference identifiers in backticks (`my_method`, `param`,
    `MyClass.method`). Literal option values get backticks too
    (`"merge"`, `"error"`), as identifiers do.
  - Wrap prose and `Args` / `Returns` / `Raises` descriptions to fill
    each line toward the 88-column limit; don't break early when the next
    word still fits. Leave rST literal blocks (after `::`) and code
    examples verbatim.
  - No `--` (double-hyphen) in docstring or message prose; use a
    parenthesis, colon, or semicolon.
- Exception messages use a terse, lower-case house style: no leading
  capital, no trailing period. Prefer `f"<subject> must be <constraint>
  (got {value})"` for validation, or a short imperative for
  mutually-exclusive options (`"give exactly one of: height_scale, or
  wavelength and refractive_delta"`). Name the valid set or the fix, not
  just the failure (`"unsupported extension 'X': expected bin, txt"`),
  so the message tells the caller what to do next. Keep each message on
  one line — if it would wrap, shorten it or hoist a value to a local
  rather than splitting the string literal.
- Comments must earn their place: delete ones that restate the code. When
  an implementation note states intent or a contract, prefer promoting it
  to a docstring.
- Within a function body, separate logical groups with a single blank
  line and put a blank line before the final `return`; leave a tightly
  coupled one- or two-line body unbroken.
- In a long module, group related definitions under a boxed comment
  banner — a centred title between two `#`-bordered rules.
- Standalone runnable scripts carry PEP 723 inline metadata (the
  `# /// script` block). `uv` manages it (`uv add --script`); add or edit
  it by hand only when explicitly asked.

## Commit convention

Commit messages use a [Gitmoji](https://gitmoji.dev/) prefix and wrap
package/tool names in backticks:

```
<emoji> <Imperative summary; tool names in `backticks`>

<Optional body explaining *why*>
```

Common prefixes used in this project:

| Prefix | Use for                                       |
| ------ | --------------------------------------------- |
| ✨     | New feature                                   |
| ♻️     | Refactor (no user-visible behavior change)    |
| 🔥     | Remove dead / vestigial code                  |
| 🐛     | Bug fix                                       |
| 📝     | Docstrings, README, CHANGELOG                 |
| ✏️     | Typo or other small text fix                  |
| 💄     | Style (no behavior change)                    |
| ✅     | Tests added or updated                        |
| ⚡     | Performance optimization                      |
| 🏷️     | Type-hint-only change                         |
| 💬     | Code comment                                  |
| 🗑️     | Deprecation signal                            |
| 📦     | Re-export / packaging structure               |
| 🚚     | Move / rename files                           |
| ⬆️     | Bump a dependency or tool version             |
| 🔧     | Config (`pyproject.toml`, `ruff`, `ty`, ...)  |
| 🔖     | Release a version (commit + matching tag)     |

Keep commits single-purpose; don't rewrite published history; don't
skip git hooks. AI assistants append a `Co-Authored-By` trailer with
their own published identity (e.g. `Claude <noreply@anthropic.com>`).

## Releases

`iivs-lib` follows SemVer; in pre-1.0, a minor bump may carry
breaking changes.

Releases publish automatically via GitHub Actions **Trusted Publishing**
(OIDC — no PyPI token is stored anywhere). Procedure for `X.Y.Z`:

1. Move `CHANGELOG.md`'s `[Unreleased]` content into a dated
   `[X.Y.Z] - YYYY-MM-DD` section. Drop entries whose subject was both
   introduced *and* renamed / removed / fixed within the same cycle —
   upgraders never saw the intermediate state.
2. Bump `version` in `pyproject.toml`; `uv sync --group dev` to refresh
   `uv.lock`.
3. Commit `🔖 Release version X.Y.Z` (body references the `[X.Y.Z]`
   entry) and push.
4. Tag and push:
   `git tag -a vX.Y.Z -m "🔖 Release version X.Y.Z" && git push origin vX.Y.Z`.

The `v*.*.*` tag triggers `.github/workflows/publish.yml`: it reruns
CI across the OS matrix, then `build` first **verifies the tag matches
`pyproject.toml`'s version** (a mismatch fails fast) before building
the distributions once and uploading them as an artifact. It publishes
them to TestPyPI as a staging rehearsal (`skip-existing`, so a rerun is
safe), then — once you approve the `pypi` environment gate — publishes
the same artifacts to PyPI via OIDC. Finally a `github-release` job
**creates the GitHub Release automatically** — using the matching
`CHANGELOG.md` section as the notes and attaching the sdist + wheel —
so there is no manual release step after the PyPI gate. Keeping `build`
separate from publish keeps the `id-token: write` permission scoped
to the publish jobs only.

### One-time setup (before the first release)

1. **PyPI Trusted Publisher** — on PyPI, register a GitHub Actions
   publisher for the project: owner `iivs-lab`, repo
   `iivs-lib`, workflow `publish.yml`, environment `pypi`. Use
   PyPI's *pending publisher* form for the very first upload (before
   the project exists on PyPI).
2. **TestPyPI Trusted Publisher** — register the *same* publisher on
   [test.pypi.org](https://test.pypi.org/manage/account/publishing/)
   for the staging job: owner `iivs-lab`, repo
   `iivs-lib`, workflow `publish.yml`. The `testpypi` job has
   no environment, so leave that field blank.
3. **GitHub `pypi` environment** — create an environment named `pypi`
   in the repo settings and add yourself as a *required reviewer*, so
   every publish waits for explicit approval.


## Template

Generated from a copier template. `.copier-answers.yml` records the
answers; run `copier update --UNSAFE` to pull later template changes.

---

<!-- Add project-specific guidance below. -->
