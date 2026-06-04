# Agent guide — `iivs-lib`

Guidance for AI coding assistants working on this project.
`CLAUDE.md` loads this file via the `@AGENTS.md` import.

## Project

- Package: `iivs/`
- Python:  3.14+
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
- Name modules after their primary concept — usually a **singular**
  noun (`header.py`, `sequence.py`, `timestamp.py`), even when the
  module manages many instances (a sequence module is still
  `sequence.py`). Reserve **plural** names for modules that are a flat
  collection of co-equal peers with no single dominant concept
  (`constants.py`, `exceptions.py`, `utils.py`). The module name need
  not mirror a data file it handles (`timestamp.py` reads
  `timestamps.txt`).
- Group data-format code by modality, then by file format. Each modality
  under `iivs.dhm.data` (`phase`, `intensity`, `hologram`, `timestamp`)
  splits a multi-format modality into per-format modules plus a
  format-agnostic `core` and a `base` holding the abstract sequence types —
  e.g. `phase/{core,base,bin,txt,tif,npy}.py`, `hologram/{base,tif,raw,npy}.py`
  (a modality drops `core` when it has no format-agnostic logic of its own).
  Building blocks shared across modalities live in the data-root `common`
  package, split by concern into submodules and all re-exported from `common`
  (so `from iivs.dhm.data.common import X` reaches any of them): `bin` (the
  `KoalaBinHeader` base + `.bin` pixel I/O, used by `phase` and `intensity`),
  `txt` (the `Float/Txt` header/grid codecs `KoalaTxtHeaderCodec` /
  `parse_txt_grid`), `image` (the uint8-image folder/list codec bases + `.tif`
  reader: `ImageFileFolder` / `ImageFileList`, `ImageTifFolder` /
  `ImageTifList`, `load_uint8_tif`), `npy` (the `.npy` shape reader
  `read_npy_shape` + writer), `sequence` (the numbered-folder
  `SequentialFileFolder` template that every `*Folder` builds on, plus the
  `FrameShapedMixin`), `validation` (the float32/uint8 image validators), and
  `extension` (the filename helpers `ensure_file_extension` /
  `with_file_extension` / `numbered_name`). Keep the submodules acyclic
  (`extension`/`validation`/`bin`/`npy` are leaves; `sequence` → `extension`;
  `txt` → `bin`, `extension`; `image` → `validation`, `extension`,
  `sequence`). Prefer a shared base/template + a small mixin over copy-pasting
  across modalities.
- A `*Folder` is the auto-discovered special case of its `*List`, so it
  *subclasses the list* (mirroring kaparoo's `FileFolderSequence` ⊂
  `FileListSequence`) and reuses its `load_file`. `FileFolderSequence.__init__
  (root)` discovers the files and cooperatively calls the list's `__init__`
  down the MRO, so define the list *before* the folder in the module.
- `FILE_EXT` lives on the concrete `*List`, not the folder. The shared list
  base does the check -- `KoalaFloatFileList` (float) and `ImageFileList`
  (uint8) validate every given path's `.<FILE_EXT>` in `__init__` via
  `common.ensure_file_extension` (so a wrong-format file fails up front, not on
  decode); the `*Folder` inherits `FILE_EXT` for both discovery and that check.
  It is a one-line helper call in those bases, not a dedicated
  `ExtensionChecked*` class in the MRO. Single-file `*File` sources call
  `ensure_file_extension(path, ext)` in `__init__` themselves.
- The float32 list/folder machinery (the `.<FILE_EXT>` check, `get_meta`,
  `get_header`, `load_with_header`, the `header` / `frame_shape` /
  `_validate_content` of a numbered folder) lives **once** in
  `data.common.float`: `KoalaFloatFileList[H]` / `KoalaFloatFileFolder[H]`,
  generic in the header type `H` over an abstract ``(read_header, decode)``
  codec where `_decode` returns ``(image, header)``. Each modality's
  `<Modality>FileList` / `<Modality>FileFolder` base (in the modality's
  `base.py`, left out of `__all__` -- internal but underscore-free) just binds
  `H` and adds modality-only bits: phase overrides `_postprocess` for unit
  conversion and `_after_header` to resolve `target_unit`; intensity adds
  nothing. The concrete `*BinList` / `*TxtList` supply the codec *and*
  `FILE_EXT`; the `*BinFolder` / `*TxtFolder` just combine `<Modality>FileFolder`
  with their `*List`. So a new format is a couple of codec methods, not a copied
  list+folder. (Bind `H` with a **string** subscript -- `KoalaFloatFileList
  ["PhaseBinHeader"]` -- so the header import stays under `TYPE_CHECKING` and the
  `base` ↔ `bin` cycle is avoided; ruff cannot see the string as a use, so that
  one `TYPE_CHECKING` import carries `# noqa: F401`.)
- Mark cross-cutting capabilities with a mixin, not a class per modality.
  `data.common.SequentialFileFolder` mixes in `data.common.FrameShapedMixin`
  (forces `frame_shape`) for every numbered folder; a single-file source like
  `HologramRawFile` mixes it in directly. There is no `Uniform<Modality>
  Sequence`: "a uniform float phase sequence" is just
  `isinstance(x, PhaseFloatSequence) and isinstance(x, FrameShapedMixin)`.
- Split the role base when a modality has more than one pixel representation.
  `PhaseSequence[T, M]` is the dtype-generic modality base; `PhaseFloatSequence`
  (quantitative float32, from `Float/{Bin,Txt}`) and `PhaseImageSequence`
  (uint8 `Image/*.tif` preview) specialize it. A modality with a single
  representation (e.g. `Hologram`) keeps a flat `<Modality>Sequence`.
- Name sequence classes by role vs backing. Abstract role types keep the
  `Sequence` suffix (`PhaseSequence`, `HologramSequence`,
  `TimestampSequence`); concrete types drop it and read
  `<Modality><Format><Backing>`, the backing mirroring kaparoo's templates
  — `Folder` (`FileFolderSequence`), `List` (`FileListSequence`), `File`
  (`SingleFileSequence`): e.g. `PhaseBinFolder`, `PhaseBinList`,
  `HologramRawFile`. Pluralize the modality when it is also a single-item
  class, to mark the collection (`Timestamp` item → `TimestampsTxtFile`).
  Format tokens copy the literal extension in TitleCase (`Bin`, `Tif`,
  `Raw`, `Txt`); acronyms stay upper-case (`FPS`).
- Credit external data sources. Formats originating from a vendor tool name
  that vendor in docstrings/messages, and are acknowledged in the README and
  the owning package's docstring (e.g. Lyncée Tec Koala in `iivs.dhm.data`).
- Fixtures live in `tests/conftest.py` when you need them — modern
  default, applies to `tests/` only. Use a root `conftest.py` only for
  `pytest_plugins` declarations, doctest fixtures shared with source
  files, or project-wide collection hooks.
- `@override` marks overrides of a *named* base contract (`get_item`,
  `load_file`, `frame_shape`, `_read_header`, ...), where it guards against the
  base drifting. Do **not** put it on dunders (`__len__`, `__reduce__`,
  `__init__`): implementing a dunder is self-evidently an override, and some
  (like `__reduce__`) only "override" `object`, so the decorator is pure noise.
- `@abstractmethod` (not a bare `raise NotImplementedError`) for hooks a
  subclass *must* implement -- the codec hooks (`_read_header` / `_decode` /
  `load_file`), `frame_shape`, `KoalaTxtHeaderCodec._from_geometry`, and the
  `KoalaBinHeader` serializers -- so an incomplete subclass fails at
  construction, not at first call. Make the holder an ABC if it isn't already
  (e.g. `KoalaBinHeader(ABC)`). The exception is an *optional* hook like
  `SequentialFileFolder._validate_content`, which a names-only folder never
  calls: keep that a plain `raise NotImplementedError` so it is not forced on
  every subclass.
- `ty` has no plugin system; rely on standard typing (PEP 681
  `dataclass_transform`, `.pyi` stubs), not type-checker plugins.
- Suppress `ty` errors with `# ty: ignore[<error-name>]` using `ty`'s
  own error names (e.g. `invalid-argument-type`), not mypy/pyright
  codes. Always include the specific code rather than bare
  `# ty: ignore` — bare suppressions can mask future regressions.

## Python style

`ruff` enforces most of this — run `uv run ruff check --fix` rather than
applying it by hand.

- Every module starts with `from __future__ import annotations` (ruff
  isort `required-imports`). Empty `__init__.py` package markers are
  exempt.
- Use builtin generics — `list`, `dict`, `tuple`, `type` — never
  `typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Type` (ruff `UP`).
- Imports are grouped and sorted: standard library, third party, first
  party, then a trailing `if TYPE_CHECKING:` block (grouped the same
  way). Within a group `import X` precedes `from X import Y`; entries are
  alphabetical.
- Docstrings are optional — write them where they clarify intent, not
  mechanically on every function, class, or method. When written,
  document *intent and contracts, not mechanism*:
  - Lead with a one-line summary — a declarative noun phrase for
    classes ("An ordered, read-only view over ..."), an imperative
    verb phrase for functions and methods ("Yield successive windows
    from `items`.").
  - Surface what callers cannot infer from the signature alone:
    invariants, edge cases, what subclasses must override, policy
    trade-offs. Skip restating what the code already shows.
  - Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
    sections (`Args:`, `Returns:`, `Yields:`, `Raises:`,
    `Type Parameters:`); omit types from `Args:` since the signature
    already carries them. Custom sections (`Example:`, and ad-hoc
    labels) are welcome when they clarify a real pitfall or pattern.
  - Reference identifiers in backticks (`my_method`, `param`,
    `MyClass.method`).
- Exception messages use a terse, lower-case house style: no leading
  capital, no trailing period. Prefer `f"<subject> must be <constraint>
  (got {value})"` for validation, or a short imperative for
  mutually-exclusive options (`"give height_scale, or wavelength and
  refractive_delta (not both)"`). Keep each message on one line — if it
  would wrap, shorten it or hoist a value to a local rather than splitting
  the string literal.
- Comments must earn their place: delete ones that restate the code. When
  an implementation note states intent or a contract, prefer promoting it
  to a docstring.
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
CI across the OS matrix, builds the distributions once and uploads
them as an artifact, publishes them to TestPyPI as a staging
rehearsal, then — once you approve the `pypi` environment gate —
publishes the same artifacts to PyPI via OIDC. Keeping `build`
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
