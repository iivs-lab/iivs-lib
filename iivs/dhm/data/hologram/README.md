# `iivs.dhm.data.hologram`

Readers, writers, and lazy sequences for Lyncée Tec Koala **holograms** — the
raw uint8 interferograms, in Koala's single multi-frame `.raw` stack or as
per-frame `.tif` / `.npy`.

Holograms are 8-bit and carry **no physical calibration** (no pixel size, unit,
or height scale), so the sequences are simpler than
[`phase`](../phase/README.md) / [`intensity`](../intensity/README.md): there is
no float header, no `get_header` / `load_with_header`, and no unit handling.

## Endpoints by format

| Koala source | Reader / writer | Sequence |
| --- | --- | --- |
| `Holograms/holo.raw` (one multi-frame stack) | `read_hologram_raw_header` / `save_hologram_raw` | `HologramRawFile` |
| `.tif` (one uint8 image per file) | `load_hologram_tif` / `save_hologram_tif` | `HologramTifList` / `HologramTifFolder` |
| `.npy` (one uint8 image per file) | `load_hologram_npy` / `save_hologram_npy` | `HologramNpyFolder` |

- **`HologramRawFile`** wraps a single `.raw` file (a 16-byte
  `HologramRawHeader` then row-major frames) as a lazy, read-only `np.memmap`,
  so a large multi-frame file is never loaded whole.
- A **`*List`** wraps an arbitrary list of `.tif` files (any location, may even
  differ in shape); a **`*Folder`** auto-discovers `{index:05d}_holo.<ext>`
  files under one root and shares one `frame_shape`.
- `save_hologram_raw` accepts a single image, an `(N, H, W)` stack, or any uint8
  `DataSequence` (a `HologramSequence`, or a `kaparoo` composer such as a
  `ConcatSequence`), and streams frames one at a time (so re-encoding a big
  folder never materializes the whole stack).

## The sequence interface

Every hologram sequence is a `kaparoo.data.sequences.DataSequence`. Most of what
you call is **inherited** and so does not appear in this package's source — it is
listed here so you do not have to chase base classes.

**All hologram sequences** (`.raw`, `.tif`, `.npy`):

| Call | Result |
| --- | --- |
| `seq[i]` | the uint8 hologram at `i` (a fresh `np.ndarray`) |
| `seq[a:b]` | a sequence of holograms (lazy slice) |
| `len(seq)` | number of frames |
| `for frame in seq:` | iterate frames, lazily |
| `seq.get_item(i)` | the hologram at `i` (what `seq[i]` calls) |
| `seq.get_meta(i)` | the per-item metadata (see below) |
| `seq.get_pair(i)` | `(image, meta)` together |
| `seq.get_items(idxs)` / `get_metas(idxs)` / `get_pairs(idxs)` | the batched forms over a list of indices |
| `seq.value_range(index=None)` | the uint8 `(min, max)` — global (cached), or of frame `index` |

The **metadata** `get_meta(i)` returns differs by backing:

- `HologramRawFile` → the **frame index** (`int`); also exposes `seq.path` (the
  `.raw` file), `seq.header` (a `HologramRawHeader`: `width` / `height` /
  `bit_depth` / `frame_count` and `shape`), `seq.frame_shape`, and `seq.frames`
  — the whole stack as a lazy, read-only `(frame_count, H, W)` memmap for
  zero-copy bulk access.
- `HologramTifList` / `*Folder`, `HologramNpyFolder` → the source **`Path`**;
  lists add `seq.get_file(i)` / `seq.files`, folders add `seq.root`,
  `seq.frame_shape`, and `seq.validate(level="names" | "data")`.

> Each indexed item is a fresh, writable copy (ready for `torch.from_numpy`);
> use `HologramRawFile.frames` directly for a zero-copy view. The sequences
> pickle to just their path(s), so they are cheap for a PyTorch `DataLoader`.

## Conversion

`convert_hologram_sequence(dest, sequence, *, ext)` re-encodes any hologram
sequence to `dest`:

- `ext="raw"` → one multi-frame `.raw` stack at `dest` (streamed frame by
  frame).
- `ext="tif"` / `"npy"` → one numbered file per frame in the `dest` folder.

Every format is lossless uint8. The `sequence` may be any uint8 `DataSequence`,
not just a file-backed `HologramSequence` — a `kaparoo` composer (e.g. a
`ConcatSequence` of acquisitions) works too, so holograms need no separate
`save_hologram_folder`.

## Time-lapse composition (`open_holograms`)

`open_holograms(holo_dir)` opens one acquisition's `Holograms/` folder as a single
`HologramSequence` — the `holo.raw` stack if present, else the numbered tif folder —
raising if a folder holds both (a real acquisition yields only one). The folder's layout
is the `HOLOGRAM_TREE` `hierarchy` spec, and `search_holograms(root, ...)` returns the
hologram sequence of every time-lapse under `root` that has a `Holograms/` folder (via
`kaparoo`'s `search_dirs`, with `predicate` a check on the `HologramSequence`). A
`Holograms/` holding both a `.raw` stack and `.tif` previews is ambiguous;
`on_conflict="skip"` (default) drops that time-lapse and warns so one malformed
acquisition does not abort the scan, while `"raise"` aborts. `search_ambiguous_holograms(root, ...)`
is the auditing counterpart, returning those ambiguous `Holograms/` folders to fix. The
ambiguity is a distinct `AmbiguousHologramsError` (a `ValueError`), so a corrupt
`holo.raw` surfaces instead of being mistaken for it. All three are what
[`iivs.dhm.data.timelapse`](../README.md#opening-a-whole-time-lapse-koalatimelapse)'s
`KoalaTimelapse` composes for the holograms.

## Example

```python
from iivs.dhm.data.hologram import HologramRawFile, convert_hologram_sequence

holos = HologramRawFile("scan/Holograms/holo.raw")

holos.header.frame_count      # frames in the stack
holos.frame_shape             # (H, W)
frame = holos[0]              # writable copy of frame 0
stack = holos.frames          # zero-copy (frame_count, H, W) memmap

# Re-encode the stack into a per-frame .tif folder:
convert_hologram_sequence("scan/Holograms/tif", holos, ext="tif")
```
