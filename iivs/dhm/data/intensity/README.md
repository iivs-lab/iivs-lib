# `iivs.dhm.data.intensity`

Readers, writers, and lazy sequences for Lyncée Tec Koala **intensity**
reconstructions — the float32 intensity image Koala exports alongside phase, and
its uint8 display preview.

Intensity mirrors [`phase`](../phase/README.md) but carries **no unit or height
scale**: the `.bin` / `.txt` header is just the geometry (Koala writes the
phase-only bytes as a no-op sentinel), so there is no `PhaseUnit` analogue and no
`target_unit`. It still has the shared, unit-less `value_range()` (the data's
`(min, max)`), but no unit-aware overload and no `phbounds.txt` display-bounds
record — its previews are normalized per frame.

## Endpoints by format

| Koala source | Single image | Sequence (list / folder) |
| --- | --- | --- |
| `Float/Bin` (float32, with header) | `load_intensity_bin` / `save_intensity_bin` / `read_intensity_bin_header` | `IntensityBinList` / `IntensityBinFolder` |
| `Float/Txt` (float32, with header) | `load_intensity_txt` / `save_intensity_txt` / `read_intensity_txt_header` | `IntensityTxtList` / `IntensityTxtFolder` |
| `.npy` (float32, **header-less**) | `load_intensity_npy` / `save_intensity_npy` | `IntensityNpyFolder` |
| `Image/*.tif` (**uint8 preview**) | — | `IntensityTifList` / `IntensityTifFolder` |

- A **`*List`** wraps an explicit, arbitrary list of files; a **`*Folder`**
  auto-discovers `{index:05d}_intensity.<ext>` files under one root and shares a
  single `IntensityBinHeader`.
- `.npy` is header-less, so its only metadata, `pixel_size`, is passed to the
  `IntensityNpyFolder` constructor and shared by every frame.
- The `Image/*.tif` previews are an 8-bit visualization
  (`IntensityImageSequence`), not the quantitative float source
  (`IntensityFloatSequence`); the LZW previews decode via core `imagecodecs`.
  Verified against a real acquisition, Koala normalizes each intensity preview
  **per frame** (its own min/max → 0–255), *not* globally — so there is no
  bounds record and no `Image → Float` reconstruction here (unlike `phase`,
  whose previews share one global `phbounds.txt`). Use the `Float` source for
  quantitative intensity.
- The phase and intensity `.bin` formats share the `common.KoalaBinHeader` base;
  `IntensityBinHeader` exposes `width`, `height`, `pixel_size`, and the
  geometry conveniences (`shape`, `field_of_view[_um]`, `pixel_size_um`, ...).

### Format-agnostic entry points

When the format is only known at runtime (from a path's suffix), these pick the
right symbol for you, over `.bin` / `.txt` / `.npy`:

| Entry point | Picks |
| --- | --- |
| `load_intensity(path, *, return_header=False)` | `load_intensity_{bin,txt,npy}` — the image, or `(image, header)` when `return_header` (with `header` `None` for the header-less `.npy`) |
| `read_intensity_header(path)` | `read_intensity_{bin,txt}_header` (**`.npy` excluded** — header-less, **raises**; unlike `load_intensity`'s optional `None`) |
| `save_intensity(path, data, ...)` | `save_intensity_{bin,txt,npy}` (`.npy` ignores `pixel_size`, with a warning) |
| `intensity_list(files)` | `Intensity{Bin,Txt}List` by the files' shared extension |
| `intensity_folder(root)` | `Intensity{Bin,Txt,Npy}Folder` by the folder's contents (an `.npy` folder needs `pixel_size`) |

`intensity_folder` discovers the format with `kaparoo`'s `search_files` + a
`Regex` filter, and takes the same `prefer` argument as `phase_folder` to resolve
a folder holding more than one format (`None` raises; a format or priority
sequence picks the first present one).

## The sequence interface

Every intensity sequence is a `kaparoo.data.sequences.DataSequence`. Most of what
you call is **inherited** and so does not appear in this package's source — it is
listed here so you do not have to chase base classes.

**All intensity sequences** (list, folder, `.npy`, preview):

| Call | Result |
| --- | --- |
| `seq[i]` | the image at `i` (a fresh `np.ndarray`) |
| `seq[a:b]` | a sequence of images (lazy slice) |
| `len(seq)` | number of frames |
| `for img in seq:` | iterate images, lazily |
| `seq.get_item(i)` | the image at `i` (what `seq[i]` calls) |
| `seq.get_meta(i)` | the per-item metadata — the source `Path` |
| `seq.get_pair(i)` | `(image, meta)` together |
| `seq.get_items(idxs)` / `get_metas(idxs)` / `get_pairs(idxs)` | the batched forms over a list of indices |

**Float sources** (`Bin` / `Txt` / `Npy`, i.e. `IntensityFloatSequence`) add:

| Call | Result |
| --- | --- |
| `seq.get_header(i)` | the header of file `i`, **without** decoding pixels |
| `seq.load_with_header(i)` | `(image, header)` in a single read |

**Lists** (`*List`) add `seq.get_file(i)` and `seq.files`.

**Folders** (`*Folder`) add `seq.root`, `seq.header` (the one shared header),
`seq.frame_shape` (`(height, width)`), and `seq.validate(level=...)` (re-check
`"names"` / `"headers"` / `"data"`).

> Items are fresh, writable copies (ready for `torch.from_numpy`); the sequences
> pickle to just their path(s), so they are cheap for a PyTorch `DataLoader`.

## Conversion

`convert_intensity_folder(root, folder, *, ext)` re-encodes a folder into a new
numbered folder under `root`; `convert_intensity_list(sequence, *, ext)`
rewrites each list file in place with the new suffix. Targets: `"bin"` / `"txt"`
/ `"npy"` (all lossless float32).

For a composed or transformed sequence (a `kaparoo` `ConcatSequence`, a sliced
view, …) — which has no folder header — use
`save_intensity_folder(root, images, *, ext, pixel_size=...)`, supplying
`pixel_size` explicitly; `convert_intensity_folder` is the file-folder
convenience that reads it off the header for you.

## Time-lapse composition (`IntensityGroup`)

`IntensityGroup(intensity_dir)` opens one acquisition's `Intensity/` folder:
`.bin_folder` / `.txt_folder` (which may coexist), `.quantitative` (`.bin`-preferred),
`.tif_folder` (the uint8 `Image` preview), the shared `.num_frames` / `.frame_shape`,
and the `.is_consistent` (tolerant) / `.is_usable` (has quantitative data and is
consistent) cross-format checks. The folder's layout is
the `INTENSITY_TREE` `hierarchy` spec, and the format-specific searches
`search_intensity_bin_folders(root, ...)` / `search_intensity_txt_folders` /
`search_intensity_tif_folders` return that one folder (an `IntensityBinFolder` /
`IntensityTxtFolder` / `IntensityTifFolder`) for every time-lapse under `root` that has
it (via `kaparoo`'s `search_dirs`, with `predicate` a check on the opened folder). These
are what
[`iivs.dhm.data.timelapse`](../README.md#opening-a-whole-time-lapse-koalatimelapse)'s
`KoalaTimelapse` composes for the intensity modality.

## Example

```python
from iivs.dhm.data.intensity import IntensityBinFolder

acq = IntensityBinFolder("scan/Intensity/Float/Bin")

acq.header.pixel_size_um   # acquisition geometry
acq.frame_shape            # (H, W)
img = acq[0]               # first frame (float32)
img, hdr = acq.load_with_header(0)

for frame in acq:          # lazy iteration
    ...
```
