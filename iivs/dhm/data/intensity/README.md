# `iivs.dhm.data.intensity`

Readers, writers, and lazy sequences for Lyncée Tec Koala **intensity**
reconstructions — the float32 intensity image Koala exports alongside phase, and
its uint8 display preview.

Intensity mirrors [`phase`](../phase/README.md) but carries **no unit or height
scale**: the `.bin` / `.txt` header is just the geometry (Koala writes the
phase-only bytes as a no-op sentinel), so there is no `PhaseUnit` analogue, no
`target_unit`, and no `bounds_nm`.

## Endpoints by format

| Koala source | Single image | Sequence (list / folder) |
| --- | --- | --- |
| `Float/Bin` (float32, with header) | `load_intensity_bin` / `save_intensity_bin` / `read_intensity_bin_header` | `IntensityBinList` / `IntensityBinFolder` |
| `Float/Txt` (float32, with header) | `load_intensity_txt` / `save_intensity_txt` / `read_intensity_txt_header` | `IntensityTxtList` / `IntensityTxtFolder` |
| `.npy` (float32, **header-less**) | `save_intensity_npy` | `IntensityNpyFolder` |
| `Image/*.tif` (**uint8 preview**) | — | `IntensityTifList` / `IntensityTifFolder` |

- A **`*List`** wraps an explicit, arbitrary list of files; a **`*Folder`**
  auto-discovers `{index:05d}_intensity.<ext>` files under one root and shares a
  single `IntensityBinHeader`.
- `.npy` is header-less, so its only metadata, `pixel_size`, is passed to the
  `IntensityNpyFolder` constructor and shared by every frame.
- The `Image/*.tif` previews are an 8-bit visualization
  (`IntensityImageSequence`), not the quantitative float source
  (`IntensityFloatSequence`); decoding them needs the `iivs-lib[image]` extra.
- The phase and intensity `.bin` formats share the `common.KoalaBinHeader` base;
  `IntensityBinHeader` exposes `width`, `height`, `pixel_size`, and the
  geometry conveniences (`shape`, `field_of_view[_um]`, `pixel_size_um`, ...).

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
