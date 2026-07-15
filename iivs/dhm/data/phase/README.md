# `iivs.dhm.data.phase`

Readers, writers, and lazy sequences for Lyncée Tec Koala **phase** data — the
quantitative float32 phase reconstruction and its uint8 display preview.

> Phase is the optical phase shift, in radians by default. The physical height
> it represents is `phase * height_scale` (metres per radian); see
> [Units](#units--calibration).

## Endpoints by format

| Koala source | Single image | Sequence (list / folder) |
| --- | --- | --- |
| `Float/Bin` (float32, with header) | `load_phase_bin` / `save_phase_bin` / `read_phase_bin_header` | `PhaseBinList` / `PhaseBinFolder` |
| `Float/Txt` (float32, with header) | `load_phase_txt` / `save_phase_txt` / `read_phase_txt_header` | `PhaseTxtList` / `PhaseTxtFolder` |
| `.npy` (float32, **header-less**) | `load_phase_npy` / `save_phase_npy` | `PhaseNpyFolder` |
| `Image/*.tif` (**uint8 preview**, not quantitative) | — | `PhaseTifList` / `PhaseTifFolder` |

- A **`*List`** wraps an explicit, arbitrary list of files (any location, no
  naming rule); each file is read independently.
- A **`*Folder`** auto-discovers `{index:05d}_phase.<ext>` files under one root,
  shares a single acquisition header, and is the same-shape special case of the
  list.
- `Bin` / `Txt` carry the full header (`PhaseBinHeader`); `.npy` carries none, so
  its metadata (`pixel_size`, `unit`, `height_scale`) is passed to the
  `PhaseNpyFolder` constructor and shared by every frame.
- The `Image/*.tif` previews are an 8-bit visualization (`PhaseImageSequence`),
  **not** a substitute for the quantitative float source (`PhaseFloatSequence`).
  The LZW-compressed previews decode via the core `imagecodecs` dependency.

### Format-agnostic entry points

When the format is only known at runtime (from a path's suffix), these pick the
right symbol for you, over `.bin` / `.txt` / `.npy`:

| Entry point | Picks |
| --- | --- |
| `load_phase(path, *, return_header=False)` | `load_phase_{bin,txt,npy}` — the image, or `(image, header)` when `return_header` (with `header` `None` for the header-less `.npy`) |
| `read_phase_header(path)` | `read_phase_{bin,txt}_header` (**`.npy` excluded** — header-less, **raises**; unlike `load_phase`'s optional `None`) |
| `save_phase(path, data, ...)` | `save_phase_{bin,txt,npy}` (`.npy` ignores the header metadata, with a warning) |
| `phase_list(files)` | `Phase{Bin,Txt}List` by the files' shared extension |
| `phase_folder(root)` | `Phase{Bin,Txt,Npy}Folder` by the folder's contents (an `.npy` folder needs `pixel_size` + `unit`) |

`phase_folder` discovers the format with `kaparoo`'s `search_files` + a
`Regex` filter. If a folder holds more than one format it raises by default; pass
`prefer` (a format or a priority sequence, e.g. `prefer=("bin", "txt")`) to pick
one — modeled on `kaparoo.filesystem.hierarchy.Exclusive(on_conflict="priority")`.

## Units & calibration

- `PhaseUnit` — `UNKNOWN` / `RADIANS` / `METERS` are stored on disk;
  `NANOMETERS` is a code-only convenience (saving converts it to `METERS`).
- `PhaseBinHeader` — `width`, `height`, `pixel_size` (m), `height_scale`
  (m per rad), `unit`, with conveniences `shape`, `pixel_count`,
  `field_of_view[_um]`, `pixel_size_um`, `height_scale_nm`.
- `convert_phase_unit(data, *, source, target, height_scale)` — rescale an image
  between units; `RADIANS <-> METERS` uses `height_scale`, `METERS <->
  NANOMETERS` the fixed `1e9`.
- A float list/folder applies a **`target_unit`** on load: pass it to the
  constructor to get every frame back in one unit (`None` keeps each file's
  stored unit).

## The sequence interface

Every phase sequence is a `kaparoo.data.sequences.DataSequence`. Most of what you
call on a `PhaseBinFolder` / `PhaseBinList` is **inherited** and so does not
appear in this package's source — it is listed here so you do not have to chase
base classes.

**All phase sequences** (list, folder, `.npy`, preview):

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

**Float sources** (`Bin` / `Txt` / `Npy`, i.e. `PhaseFloatSequence`) add:

| Call | Result |
| --- | --- |
| `seq.get_header(i)` | the header of file `i`, **without** decoding pixels |
| `seq.load_with_header(i)` | `(image, header)` in a single read |
| `seq.target_unit` | the unit images are returned in (`None` = each file's own) |
| `seq.value_range(index=None, unit=None)` | the `(min, max)` of the values — global (cached), or of frame `index`; `unit=None` follows `target_unit`, or pass a `PhaseUnit` for a fixed unit. Non-finite values are ignored |

**Lists** (`*List`) add `seq.get_file(i)` (the `Path` at `i`) and `seq.files`
(an immutable snapshot of all paths).

**Folders** (`*Folder`) additionally expose:

| Call | Result |
| --- | --- |
| `seq.root` | the scanned directory |
| `seq.header` | the one shared acquisition header |
| `seq.frame_shape` | `(height, width)` shared by every frame |
| `seq.validate(level=...)` | re-check names / headers / data (`"names"`, `"headers"`, `"data"`) |

> Items are fresh, writable copies — hand them straight to `torch.from_numpy`.
> The sequences pickle to just their path(s), so they are cheap to send to
> worker processes (e.g. a PyTorch `DataLoader`).

## Display bounds (`phbounds.txt`)

`PhaseBounds(min_nm, max_nm)` is the record Koala uses to map the float phase
onto the 0–255 previews. Verified against a real acquisition, Koala normalizes
the phase previews **globally** — one `[min, max]` (from `phbounds.txt`) spans
the whole acquisition, *not* per frame — so a single `PhaseBounds` is the right
unit, and `decode_preview` / `encode_preview` match Koala's previews to within
one 8-bit code. (Intensity previews differ: they are normalized per frame, which
is why `intensity` has no bounds record — see its README.)

- `PhaseBounds.from_file(path)` / `bounds.to_file(path)` — read / write a
  `phbounds.txt` (a `[nm]` tag then `min max`). `read_phbounds` / `write_phbounds`
  are the free-function aliases.
- `float_seq.value_range(unit=NANOMETERS)` recomputes the global `(min, max)`
  straight from a quantitative source, so the previews are never the authoritative
  value; wrap it as `PhaseBounds(*float_seq.value_range(unit=NANOMETERS))` for a
  display-bounds record.
- `bounds.decode_preview(u8)` maps a uint8 `Image/*.tif` preview back toward
  phase in nm (lossy — 8-bit quantized, step `(max−min)/255`); `bounds.encode_preview(nm)`
  is the forward render (`[min, max]`→`0–255`, clamped) that mirrors Koala. Pair
  a preview with a `PhaseBounds` from disk or one built from the `Float` twin's
  `value_range(unit=NANOMETERS)`: `bounds.decode_preview(preview_seq[i])`.
- Whole-sequence twins of that map (lazy — each frame is converted on access):
  `float_seq.to_image(bounds=None)` returns a uint8 `PhaseImageSequence`
  (frames are put in nm via their header first, so `target_unit` is irrelevant;
  `None` derives `bounds` from `value_range(unit=NANOMETERS)`), and
  `image_seq.to_float(bounds, *, target_unit=NANOMETERS, height_scale=…)`
  returns a float32 `PhaseFloatSequence` **reconstruction** — 8-bit-quantized,
  *not* the quantitative `Float` source. `target_unit` picks the output unit
  (NANOMETERS / METERS need no scale; RADIANS needs `height_scale`, or
  `wavelength` + `refractive_delta`). The reconstruction view is a `kaparoo`
  `TransformedSequence`; each view exposes `.source` / `.bounds`.

## Time-lapse composition (`PhaseGroup`)

`PhaseGroup(phase_dir)` opens one acquisition's `Phase/` folder: `.bin_folder` /
`.txt_folder` (the `Float/{Bin,Txt}` sources, which may coexist), `.quantitative`
(`.bin`-preferred), `.tif_folder` (the uint8 `Image` preview), the shared `.num_frames`
/ `.frame_shape`, and the `.is_consistent` (tolerant) / `.is_usable` (has
quantitative data and is consistent) cross-format checks. The
folder's layout is the `PHASE_TREE` `hierarchy` spec, and the format-specific searches
`search_phase_bin_folders(root, ...)` / `search_phase_txt_folders` /
`search_phase_tif_folders` return that one folder (a `PhaseBinFolder` /
`PhaseTxtFolder` / `PhaseTifFolder`) for every time-lapse under `root` that has it (via
`kaparoo`'s `search_dirs`, with `predicate` a check on the opened folder). These are what
[`iivs.dhm.data.timelapse`](../README.md#opening-a-whole-time-lapse-koalatimelapse)'s
`KoalaTimelapse` composes for the phase modality.

## Examples

```python
from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit

acq = PhaseBinFolder("scan/Phase/Float/Bin", target_unit=PhaseUnit.NANOMETERS)

acq.header.field_of_view_um   # (height_um, width_um) for the acquisition
acq.frame_shape               # (H, W)
len(acq)                      # frame count

height_nm = acq[0]            # first frame, already in nm (target_unit)
img, hdr = acq.load_with_header(0)   # image + its header in one read
lo, hi = acq.value_range(unit=PhaseUnit.NANOMETERS)  # global (min, max) in nm

for frame in acq:             # lazy iteration
    ...

# Re-encode the acquisition to the text format (new numbered folder):
from iivs.dhm.data.phase import convert_phase_folder
convert_phase_folder("scan/Phase/Float/Txt", acq, ext="txt")

# Export a composed / transformed sequence (no folder header) to a new folder.
# `convert_phase_*` need a file-backed source; `save_phase_folder` takes any
# image sequence plus explicit metadata, so it works with kaparoo composers:
from kaparoo.data.sequences import ConcatSequence
from iivs.dhm.data.phase import save_phase_folder
combined = ConcatSequence(PhaseBinFolder("run1/.../Bin"), PhaseBinFolder("run2/.../Bin"))
save_phase_folder("merged/Bin", combined, ext="bin", pixel_size=2.84e-7, height_scale=2.1e-7)
```
