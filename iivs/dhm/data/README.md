# `iivs.dhm.data`

Readers, writers, and lazy sequences for Lyncée Tec Koala acquisition data. Each
imaging modality is its own subpackage; the `timestamp` module and the internal
`koala` package round it out.

## Subpackages

| Package | Modality | README |
| --- | --- | --- |
| [`hologram`](./hologram/README.md) | Raw uint8 holograms (`.raw` / `.tif` / `.npy`) | single-file stack + per-frame folders |
| [`phase`](./phase/README.md) | Quantitative float32 phase + uint8 previews | per-format endpoints, units, `phbounds.txt` |
| [`intensity`](./intensity/README.md) | Quantitative float32 intensity + uint8 previews | the phase twin, without units |
| [`koala`](./koala/README.md) | Cross-modality building blocks (**internal**) | architecture map for contributors |

Each modality splits a format into per-format codec modules over a shared base,
exposing per-format `load_*` / `save_*` / `read_*_header` functions and lazy
`*List` / `*Folder` sequence types. See each README for the endpoints and the
inherited sequence interface (`seq[i]`, `len`, iteration, `get_item` /
`get_meta` / `get_pair`, and the modality-specific accessors).

The readers/writers and sequences are validated end-to-end against a real Koala
acquisition. One non-obvious finding worth recording: the uint8 `Image/*.tif`
previews are normalized differently per modality — **phase** previews share one
**global** `[min, max]` (`phbounds.txt`), so a single `PhaseBounds` reconstructs
them; **intensity** previews are normalized **per frame**, so intensity has no
bounds record and no `Image → Float` path. The `Float` data is the exact source
for both.

## Opening a whole time-lapse (`KoalaTimelapse`)

`KoalaTimelapse(root, *, fps=None)` (in `iivs.dhm.data.timelapse`) **composes** one
acquisition's per-modality groups into a single lazy object over the standard Koala
layout, tolerating absent parts. Holograms, phase, and intensity are **independent**
(any subset may be present); when several are, their frame counts must agree.

| Accessor | Result |
| --- | --- |
| `tl.phase` / `tl.intensity` | a `PhaseGroup` / `IntensityGroup` (always present); each has `.float_bin` / `.float_txt` (the `Float/{Bin,Txt}` sources, which may coexist), `.quantitative` (`.bin`-preferred), and `.previews` (the uint8 `Image` folder) — each `None` when absent |
| `tl.holograms` | the `Holograms/*.raw` stack **or** numbered tif folder, or `None` (raises if a folder holds both) |
| `tl.timestamps` | `timestamps.txt` if present, else `TimestampsFixedFPS` from `fps` (when the frame count is known), else `None` |
| `tl.phase_bounds` | the `phbounds.txt` `PhaseBounds`, or `None` |

Consistency is exposed as flat properties: `frame_counts` (per present source, keyed
`<modality>_<format>`), `counts_agree` (one time-lapse ⇒ every source shares a length),
`has_reconstruction` (phase or intensity present, vs a holograms-only acquisition), and
`has_holograms`. `tl.validate()` returns a `hierarchy.ValidationReport` of the root's
*structure* against `KOALA_TIMELAPSE_TREE`, which **composes** each modality's own
subtree (`phase.PHASE_TREE`, `intensity.INTENSITY_TREE`, `hologram.HOLOGRAM_TREE`) plus
the root `timestamps.txt` / `phbounds.txt`.

`search_timelapses(root, *, require=None, name_filter=None, part_filter=None,
predicate=None, exclude=None, min_depth=1, max_depth=None, ordered=True, fps=None)`
returns the `KoalaTimelapse` list for the acquisition folders found under `root`. It
delegates the walk to `kaparoo`'s `search_dirs` (no manual recursion), so it shares the
same `name_filter` (on the time-lapse folder's own name), `part_filter`, `exclude`,
`min_depth` / `max_depth`, and `ordered`. `require` names the modality folders / files
that must all be present (default: any one modality); `predicate` is a final check on
the built **`KoalaTimelapse`** (not its path).

```python
from iivs.dhm.data.timelapse import KoalaTimelapse, search_timelapses

tl = KoalaTimelapse("scan/2026-01-15_cardiomyocytes")
phase_nm = tl.phase.quantitative      # a PhaseFloatSequence (bin preferred), or None
holo0 = tl.holograms[0]               # first hologram frame
assert tl.counts_agree                # phase / intensity / holograms / timing align
assert tl.validate().ok               # matches the expected layout

# every time-lapse under scans/ that has phase, timing synthesized at 20 fps when absent
for t in search_timelapses("scans/", require=["Phase"], fps=20.0):
    print(t.root.name, t.frame_counts)
```

Each modality package owns its piece — `PhaseGroup` + `PHASE_TREE`, `IntensityGroup` +
`INTENSITY_TREE`, and `open_holograms` + `HOLOGRAM_TREE` — usable on their own;
`timelapse` just composes them.

## The `timestamp` module

Per-frame acquisition timing, as its own `DataSequence` (each item is a
`Timestamp`, its metadata the frame index). The technique-agnostic types were
hoisted to [`iivs.common.data`](../../common/data) so any time-lapse modality can
reuse them; `iivs.dhm.data.timestamp` keeps only the Koala-specific reader.

From `iivs.common.data`:

- `Timestamp(elapsed_ms, interval_ms)` — one frame's timing (elapsed since start;
  gap from the previous frame, `0.0` for the first).
- `TimestampSequence` — the read-only interface (`mean_interval_ms` /
  `mean_frame_rate`) every source implements.
- `TimestampsFixedFPS(*, frame_rate, num_frames)` — synthesize evenly-spaced
  timing from a frame rate.

From `iivs.dhm.data.timestamp`:

- `TimestampsTxtFile(path)` — read a Koala `timestamps.txt` (lines of
  `<index> <time> <date> <elapsed_ms>`, contiguous from 0).

Both sequences are a `kaparoo.data.sequences.DataSequence`, so the inherited
interface applies:

| Call | Result |
| --- | --- |
| `ts[i]` / `len(ts)` / `for t in ts:` | the `Timestamp` at `i` / frame count / lazy iteration |
| `ts.get_item(i)` / `ts.get_meta(i)` / `ts.get_pair(i)` | the `Timestamp` / its frame index / both |
| `ts.timestamps` | all frames as an immutable tuple |
| `ts.mean_interval_ms` | mean gap between consecutive frames (ms) |
| `ts.mean_frame_rate` | mean frame rate (fps) |
| `TimestampsTxtFile(path)` additionally: `ts.path` | the source file |

```python
from iivs.dhm.data.timestamp import TimestampsTxtFile

ts = TimestampsTxtFile("scan/timestamps.txt")
ts.mean_frame_rate            # fps over the acquisition
elapsed = ts[10].elapsed_ms   # ms since acquisition start
```

## Acknowledgement

The `.bin`, `.tif` / `.raw`, and `timestamps.txt` formats are
[Lyncée Tec](https://www.lynceetec.com/)'s proprietary Koala formats; the `.bin`
container (float32, shared by phase and intensity) was cross-checked against
[`pyKoalaUtils`](https://github.com/lynceetec/pyKoalaUtils) (MIT). This package
is an independent reimplementation and contains no code from it.
