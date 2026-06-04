# `iivs.dhm.data`

Readers, writers, and lazy sequences for Lyncée Tec Koala acquisition data. Each
imaging modality is its own subpackage; the `timestamp` module and the internal
`common` package round it out.

## Subpackages

| Package | Modality | README |
| --- | --- | --- |
| [`phase`](./phase/README.md) | Quantitative float32 phase + uint8 previews | per-format endpoints, units, `phbounds.txt` |
| [`intensity`](./intensity/README.md) | Quantitative float32 intensity + uint8 previews | the phase twin, without units |
| [`hologram`](./hologram/README.md) | Raw uint8 holograms (`.raw` / `.tif` / `.npy`) | single-file stack + per-frame folders |
| [`common`](./common/README.md) | Cross-modality building blocks (**internal**) | architecture map for contributors |

Each modality splits a format into per-format codec modules over a shared base,
exposing per-format `load_*` / `save_*` / `read_*_header` functions and lazy
`*List` / `*Folder` sequence types. See each README for the endpoints and the
inherited sequence interface (`seq[i]`, `len`, iteration, `get_item` /
`get_meta` / `get_pair`, and the modality-specific accessors).

## The `timestamp` module

Per-frame acquisition timing, as its own `DataSequence` (each item is a
`Timestamp`, its metadata the frame index).

- `Timestamp(elapsed_ms, interval_ms)` — one frame's timing (elapsed since start;
  gap from the previous frame, `0.0` for the first).
- `TimestampsTxtFile(path)` — read a Koala `timestamps.txt` (lines of
  `<index> <time> <date> <elapsed_ms>`, contiguous from 0).
- `TimestampsFixedFPS(*, frame_rate, num_frames)` — synthesize evenly-spaced
  timing from a frame rate.

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

The `.bin`, `.tif` / `.raw`, and `timestamps.txt` formats originate from
[Lyncée Tec](https://www.lynceetec.com/)'s Koala software; the `.bin` layout was
cross-checked against [`pyKoalaUtils`](https://github.com/lynceetec/pyKoalaUtils)
(MIT). This package is an independent reimplementation and contains no code from
it.
