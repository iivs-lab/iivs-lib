"""Readers, writers, and sequences for Lyncée Tec Koala acquisition data.

The imaging modalities are each their own subpackage: `phase` and `intensity`
(quantitative float32 ``.bin`` / ``.txt`` / ``.npy``, plus uint8 ``.tif`` previews) and
`hologram` (uint8 ``.raw`` / ``.tif`` / ``.npy``). A `timestamp` module handles
``timestamps.txt``, and a `timelapse` module opens a whole acquisition
(`KoalaTimelapse`, wiring every modality over the standard Koala layout). Within a
modality, each file
format is a codec module exposing a loader, a saver, and lazy sequence types, and a
`convert` helper re-encodes a sequence from one format to another. Blocks shared across
modalities live in `koala`; `constants` holds the lab's default optical parameters.

The ``.bin``, ``.tif`` / ``.raw``, and ``timestamps.txt`` formats are `Lyncée Tec
<https://www.lynceetec.com/>`_'s proprietary Koala formats. The ``.bin`` container
(float32, shared by phase and intensity) was verified against their reference
implementation, `pyKoalaUtils <https://github.com/lynceetec/pyKoalaUtils>`_ (MIT); this
package is an independent reimplementation and contains no code from it.
"""
