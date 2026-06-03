"""Readers, writers, and sequences for Lyncée Tec Koala acquisition data.

The phase / intensity ``.bin``, hologram ``.tif`` / ``.raw``, and
``timestamps.txt`` formats originate from
`Lyncée Tec <https://www.lynceetec.com/>`_'s Koala software. The ``.bin``
layout was verified against their reference implementation,
`pyKoalaUtils <https://github.com/lynceetec/pyKoalaUtils>`_ (MIT); this package
is an independent reimplementation and contains no code from it.
"""

from __future__ import annotations
