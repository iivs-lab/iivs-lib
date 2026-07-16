from __future__ import annotations

from iivs.dhm.data.koala.layout import open_folder


def spy_on_open(monkeypatch):
    """Record every path a `ReconstructionGroup` opens; return the (live) list of them.

    Only the group's own opener is watched, so an empty list means no phase / intensity
    format folder was reached. `open_holograms` binds `open_folder` in its own module
    and is deliberately not counted here.
    """
    real, opened = open_folder, []

    def spy(path, folder):
        opened.append(path)
        return real(path, folder)

    monkeypatch.setattr("iivs.dhm.data.koala.layout.open_folder", spy)
    return opened


def count_reads(monkeypatch, folder_cls):
    """Count `folder_cls`'s file reads; return the (live) `{"decode", "header"}` counts.

    One entry per pass over a file: `_decode` reads the whole thing, `_read_header` only
    its header. `load_with_header` promises to decode once and read no header
    separately, which is a claim about these two numbers and nothing the return value
    can show.
    """
    counts = {"decode": 0, "header": 0}

    for hook, key in (("_decode", "decode"), ("_read_header", "header")):
        real = getattr(folder_cls, hook)

        def spy(self, *args, _real=real, _key=key, **kwargs):
            counts[_key] += 1
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(folder_cls, hook, spy)

    return counts
