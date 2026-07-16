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
