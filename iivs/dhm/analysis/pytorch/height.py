from __future__ import annotations

__all__ = (
    "OpticalHeight",
    "height_to_opd",
    "height_to_phase",
    "opd_to_height",
    "phase_to_height",
)

from typing import TYPE_CHECKING

from kaparoo.utils.optional import unwrap_or_factory
from torch import nn

from iivs.dhm.analysis.height import OpticalHeightConverter
from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.analysis.pytorch.opd import OpticalPathDifference
from iivs.dhm.constants import DEFAULT_REFRACTIVE_DELTA, DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor


class OpticalHeight(nn.Module):
    """Torch `nn.Module` for the phase <-> optical height relation.

    The torch twin of `iivs.dhm.analysis.height.OpticalHeightConverter`, mirroring
    its composition: the module owns an `OpticalPathDifference` submodule
    (`opd_converter`) and binds the cached `height_scale` (nm of height per rad, a
    plain float reused from the NumPy engine, so the physics is shared). Phase is
    the preferred representation: an OPD input enters through phase via the owned
    submodule, whose wavelength then cancels. The `convert_*` / `forward` methods
    are pure scalar multiplies, so they preserve the input tensor's dtype, device,
    and autograd graph.

    Attributes:
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        opd_converter: The owned phase <-> OPD submodule, backing the phase
            path's wavelength and the `opd` entry / exit.
        height_scale: nm of height per rad of phase (a plain float).
    """

    def __init__(
        self,
        *,
        refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
        opd_converter: OpticalPathDifference | None = None,
    ) -> None:
        """Bind the delta and the phase <-> OPD submodule (lab default when None)."""
        super().__init__()

        self.opd_converter = unwrap_or_factory(opd_converter, OpticalPathDifference)

        engine = OpticalHeightConverter(
            refractive_delta=refractive_delta,
            opd_converter=OPDConverter(wavelength=self.opd_converter.wavelength),
        )

        self.refractive_delta = engine.refractive_delta
        self.height_scale = engine.height_scale

    @classmethod
    def from_args(cls, *, wavelength: float, refractive_delta: float) -> Self:
        """Build a module from plain parameters, constructing the OPD submodule."""
        opd_converter = OpticalPathDifference(wavelength=wavelength)
        return cls(refractive_delta=refractive_delta, opd_converter=opd_converter)

    @property
    def wavelength(self) -> float:
        """The owned OPD submodule's wavelength, in m."""
        return self.opd_converter.wavelength

    @property
    def wavelength_nm(self) -> float:
        """The owned OPD submodule's wavelength, in nm."""
        return self.opd_converter.wavelength_nm

    def forward(self, phase: Tensor) -> Tensor:
        """Convert `phase` (rad) to optical height (nm); the `nn.Module` call form."""
        return self.convert_from_phase(phase)

    def convert_from_phase(self, phase: Tensor) -> Tensor:
        """Convert `phase` (rad) to optical height (nm)."""
        return phase * self.height_scale

    def convert_to_phase(self, height: Tensor) -> Tensor:
        """Convert optical `height` (nm) to phase (rad)."""
        return height / self.height_scale

    def convert_from_opd(self, opd: Tensor) -> Tensor:
        """Convert `opd` (nm) to optical height (nm), entering through phase.

        The owned `opd_converter` first maps `opd` back to phase; its wavelength
        cancels, leaving ``opd / refractive_delta``.
        """
        return self.convert_from_phase(self.opd_converter.convert_to_phase(opd))

    def convert_to_opd(self, height: Tensor) -> Tensor:
        """Convert optical `height` (nm) to OPD (nm), exiting through phase."""
        return self.opd_converter.convert_from_phase(self.convert_to_phase(height))


def phase_to_height(
    phase: Tensor,
    *,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
) -> Tensor:
    """Convert phase (rad) to optical height (nm); a one-shot `OpticalHeight`.

    Preserves the input tensor's dtype, device, and autograd graph. For repeated
    use, build an `OpticalHeight` (or read its `height_scale`) once.

    Args:
        phase: Phase image (or batch), in rad.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
    """
    module = OpticalHeight.from_args(
        wavelength=wavelength, refractive_delta=refractive_delta
    )
    return module.convert_from_phase(phase)


def height_to_phase(
    height: Tensor,
    *,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
) -> Tensor:
    """Convert optical height (nm) to phase (rad); a one-shot `OpticalHeight`.

    The inverse of `phase_to_height`; unlike the OPD one-shots, this conversion
    genuinely depends on the wavelength. Preserves the input tensor's dtype,
    device, and autograd graph.

    Args:
        height: Optical height image (or batch), in nm.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
    """
    module = OpticalHeight.from_args(
        wavelength=wavelength, refractive_delta=refractive_delta
    )
    return module.convert_to_phase(height)


def opd_to_height(
    opd: Tensor, *, refractive_delta: float = DEFAULT_REFRACTIVE_DELTA
) -> Tensor:
    """Convert OPD (nm) to optical height (nm); a one-shot `OpticalHeight`.

    Preserves the input tensor's dtype, device, and autograd graph. For repeated
    use, build an `OpticalHeight` once.

    Args:
        opd: OPD image (or batch), in nm.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
    """
    return OpticalHeight(refractive_delta=refractive_delta).convert_from_opd(opd)


def height_to_opd(
    height: Tensor, *, refractive_delta: float = DEFAULT_REFRACTIVE_DELTA
) -> Tensor:
    """Convert optical height (nm) to OPD (nm); a one-shot `OpticalHeight`.

    The inverse of `opd_to_height`; preserves dtype, device, and the autograd graph.

    Args:
        height: Optical height image (or batch), in nm.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
    """
    return OpticalHeight(refractive_delta=refractive_delta).convert_to_opd(height)
