from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.opd import phase_to_opd as np_phase_to_opd

torch = pytest.importorskip("torch")

from iivs.dhm.analysis.pytorch.opd import (  # noqa: E402
    OpticalPathDifference,
    opd_to_phase,
    phase_to_opd,
)


def test_module_forward_matches_convert():
    conv = OpticalPathDifference(wavelength=666e-9)
    phase = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(conv(phase), conv.convert_to_opd(phase))  # forward path


def test_phase_to_opd_matches_numpy():
    phase = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    got = phase_to_opd(phase, wavelength=666e-9)
    expected = np_phase_to_opd(phase.numpy(), wavelength=666e-9)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6)


def test_roundtrip():
    phase = torch.rand(4, 4)
    back = opd_to_phase(phase_to_opd(phase))
    np.testing.assert_allclose(back.numpy(), phase.numpy(), rtol=1e-5)


def test_preserves_grad_and_device():
    phase = torch.ones(2, 2, requires_grad=True)
    opd = phase_to_opd(phase)
    assert opd.requires_grad
    assert opd.device == phase.device
    opd.sum().backward()
    assert phase.grad is not None
