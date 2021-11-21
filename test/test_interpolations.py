"""
Set the path to import from parent directory.
"""
import os
import sys
import inspect
import pytest
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Try imports for good work of the LSP
try:
    from ..interpolations import Interpolation
    from ..parsers import Information
except ImportError:
    from interpolations import Interpolation
    from parsers import Information


@pytest.fixture
def information() -> Information:
    return Information(os.path.join(parentdir, "data/info_ean.json"))


@pytest.fixture
def interpolation(information) -> Interpolation:
    return Interpolation(information)


def test_interpolation(interpolation: Interpolation, information: Information):
    radius = interpolation.ionic_radii_to_exclusion(0.59)[()]
    assert round(radius, 5) == 0.222
    li_cnr = information["Li"]["cnrs"]["anion"].cnr
    li_interp = interpolation.coordination_number(0.59, 1, "anion")
    close = np.isclose(li_cnr, li_interp)
    n_close = sum(close)
    np.savetxt('test.txt', np.array((li_cnr, li_interp, close)).T)
    assert close[:n_close].all()
    assert not li_interp[n_close:].any()
