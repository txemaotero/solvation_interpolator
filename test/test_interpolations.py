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
    for _, system in information.items():
        radius = system['ionic_radius']
        radius_int = interpolation.ionic_radii_to_exclusion(radius)[()]
        assert round(radius_int, 5) == system['cnrs'].exclusion_radius

        for key in ('anion', 'cation', 'metal'):
            cnr = system["cnrs"][key].cnr
            interp = interpolation.coordination_number(radius, system['Q'], key)
            interp[interp<1e-10] = 0
            # Admit difference of one bin
            non_zero_cnr = np.nonzero(cnr)[0][0]
            non_zero_interp = np.nonzero(interp)[0][0]
            diff = abs(non_zero_cnr - non_zero_interp)
            assert diff <= 1
            last_non_zero_cnr = np.nonzero(cnr)[0][-1]
            last_non_zero_interp = np.nonzero(interp)[0][-1]
            compare = cnr[non_zero_cnr:last_non_zero_cnr]
            compare_int = interp[non_zero_interp:last_non_zero_interp]
            min_len = min(len(compare), len(compare_int))
            assert np.allclose(compare[:min_len], compare_int[:min_len])

        charge = system["cnrs"].total_charge_distribution
        interp = interpolation.charge_distribution(radius, system['Q'])
        non_zero_charge = np.nonzero(charge)[0][0]
        non_zero_interp = np.nonzero(interp)[0][0]
        diff = abs(non_zero_charge - non_zero_interp)
        assert diff <= 1
        last_non_zero_charge = np.nonzero(charge)[0][-1]
        last_non_zero_interp = np.nonzero(interp)[0][-1]
        compare = charge[non_zero_charge:last_non_zero_charge]
        compare_int = interp[non_zero_interp:last_non_zero_interp]
        min_len = min(len(compare), len(compare_int))
        assert np.allclose(compare[:min_len], compare_int[:min_len])
