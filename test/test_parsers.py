"""
Set the path to import from parent directory.
"""
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np

# Try imports for good work of the LSP
try:
    from ..parsers import (
        _get_index_to_fit,
        _calc_integral,
        Decoder,
        CoordNumber,
        CoordNumbers,
        Information,
    )
except ImportError:
    from parsers import (
        _get_index_to_fit,
        _calc_integral,
        Decoder,
        CoordNumber,
        CoordNumbers,
        Information,
    )


def test_get_index_to_fit():
    """
    Test the _get_index_to_fit function.
    """
    arange = np.arange(100)
    assert _get_index_to_fit(arange) == 90


def test_calc_integral():
    """
    Test the _calc_integral function.
    """
    x = np.linspace(1, 100, 1000)
    y = -1/x**2
    assert np.isclose(_calc_integral(x, y), -1, rtol=1e-2)

def test_decoder():
    """
    Test the Decoder class.
    """
