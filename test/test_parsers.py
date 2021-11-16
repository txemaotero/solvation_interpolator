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
