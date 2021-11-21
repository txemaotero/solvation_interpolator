"""
Set the path to import from parent directory.
"""
import os
import sys
import inspect
import json
import pytest

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
        CnrFileItemType,
    )
except ImportError:
    from parsers import (
        _get_index_to_fit,
        _calc_integral,
        Decoder,
        CoordNumber,
        CoordNumbers,
        Information,
        CnrFileItemType,
    )


@pytest.fixture
def json_path(tmp_path) -> str:
    path = tmp_path / "test.json"
    with open(os.path.join(currentdir, "./info_test.json")) as f:
        fcontent = f.read()
    fcontent = fcontent.replace("{path}", os.path.join(parentdir, "data"))
    path.write_text(fcontent)
    return path


@pytest.fixture
def coord_number() -> CoordNumber:
    """
    Fixture for CoordNumber class.
    """
    distances = np.linspace(0, 2, 100)
    return CoordNumber(
        fpath=os.path.join(parentdir, "data/EAN/cnr_BA_anion_cation_metal.txt"),
        column=1,
        header="Test header",
        charge=2,
        distances=distances,
        cnr=np.concatenate((np.zeros(10), np.arange(90))),
    )


@pytest.fixture
def coord_numbers_info(json_path: str) -> CnrFileItemType:
    """
    Fixture for CoordNumbers class.
    """
    data = json.load(
        open(json_path, "r"),
        cls=Decoder,
    )
    return data["Li"]["cnrs"]


@pytest.fixture
def information(json_path: str) -> Information:
    """
    Fixture for Information class.
    """
    return Information(json_path)


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
    y = -1 / x ** 2
    assert np.isclose(_calc_integral(x, y), -1, rtol=1e-2)


def test_decoder():
    """
    Test the Decoder class.
    """
    test_json = """
    {
        "coord_numbers": [
            {
                "test": "1.5",
                "information": {
                    "name": "test",
                    "value": "1"
                }
            }
        ]
    }
    """
    parsed_json = json.loads(test_json, cls=Decoder)
    assert parsed_json["coord_numbers"][0]["test"] == 1.5
    assert parsed_json["coord_numbers"][0]["information"]["name"] == "test"
    assert parsed_json["coord_numbers"][0]["information"]["value"] == 1


def test_coord_number(coord_number):
    """
    Test the CoordNumber class.
    """
    chr_dis = coord_number.charge_distribution
    assert len(coord_number.distances) == 100
    assert np.isclose(chr_dis[0], 0.0, rtol=1e-2)
    assert np.isclose(chr_dis[-1], 89 * 2, rtol=1e-2)
    distance = np.linspace(0, 2, 100)[11]
    assert coord_number.exclusion_radius == distance
    assert coord_number.charge == 2


def test_coord_numbers(coord_numbers_info):
    """
    Test the CoordNumbers class.
    """
    coord_numbers = CoordNumbers("Li", coord_numbers_info)
    assert set(coord_numbers.keys()) == {"anion", "cation", "metal"}
    assert isinstance(coord_numbers["anion"], CoordNumber)
    assert coord_numbers["anion"].charge == -1
    assert coord_numbers["cation"].charge == 1
    assert coord_numbers["metal"].charge == 1
    tot_char = coord_numbers.total_charge_distribution
    assert isinstance(tot_char, np.ndarray)
    assert tot_char.shape == (len(coord_numbers.distances),)
    assert len(coord_numbers.distances) == 1250


def test_information(information: Information):
    """
    Test the Information class.
    """
    assert set(information.keys()) == {"Li", "Na"}
    assert isinstance(information["Li"], dict)
    assert set(information["Li"].keys()) == {
        "cnrs",
        "Q",
        "enthalpy",
        "enthalpy_md",
        "ionic_radius",
        "cnrs",
    }
    assert isinstance(information["Li"]["cnrs"], CoordNumbers)
    assert information["Li"]["Q"] == 1
    assert information["Li"]["enthalpy"] == -505.71
    assert information["Li"]["enthalpy_md"] == -695.7
    assert information["Li"]["ionic_radius"] == 0.59
    assert isinstance(information.distances, np.ndarray)
    assert len(information.distances) == len(information["Li"]["cnrs"]["anion"].cnr)
    excl_rad = {k: v for k, v in zip(information.keys(), information.exclusion_radii)}
    compare = {"Li": 0.222, "Na": 0.266}
    assert excl_rad == compare
    assert round(information["Li"]["cnrs"].electrostatic_work) == -818
    assert round(information["Na"]["cnrs"].electrostatic_work) == -674
