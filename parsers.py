"""
Module with the utilities to parse the data from the json file with all the
information needed to build the interpolation functions.
"""
import json
import numpy as np

from typing import Any, Dict, List, TypedDict, Set, Tuple, Union


class CnrItemType(TypedDict):
    """
    TypedDict to store the information of the coordination number section.
    """

    q: float
    fpath: str
    column: int
    header: str
    distances: np.ndarray
    cnr: np.ndarray


class CnrFileItemType(TypedDict):
    q: float
    fpath: Union[str, Tuple[str, int]]


class ItemType(TypedDict):
    """
    TypedDict to store the information of the json file.
    """

    Q: float
    ionic_radius: float
    enthalpy: float
    enthalpy_md: float
    cnrs: Dict[str, CnrItemType]


InfoType = Dict[str, ItemType]


class Decoder(json.JSONDecoder):
    def decode(self, s: Any) -> Any:
        result = super().decode(s)
        return self._decode(result)

    def _decode(self, o: Any) -> Any:
        if isinstance(o, str):
            if o.isnumeric():
                return int(o)
            try:
                return float(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o


class Information:
    """
    Class to manage the iformation needed to build the interpolation.

    The dictionary items is ordered by the Q/ionic_radius**2 value.

    Parameters
    ----------
    filename : str
        The name of the file with the information in json format. This file
        should contain one object per system to use in the interpolation. An
        example of one of this objects would be:

            "label": {
                "Q": 1,
                "ionic_radius": 0.6,
                "enthalpy": -505.71,   # Experimental enthalpy
                "enthalpy_md": -695.7, # Simulation enthalpy
                "cnrs": {              # The coordination numbers for each mol
                    "mol1_label": {
                        "q": -1,  # The charge of the molecule around
                        "fpath": "./path/to/file/with/cnr.txt"
                    },
                    "mol2_label": {
                        "q": 1,
                        # You can spcify the column in the file with the cnr
                        "fpath": ["./path/to/file/with/cnr.txt", 2]
                    },
                    "mol3_label": {
                        "q": 1,
                        "fpath": ["./path/to/file/with/cnr.txt", 3]
                    }
                }
            }

        The enthalpies are optional and just needed if estimations of them are
        desired.

    """

    def __init__(self, filename: str):
        self.filename = filename
        self.data = self.read_file()
        self.data = dict(
            sorted(
                self.data.items(), key=lambda x: x[1]["Q"] / x[1]["ionic_radius"] ** 2
            )
        )
        # Check that all the cnrs have the same mol_labels
        self.mol_labels: Set[str] = set()
        for system in self.data.values():
            labels = set(system["cnrs"].keys())
            if not self.mol_labels:
                self.mol_labels = labels
            else:
                if self.mol_labels != labels:
                    raise ValueError(
                        "The cnrs of the systems should have the same mol_labels"
                    )

    def read_file(self) -> InfoType:
        """
        Read the json file and return the data.

        Returns
        -------
        dict
            The data read from the file.

        """
        with open(self.filename, "r") as f:
            data = json.load(f, cls=Decoder)

        # Check and parse the cnr part
        for label, system in data.items():
            if not all(k in system for k in ("cnrs", "Q", "ionic_radius")):
                raise ValueError(
                    "The system {} must have the keys cnrs, Q and ionic_radius".format(
                        label
                    )
                )
            system["cnrs"] = CoordNumbers(system["cnrs"])
            for mol_label, mol_data in system["cnrs"].items():
                if isinstance(mol_data["fpath"], list):
                    if len(mol_data["fpath"]) != 2:
                        raise ValueError(
                            "The information for the cnr file "
                            "for the molecule {} is not correct".format(mol_label)
                        )
                    mol_data["column"] = mol_data["fpath"][1]
                    mol_data["fpath"] = mol_data["fpath"][0]
                else:
                    mol_data["column"] = 1

                # Check if there is a header in the file
                with open(mol_data["fpath"], "r") as f:
                    first_line = f.readline()
                    if first_line.startswith("#"):
                        mol_data["header"] = first_line.strip()
                    else:
                        mol_data["header"] = None
                # Read data
                mol_data["distances"], mol_data["cnr"] = np.loadtxt(
                    mol_data["fpath"], usecols=[0, mol_data["column"]]
                ).T

        return data

    def __getitem__(self, key: str) -> ItemType:
        """
        Get the information for the system with the given label.

        Parameters
        ----------
        key : str
            The label of the system.

        Returns
        -------
        data : dict
            The information for the system with the given label.

        """
        return self.data[key]

    def values(self) -> List[ItemType]:
        """
        Get the information for all the systems.

        Returns
        -------
        data : list
            The information for all the systems.

        """
        return list(self.data.values())

    @property
    def headers(self) -> List[str]:
        """
        Get the headers of the cnr files.

        Returns
        -------
        headers : list
            The headers of the cnr files.

        """
        headers = []
        for system in self.data.values():
            for mol_data in system["cnrs"].values():
                headers.append(mol_data["header"])
        return headers

    @property
    def electric_fields(self) -> List[float]:
        """
        Get the values of Q/ionic_radius**2.

        Returns
        -------
        electric_fields : list
            The electric fields.

        """
        electric_fields = []
        for system in self.data.values():
            electric_fields.append(system["Q"] / system["ionic_radius"] ** 2)
        return electric_fields


class CoordNumbers:
    """
    Class to manage the coordination numbers.

    The dictionary items is ordered by the q value.

    Parameters
    ----------
    cnrs : dict
        The coordination numbers for each mol.

    """

    def __init__(self, cnrs: Dict[str, CnrFileItemType]):
        self.cnrs = self._validate(cnrs)

    def _validate(self, cnrs: Dict[str, CnrFileItemType]) -> Dict[str, CnrItemType]:
        system = {}
        for mol_label, mol_data in cnrs.items():
            system[mol_label] = aux = {}
            if isinstance(mol_data["fpath"], list):
                if len(mol_data["fpath"]) != 2:
                    raise ValueError(
                        "The information for the cnr file "
                        "for the molecule {} is not correct".format(mol_label)
                    )
                aux["column"] = mol_data["fpath"][1]
                aux["fpath"] = mol_data["fpath"][0]
            else:
                aux["column"] = 1

            # Check if there is a header in the file
            with open(aux["fpath"], "r") as f:
                first_line = f.readline()
                if first_line.startswith("#"):
                    aux["header"] = first_line.strip()
                else:
                    aux["header"] = None
            # Read data
            aux["distances"], aux["cnr"] = np.loadtxt(
                aux["fpath"], usecols=[0, aux["column"]]
            ).T
        return system

    def __getitem__(self, key: str) -> CnrItemType:
        """
        Get the coordination numbers for the molecule with the given label.

        Parameters
        ----------
        key : str
            The label of the molecule.

        Returns
        -------
        cnr : dict
            The coordination numbers for the molecule with the given label.

        """
        return self.cnrs[key]

    def values(self) -> List[CnrItemType]:
        """
        Get the coordination numbers for all the molecules.

        Returns
        -------
        cnr : list
            The coordination numbers for all the molecules.

        """
        return list(self.cnrs.values())


if __name__ == "__main__":
    info = Information("./info_ean.json")
    print(info["Li"])
