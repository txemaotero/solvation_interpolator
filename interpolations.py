"""
Module with the utilities to use the loaded data to interpolate the data.
"""

from typing import Callable, Dict
import numpy as np
from scipy.interpolate import interp2d

from parsers import Information


class Interpolation:
    """
    Class to interpolate the data.
    """

    def __init__(self, information: Information, kind="cubic"):
        self.information = information
        self.kind = kind
        # One interpolator for each mol_type
        self._cnr_interpolators = self._get_cnr_interpolators()
        self._delta_n_interpolator = self._get_delta_n_interpolator()

    def _get_cnr_interpolators(self) -> Dict[str, Callable]:
        """
        Get the interpolator funciton for the coordination numbers of each mol type.
        """
        interpolators = {}
        # We need electric field and cnrs
        electric_fields = self.information.electric_fields
        for mol_label in self.information.mol_labels:
            cnrs, distances = [], None
            for data in self.information.values():
                mol_data = data["cnrs"][mol_label]
                if distances is None:
                    distances = mol_data["distances"]
                cnrs.append(mol_data["cnr"])

            interpolators[mol_label] = interp2d(
                distances, electric_fields, cnrs, kind=self.kind
            )

        return interpolators

    def _get_delta_n_interpolator(self) -> Callable:
        """
        Get the interpolator funciton for the sum of the coordination.
        """
        # We need electric field and delta_n
        electric_fields = self.information.electric_fields
        delta_ns = []

        return interp2d(distances, electric_fields, delta_ns, kind=self.kind)
