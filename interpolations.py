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
        # We need electric field, distances and cnrs
        electric_fields = self.information.electric_fields
        distances = self.information.distances
        for mol_label in self.information.mol_labels:
            cnrs = [data["cnrs"][mol_label].cnr for data in self.information.values()]
            interpolators[mol_label] = interp2d(
                distances, electric_fields, cnrs, kind=self.kind
            )
        return interpolators

    def _get_delta_n_interpolator(self) -> Callable:
        """
        Get the interpolator funciton for the total charge distribution.
        """
        # We need electric field and delta_n
        electric_fields = self.information.electric_fields
        distances = self.information.distances
        tot_charg_dens = [s["cnrs"].total_charge_distribution for s in self.information.values()]
        return interp2d(distances, electric_fields, tot_charg_dens, kind=self.kind)
