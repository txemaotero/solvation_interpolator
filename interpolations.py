"""
Module with the utilities to use the loaded data to interpolate the data.
"""

from typing import Callable, Dict
import numpy as np
from scipy.interpolate import interp2d, interp1d

from parsers import Information, _calc_integral, FACTOR_ENERGY


class Interpolation:
    """
    Class to interpolate the data.
    """

    def __init__(self, information: Information, kind: str = "cubic"):
        self.information = information
        self.kind = kind
        # Preload distances
        self._distances = self.information.distances
        # One interpolator for each mol_type for cnrs
        self._cnr_interpolators = self._get_cnr_interpolators()
        # Charge interpolator
        self._charge_interpolator = self._get_charge_interpolator()
        # Interpolator for the ionic radii to the exclusion radius
        self.ionic_radii_to_exclusion = interp1d(
            self.information.ionic_radii,
            self.information.exclusion_radii,
            kind=self.kind,
            fill_value="extrapolate",
        )
        # Interpolator for the electrostatic work to enthalpy
        self.electrostatic_work_to_enthalpy = interp1d(
            self.information.electrostatic_works,
            self.information.enthalpies,
            kind=self.kind,
            fill_value="extrapolate",
        )

    def _get_cnr_interpolators(self) -> Dict[str, Callable[..., np.ndarray]]:
        """
        Get the interpolator funciton for the coordination numbers of each mol type.
        """
        interpolators = {}
        # We need electric field, distances and cnrs
        electric_fields = self.information.electric_fields
        distances = self._distances
        for mol_label in self.information.mol_labels:
            # Get the cnrs and shift them to the exclusion_radius
            cnrs = []
            for data in self.information.values():
                cnr = data["cnrs"][mol_label].cnr
                excl_radius = data["cnrs"].exclusion_radius
                cnrs.append(cnr[distances > excl_radius])
            min_len = min([len(c) for c in cnrs])
            cnrs = [c[:min_len] for c in cnrs]
            aux_distances = distances[:min_len]

            interpolators[mol_label] = interp2d(
                aux_distances,
                electric_fields,
                cnrs,
                kind=self.kind,
                fill_value="extrapolate",
            )
        return interpolators

    def _get_charge_interpolator(self) -> Callable[..., np.ndarray]:
        """
        Get the interpolator funciton for the total charge distribution.
        """
        # We need electric field and delta_n
        electric_fields = self.information.electric_fields
        tot_charg_dens = []
        for data in self.information.values():
            tot_charg = data["cnrs"].total_charge_distribution
            exclusion_radius = data["cnrs"].exclusion_radius
            tot_charg_dens.append(tot_charg[self._distances > exclusion_radius])
        min_len = min([len(c) for c in tot_charg_dens])
        tot_charg_dens = [c[:min_len] for c in tot_charg_dens]
        aux_distances = self._distances[:min_len]
        return interp2d(
            aux_distances,
            electric_fields,
            tot_charg_dens,
            kind=self.kind,
            fill_value="extrapolate",
        )

    def _shift_back(self, data: np.ndarray, exclusion_radius: float) -> np.ndarray:
        """
        Shift the charge distribution back to the exclusion radius.
        """
        # Get the distances
        distances = self._distances
        # Get the index of the first point after the exclusion radius
        index = np.where(distances > exclusion_radius)[0][0]
        # Shift the distribution and check the len to match distances
        new_data = np.concatenate((np.zeros(index), data))[: len(self._distances)]
        # Shorter length means we need to concatenate the last value to the end
        if len(new_data) < len(self._distances):
            diff = len(self._distances) - len(new_data)
            new_data = np.concatenate((new_data, [new_data[-1]] * diff))
        return new_data

    def charge_distribution(self, ionic_radius: float, charge: float) -> np.ndarray:
        """
        Get the charge distribution for a given ionic radius and charge.

        Parameters
        ----------
        ionic_radius : float
            The ionic radius of the central cation.
        charge : float
            The charge of the central cation.

        Returns
        -------
        np.ndarray
            The charge distribution around the central cation.
        """
        # Get the interpolator
        interpolator = self._charge_interpolator
        exclusion_radius = self.ionic_radii_to_exclusion(ionic_radius)
        # Get the electric field
        e_field = charge / exclusion_radius ** 2
        charge_distr = interpolator(self._distances, e_field)
        # Shift back the charge to the exclusion radius
        return self._shift_back(charge_distr, exclusion_radius)

    def coordination_number(
        self, ionic_radius: float, charge: float, mol_label: str
    ) -> np.ndarray:
        """
        Get the coordination number for a given ionic radius, charge and mol_type.

        Parameters
        ----------
        ionic_radius : float
            The ionic radius of the central cation.
        charge : float
            The charge of the central cation.
        mol_label : str
            The mol_label of molecules around to compute the coordination number.

        Returns
        -------
        np.ndarray
            The coordination number distribution around the central cation of
            the given mol_label.
        """
        # Get the interpolator
        interpolator = self._cnr_interpolators[mol_label]
        exclusion_radius = self.ionic_radii_to_exclusion(ionic_radius)
        # Get the electric field
        e_field = charge / exclusion_radius ** 2
        cnr = interpolator(self._distances, e_field)
        return self._shift_back(cnr, exclusion_radius)

    def electrostatic_work(self, ionic_radius: float, charge: float) -> float:
        """
        Get the electrostatic work for a given ionic radius and charge.

        Interpolates the chargre distribution and calculates the integral.

        Parameters
        ----------
        ionic_radius : float
            The ionic radius of the central cation.
        charge : float
            The charge of the central cation.

        Returns
        -------
        float
            The electrostatic work.

        """
        # Get the interpolator
        interpolator = self._charge_interpolator
        exclusion_radius = self.ionic_radii_to_exclusion(ionic_radius)
        # Get the electric field
        e_field = charge / exclusion_radius ** 2
        # Get the charge distribution
        charge_distr = interpolator(self._distances, e_field)
        # Shift back the charge to the exclusion radius
        charge_distr = self._shift_back(charge_distr, exclusion_radius)
        # Get the electrostatic work
        return FACTOR_ENERGY * _calc_integral(
            self._distances, charge_distr / self._distances ** 2
        )

    def enthalpy(self, ionic_radius: float, charge: float) -> float:
        """
        Get the enthalpy for a given ionic radius and charge.

        Interpolates the electrostatic work and uses it to interpolate the
        enthalpy.

        Parameters
        ----------
        ionic_radius : float
            The ionic radius of the central cation.
        charge : float
            The charge of the central cation.

        Returns
        -------
        float
            The interpolated enthalpy.

        """
        # TODO: check if the work-enthalpy relation is the best for interpolate.
        elec_work = self.electrostatic_work(ionic_radius, charge)
        return self.electrostatic_work_to_enthalpy(elec_work)

