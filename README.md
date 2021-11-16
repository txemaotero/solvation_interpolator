# solvation_interpolator
Tools to interpolate the solvation properties of charged spherical particles.

# Logic

Once the interpolator is initialize with the data from the json file you can ask
for different magnitudes for "new" cations with different charges and ionic
radius. The interpolations of the functions such as the coordination numbers and
charge densities are based on the exclusion radius which is obtained
from the coordination numbers so the first thing is done before obtaining the
interpolated function is get an estimation of the exclusion radius from the
ionic one. This is done from another interpolator.

# Ionic radius

The ionic radius for the metal cations are extracted from: "Revised Effective
Ionic Radii and Systematic Studies of Interatomie Distances in Halides and
Chaleogenides" by R.D. Shannon. The data are those from coordination 4 for Li,
Na, K, Mg and 6 for Rb, Cs, Ca, Sr, Ba.

# Units

The distances for the coordination numbers as well as the ionic radius are
supposed to be in nanometers. The electrostatic work is calculated in kJ/mol.

# TODO

- Add options to work in other units.
- Add tests
