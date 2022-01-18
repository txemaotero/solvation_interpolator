import argparse

import numpy as np

from parsers import Information
from interpolations import Interpolation


parser = argparse.ArgumentParser(
    description="Interpolate solvation energies from a given file."
)

parser.add_argument("-R", "--R", type=float, help="Radius of the cation to solvate.")
parser.add_argument("-Q", "--Q", type=float, help="Charge of the cation to solvate.")
parser.add_argument(
    "-f", "--f", type=str, help="Path to the file with all the information"
)
parser.add_argument(
    "-rdf", "--rdf", type=str, default=None, help="Path to the output RDF file."
)
parser.add_argument(
    "-coord",
    "--coord",
    type=str,
    default=None,
    help="Path to write the coordination numbers functions.",
)
parser.add_argument(
    "-charge",
    "--charge",
    type=str,
    default=None,
    help="Path to write the charge distribution functions.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    information = Information(args.f)
    interpolation = Interpolation(information)

    print(
        "The interpolated electrostatic work (W) is: ",
        interpolation.electrostatic_work(args.R, args.Q),
    )

    print(
        "The interpolated enthalpy (H) is: ",
        interpolation.enthalpy(args.R, args.Q),
    )

    if args.coord is not None:
        data = [information.distances]
        header = ["Distance"]
        for mol_label in information.mol_labels:
            data.append(interpolation.coordination_number(args.R, args.Q, mol_label))
            header.append(f"cnr({mol_label})")
        header_str = '\t'.join(header)
        np.savetxt(args.coord, np.array(data).T, header=header_str)

    if args.charge is not None:
        data_charge = interpolation.charge_distribution(args.R, args.Q)
        header_charge = information.headers[0]
        np.savetxt(args.charge, data_charge, header=header_charge)
