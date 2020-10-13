import sys
from parameters import Nx, Ny, h, geometry, inlet, outlet


# Checks the data
def data_check():
    check_cst = [Nx, Ny, h]
    check_values = [inlet, outlet]

    for i in check_cst:
        if type(i) != int or i < 0 or isinstance(i, complex):
            sys.exit("Nx, Ny and h must be real positive integers")

    if not isinstance(geometry, str):
        sys.exit("Geometry must be a string")

    for i in check_values:
        if i < 0 or isinstance(i, complex):
            sys.exit("Inlet and outlet must be real positive")
