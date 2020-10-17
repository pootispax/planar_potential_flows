import sys
from parameters import Nx, Ny, h, geometry, inlet, outlet


# Checks the data
def data_check():
    check_cst = [Nx, Ny, h]
    check_values = [inlet, outlet]

    try:
        if type(Nx) != int or Nx < 0 or isinstance(Nx, complex):
            raise ValueError
    except ValueError:
        sys.exit('Nx must be a positive integer')

    try:
        if type(Ny) != int or Ny < 0 or isinstance(Ny, complex):
            raise ValueError
    except ValueError:
        sys.exit('Ny must be a positive integer')

    try:
        if type(h) != int or h < 0 or isinstance(h, complex):
            raise ValueError
    except ValueError:
        sys.exit('h must be a positive integer')

    try:
        if type(inlet) != int or inlet < 0 or isinstance(inlet, complex):
            raise ValueError
    except ValueError:
        sys.exit('inlet must be a positive integer')

    try:
        if type(outlet) != int or outlet < 0 or isinstance(outlet, complex):
            raise ValueError
    except ValueError:
        sys.exit('outlet must be a positive integer')

    try:
        if type(geometry) != str:
            raise ValueError
    except ValueError:
        sys.exit('geometry must be a string, possible values are \n\
                 "straight\n"\
                 "widening\n"\
                 "shrinkage"')
