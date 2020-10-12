import sys
from parameters import Nx, Ny, h

# Checks the data
def data_check():
    check = [Nx, Ny, h]
    for i in check:
        if type(i) != int or i < 0:
            sys.exit("Nx, Ny and h must be positive integers")
