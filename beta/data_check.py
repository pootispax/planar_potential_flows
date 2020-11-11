import os
import sys
import numpy as np
from parameters import *
import matrices

def directory_check():

    if not os.path.exists('dat'):
        os.makedirs('dat')

    if not os.path.exists('figures'):
        os.makedirs('figures')

def data_check():

    if nx <= 0 or not isinstance(nx, int):
        raise ValueError('nx must be a positive integer')

    if ny <= 0 or not isinstance(ny, int):
        raise ValueError('ny must be a positive integer')

    if h <= 0 or np.iscomplex(h):
        raise ValueError('h must be a positive float')

    if vx <= 0 or np.iscomplex(vx):
        raise ValueError('vx must be a positive float')

    if phi_ref < 0 or np.iscomplex(phi_ref):
        raise ValueError('phi_ref must be a positive float')

    if not isinstance(geometry, str):
        raise TypeError('geometry must be a string')

    if abs(np.radians(angle)) > np.arctan((.5 * ny - 1) / nx)\
            and (geometry == 'shrinkage' or geometry == 'widening'):
        raise ValueError("Invalid angle value")


def existing_data():

    if not os.path.exists('dat/G_{}_{}_{}.dat'
                     .format(geometry, nx, ny))\
        or not os.path.exists('dat/phi_{}_{}_{}.dat'
                         .format(geometry, nx, ny))\
        or not os.path.exists('dat/grad_x_{}_{}_{}.dat'
                         .format(geometry, nx, ny))\
        or not os.path.exists('dat/grad_y_{}_{}_{}.dat'
                         .format(geometry, nx, ny))\
        or not os.path.exists('dat/grad_norm_{}_{}_{}.dat'
                         .format(geometry, nx, ny))\
            or not os.path.exists('dat/pressure_{}_{}_{}.dat'
                             .format(geometry, nx, ny)):

        return True


def domain_check():

    matrix = matrices.Matrices()
    if matrix.M.max() + 1 > 5000:
        print("You started the program with a rather big domain (more than"
              " 5000 fluid cells).\nThe computation can take some time,"
              " are you sure want to continue (Yes/No) ?")
        answer = input("\n> ")

        if answer == "Yes" or answer == "yes":
            pass
        elif answer == "No" or answer == "no":
            sys.exit("\nOperation cancelled")
        else:
            sys.exit("\nInvalid input, operation cancelled")
