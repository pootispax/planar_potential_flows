import os.path as os
import numpy as np
from parameters import *


def data_check():

    if Nx <= 0 or not isinstance(Nx, int):
        raise ValueError('Nx must be a positive integer')

    if Ny <= 0 or not isinstance(Ny, int):
        raise ValueError('Ny must be a positive integer')

    if h <= 0 or np.iscomplex(h):
        raise ValueError('h must be a positive float')

    if vx <= 0 or np.iscomplex(vx):
        raise ValueError('vx must be a positive float')

    if phi_ref <= 0 or np.iscomplex(phi_ref):
        raise ValueError('phi_ref must be a positive float')

    if not isinstance(geometry, str):
        raise TypeError('geometry must be a string')

    if abs(np.radians(angle)) > np.arctan((.5 * Ny - 1) / Nx):
        raise ValueError("Invalid angle value")


def existing_data():

    if not os.exists('dat/G_{}_{}_{}.dat'
                     .format(geometry, Nx, Ny))\
        or not os.exists('dat/phi_{}_{}_{}.dat'
                         .format(geometry, Nx, Ny))\
        or not os.exists('dat/grad_x_{}_{}_{}.dat'
                         .format(geometry, Nx, Ny))\
        or not os.exists('dat/grad_y_{}_{}_{}.dat'
                         .format(geometry, Nx, Ny))\
        or not os.exists('dat/grad_norm_{}_{}_{}.dat'
                         .format(geometry, Nx, Ny))\
            or not os.exists('dat/pressure_{}_{}_{}.dat'
                             .format(geometry, Nx, Ny)):

        return True
