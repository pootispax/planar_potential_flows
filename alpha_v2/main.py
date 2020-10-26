#!/usr/bin/env python3

# Main program

import numpy as np
import matrices as m
import plot as p
import data_check as dc


matrices = m.Matrices()
plot = p.Plot()
# np.savetxt('G.dat', matrices.G, fmt='%1.i')
np.savetxt('dat/M.dat', matrices.M, fmt='%4.i')
np.savetxt('dat/cell_coords.dat', matrices.cell_coords, fmt='%4.i')
np.savetxt('dat/b.dat', matrices.b, fmt='%1.i')
np.savetxt('dat/A.dat', matrices.A, fmt='%1.i')
np.savetxt('dat/phi.dat', matrices.phi, fmt='%1.3f')
np.savetxt('dat/grad_x.dat', matrices.grad_own[0], fmt='%1.3f')
np.savetxt('dat/grad_y.dat', matrices.grad_own[1], fmt='%1.3f')
np.savetxt('dat/grad_norm.dat', matrices.grad_own[4], fmt='%1.3f')
np.savetxt('dat/pressure.dat', matrices.pressure, fmt='%1.3f')
plot.plot_geometry(matrices.G)
