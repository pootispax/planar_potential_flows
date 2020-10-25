#!/usr/bin/env python3

# Main program

import numpy as np
import matrices as m
import plot as p
import data_check as dc


matrices = m.Matrices()
plot = p.Plot()
# np.savetxt('G.dat', matrices.G, fmt='%1.i')
np.savetxt('M.dat', matrices.M, fmt='%4.i')
np.savetxt('cell_coords.dat', matrices.cell_coords, fmt='%4.i')
np.savetxt('b.dat', matrices.b, fmt='%1.i')
np.savetxt('A.dat', matrices.A, fmt='%1.i')
np.savetxt('phi.dat', matrices.phi, fmt='%1.3f')
np.savetxt('grad_x.dat', matrices.grad[0][0], fmt='%1.3f')
np.savetxt('grad_y.dat', matrices.grad[0][1], fmt='%1.3f')
plot.plot_geometry(matrices.G)
