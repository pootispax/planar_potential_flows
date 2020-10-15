#!/usr/bin/env python3

# Main program

import numpy as np
import matrices as m
import plot as p
import data_check as dc

dc.data_check()

# Builds the different objects needed
matrices = m.Matrices()
buildplots = p.BuildPlots()
buildplots.plot(matrices.G, matrices.phi_neumann, matrices.grad, 'green')

# print(matrices.G)
# print(matrices.M)
# print(matrices.cell_coords)
# print(matrices.A)
# print(matrices.phi)
np.savetxt('phi.dat', matrices.phi, fmt='%1.2f')
np.savetxt('pressure_x.dat', matrices.pressure[1][0], fmt='%1.1f')
np.savetxt('pressure_y.dat', matrices.pressure[1][1], fmt='%1.1f')

# print(matrices.grad)
# print(matrices.pressure[1])
