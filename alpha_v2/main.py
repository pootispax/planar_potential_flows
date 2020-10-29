#!/usr/bin/env python3

# Main program
import numpy as np
from parameters import Nx, Ny, geometry, recompute
import matrices
import plot
import data_check

data_check.data_check()
matrices = matrices.Matrices()

if recompute or data_check.existing_data():
    print('Computing new data ...')
    matrices.make_data()
else:
    print('Running the program using existing data')

data = matrices.load_data()

plot = plot.Plot()

plot.plot_graphs("potential", data)
plot.plot_graphs("velocity", data)
plot.plot_graphs("streamlines", data)
plot.plot_graphs("pressure", data)
