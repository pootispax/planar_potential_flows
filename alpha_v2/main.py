#!/usr/bin/env python3

# Main program
import time
from parameters import recompute
import matrices
import plot
import data_check

t_init = time.time()
data_check.data_check()
matrices = matrices.Matrices()

if recompute or data_check.existing_data():
    data_check.domain_check()
    print('\nComputing new data...')
    matrices.make_data()
else:
    print('Running the program using existing data\n')

data = matrices.load_data()

plot = plot.Plot()

plot.plot_graphs("potential", data)
plot.plot_graphs("velocity", data)
plot.plot_graphs("streamlines", data)
plot.plot_graphs("pressure", data)

t_end = time.time()
print("\nProgram executed in {:.3f} seconds."
      "\nAll the graphs are in the figures/ subfolder"
      .format(t_end - t_init))
