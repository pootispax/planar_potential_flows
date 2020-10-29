#!/usr/bin/env python3

# Main program
import numpy as np
import matrices as m
import plot as p
import data_check as dc

dc.data_check()

matrices = m.Matrices()

data = {'G': np.loadtxt('dat/G.dat', dtype=np.int8),
        'phi': np.loadtxt('dat/phi.dat', dtype=np.float32),
        'grad_x': np.loadtxt('dat/grad_x.dat', dtype=np.float32),
        'grad_y': np.loadtxt('dat/grad_y.dat', dtype=np.float32),
        'grad_norm': np.loadtxt('dat/grad_norm.dat', dtype=np.float32),
        'pressure': np.loadtxt('dat/pressure.dat', dtype=np.float32)}

plot = p.Plot()

plot.plot_graphs("potential", data)
plot.plot_graphs("velocity", data)
plot.plot_graphs("streamlines", data)
plot.plot_graphs("pressure", data)
