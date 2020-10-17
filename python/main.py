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
