#!/usr/bin/env python3

# Main program

import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from matrices import *
from plot import *


data_check()

# Builds the different objects needed
matrices = Matrices()
buildplots = BuildPlots()
buildplots.plot(matrices.G, matrices.phi, matrices.grad, 'green')

# print(matrices.G)
# print(matrices.M)
# print(matrices.cell_coords)
# print(matrices.A)
print(matrices.phi)
print(matrices.grad)

