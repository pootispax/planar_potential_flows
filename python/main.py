# Main program

import numpy as np
import matplotlib.pyplot as plt
import functions as fc


# Set constants
# x and y must be above 7 and divisible by 4
# The matrix must be square
x = 60
y = 60
h = 1 
geometry = 'straight'
# Possible values are :
#                       'straight'
#                       'widening'
#                       'shinkage'

# Derivation method
method = 'forward'
# Possible values are:
#                       'forward'
#                       'backward'
#                       'centered_2h'
#                       'centered_h'
#                       'second'


#fc.plot_matrices(x, y, h, geometry)
#fc.plot_matrices(x, y, h, 'straight')
#fc.plot_matrices(x, y, h, 'shrinkage')
#fc.plot_matrices(x, y, h, 'widening')

fc.plot_matrices(8, 8, 1, 'straight')
