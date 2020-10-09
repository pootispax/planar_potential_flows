#!/usr/bin/env python3

# Main program

import numpy as np
import matplotlib.pyplot as plt
import functions as fc


# Set constants
# x and y must be above 7 and divisible by 4
# The matrix must be square
x = 12
y = 12
h = 1
geometry = 'straight'
# Possible values are :
#                       'straight'
#                       'widening'
#                       'shrinkage'

# Derivation method
method = 'forward'
# Possible values are:
#                       'forward'
#                       'backward'
#                       'centered_2h'
#                       'centered_h'
#                       'second'


fc.build_plot(x, y, h, geometry)
# fc.build_plot(x, y, h, 'widening')
# fc.build_plot(12, 12, h, 'shrinkage')
# fc.build_plot(12, 12, h, 'widening')

