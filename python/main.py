# Main program

import numpy as np
import matplotlib.pyplot as plt
import functions as fc


# Set constants
# x and y must be above 7 and divisible by 4
# The matrix must be square
x = 12
y = 12 
h = 10 
geometry = 'straight'
# Possible values are :
#                       'straight'
#                       'widening'
#                       'shinkage'
#                       'widening-shrinkage'
#                       'shrinkage-widening'


fc.plotMatrices(x, y, h, geometry)


