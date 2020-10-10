# The different values for the program are set here

import sys

# x and y must be above 7 and divisible by 4
# The matrix must be square
Nx = 12
Ny = 12
h = 4

# Set the geometry of the problem
# Possible values are :
#                       'straight'
#                       'widening'
#                       'shrinkage'
geometry = 'widening'
inlet = 10
outlet = 10


# Checks the data
def data_check():
    check = [Nx, Ny, h]
    for i in check:
        if type(i) != int or i < 0:
            sys.exit("Nx, Ny and h must be positive integers")
