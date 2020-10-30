# This file sets the different parameters of the program

# Set the size of the area
Nx = 60
Ny = 60

# Set the inlet and outlet (Neumann and Dirichlet boundary conditions)
h = 1
vx = 1
phi_ref = 1

# Set the choosen geometry, possible values are :
#                                       'straight' (by default)
#                                       'widening'
#                                       'shrinkage'
geometry = 'widening'

# Set the value of the angle in the case of a shrinkage/widening geometry
# The angle must be in the interval [1, 25]
angle = 20

# If False, the program will check for existing data files, if they exist, it
# will not recompute the data
recompute = True
