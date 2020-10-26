# This file sets the different parameters of the program

# Set the size of the area
Nx = 20
Ny = 40

# Set the inlet and outlet (Neumann and Dirichlet boundary conditions)
h = 1
vx = 1
phi_ref = 100

# Set the choosen geometry, possible values are :
#                                       'straight' (by default)
#                                       'widening'
#                                       'shrinkage'
geometry = 'widening'

# Set the value of the angle in the case of a shrinkage/widening geometry
# The angle must be in the interval [1, 25]
angle = 10

inlet = 21
outlet = 12
