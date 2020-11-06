# This file sets the different parameters of the program

# Set the size of the area
Nx = 60
Ny = 60

# Set the size of a cell
h = 3

# Set the Neumann and Dirichlet boundary conditions
vx = 1
phi_ref = 0

# Set the relative density and the initial pressure of the fluid
rho = 50
pressure_init = 5

# Set the choosen geometry, possible values are :
#                                       'straight' (by default)
#                                       'widening'
#                                       'shrinkage'
geometry = 'shrinkage'

# Set the value of the angle in the case of a shrinkage/widening geometry
angle = 20

# If False, the program will check for existing data files, if they exist, it
# will not recompute the data
recompute = True
