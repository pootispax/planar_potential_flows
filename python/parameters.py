# The different values for the program are set here


# x and y must be above 7 and divisible by 4
# The matrix must be square
Nx = 12
Ny = 12
h = 1

# Set the geometry of the problem
# Possible values are :
#                       'straight'
#                       'widening'
#                       'shrinkage'
geometry = 'widening'
inlet = 10
outlet = 10
isopotential_number = 30

# Value for neumann condition
neumann = 0
dirichlet = 0

pressure_init = 1
rho = 1
