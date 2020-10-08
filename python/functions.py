# All the functions used by the main program are in this file
# Python version used at the start of the project : Python 3.8.5

import numpy as np
import matplotlib.pyplot as plt

########## FIRST PART ##########
# The program starts by building all needed objects

# This function builds a squared matrix of size x * y
# h represents the size of a cell
def build_box(Nx=12, Ny=12, h=1, geometry='straight'):
    G = np.zeros((Nx * h, Ny * h))
    Nx_quarter = Nx // 4
    Ny_quarter = Ny // 4

    if geometry == 'straight':
        return build_box_straight(Nx, Ny, h, G,
                                  Nx_quarter, Ny_quarter).astype(int)

    elif geometry == 'widening':
        return build_box_widening(Nx, Ny, h, G,
                                  Nx_quarter, Ny_quarter).astype(int)

    elif geometry == 'shrinkage':
        return build_box_shrinkage(Nx, Ny, h, G,
                                   Nx_quarter, Ny_quarter).astype(int)


# -----------------------------------------------------------------------------
# Build a matrix according to the straight geometry
def build_box_straight(Nx, Ny, h, G, Nx_quarter, Ny_quarter):
    for j in range(0, Ny * h):
        if j < h:
            for i in range(Nx_quarter * h + 1, 3 * Nx_quarter * h - 1):
                G[i, j] = 2

        elif j >=(Nx - 1) * h:
            for i in range(Nx_quarter * h + 1, 3 * Nx_quarter * h  - 1):
                G[i, j] = 3
    
        else:
            for i in range(Nx_quarter * h + 1, 3 * Nx_quarter * h - 1):
                G[i, j] = 1
    return G

# -----------------------------------------------------------------------------
# Build a matrix according to the widening geometry
def build_box_widening(Nx, Ny, h, G, Nx_quarter, Ny_quarter):
    Nx_offset = 0
    for j in range(0, Ny * h):
        if j < h:
            for i in range(Nx * h // 4 + 1, 3 * Nx * h // 4 - 1):
                G[i, j] = 2
        elif j >= (Ny - 1) * h:
            for i in range(Nh + 1, (Nx - 1) * h - 1):
                G[i, j] = 3
        else:
            if j <= Ny_quarter * h  - 1:
                for i in range(Nx_quarter * h + 1, 3 * Nx_quarter * h - 1):
                    G[i, j] = 1 
            elif j > 2 * Ny_quarter * h - 1:
                for i in range(h + 1, (Nx - 1) * h - 1):
                    G[i, j] = 1
            elif j >= Ny_quarter * h and j < 2 * Ny_quarter * h:
                for i in range(Nx_quarter * h - Nx_offset + 1,
                               3 * Nx_quarter * h + Nx_offset - 1):
                    G[i, j] = 1
                if (j + 1) % h == 0:
                    Nx_offset += h 
    return G

# -----------------------------------------------------------------------------
# Build a matrix according to the shrinkage geometry
def build_box_shrinkage(Nx, Ny, h, G, Nx_quarter, Ny_quarter):
    Nx_offset = 0
    for j in range(0, Ny * h):
        if j < h:
            for i in range(h + 1, (Nx - 1) * h - 1):
                G[i, j] = 2
        elif j >= (Ny - 1) * h:
            for i in range(Nx_quarter * h + 1, 3 * Nx_quarter * h - 1):
                G[i, j] = 3
        else:
            if j <= Ny_quarter * h - 1:
                for i in range(h + 1, (Nx - 1) * h - 1):
                    G[i, j] = 1 
            elif j > 2 * Ny_quarter * h - 1:
                for i in range(Nx_quarter * h + 1, 3 * Nx_quarter * h - 1):
                    G[i, j] = 1
            elif j >= Ny_quarter * h  and j < 2 * Ny_quarter * h:
                for i in range(h + Nx_offset + 1, (Nx - 1) * h - Nx_offset - 1):
                    G[i, j] = 1
                if (j + 1) % h == 0:
                    Nx_offset += h 
    
    return G


# -----------------------------------------------------------------------------
# Builds and plots the borders between walls and fluid cells
def build_walls(Nx, Ny, h, G, ax):
    X = np.linspace(0, Nx * h - 1, Nx * h)
    Y = np.linspace(0, Ny * h - 1, Ny * h)
    XX, YY = np.meshgrid(X, Y)
    
    for j in range(0, Ny * h - 1):
        for i in range(0, Nx * h - 1):
            if G[i, j] == 0:
                # Builds a wall under the wall cells
                if G[i + 1, j] != 0:
                    if j == Ny * h - 2:
                        ax.plot(XX[j][j:j + 2] + .5,
                                YY[i][i:i + 2] + .5,
                                ls='-', color='black')

                    ax.plot(XX[i][j:j + 2] - .5,
                            YY[i][i:i + 2] + .5,
                            ls='-', color='black')

                # Builds a wall above the wall cells
                if G[i - 1, j] != 0: 
                    if j == Ny * h - 2:
                        ax.plot(XX[j][j:j + 2] + .5,
                                YY[i][i:i + 2] - .5,
                                ls='-', color='black')

                    ax.plot(XX[i][j:j + 2] - .5,
                            YY[i][i:i + 2] - .5,
                            ls='-', color='black')

                # Builds a wall to the right of the wall cells
                if G[i, j + 1] != 0:
                    if j == Ny * h - 2:
                        ax.plot(YY[j][j:j + 2] + .5,
                                XX[i][i:i + 2] + .5,
                                ls='-', color='black')

                    ax.plot(YY[j][j:j + 2] + .5,
                            XX[i][i:i + 2] - .5 ,
                            ls='-', color='black')
               
                # Builds a wall to the left of the wall cells
                if G[i, j - 1] != 0:
                    if j == Ny * h - 2:
                        ax.plot(YY[j][j:j + 2] - .5,
                                XX[i][i:i + 2] + .5,
                                ls='-', color='black')
 
                    ax.plot(YY[j][j:j + 2] - .5,
                            XX[i][i:i + 2] - .5,
                            ls='-', color='black')


# -----------------------------------------------------------------------------
# Builds the matrix M containing each cell number
def build_cell_numbers(boxMatrix, Nx, Ny, h):
    Nx_sized = Nx * h
    Ny_sized = Ny * h
    M = np.zeros((Nx_sized, Ny_sized))
    indent = 1

    for j in range(0, Nx_sized):
        for i in range(0, Ny_sized):
            if boxMatrix[i, j] != 0:
                M[i, j] = indent
                indent += 1

    M = M.astype(int) # Convert each cell of the matrix to an int

    return M


# -----------------------------------------------------------------------------
# Builds array cell 
def cell_coords(G, Nx, Ny, h):
    coords_matrix = []
    indent = 1
    for j in range(0, Ny * h):
        for i in range(0, Nx * h):
            if G[i, j] != 0:
                coords_matrix.append((indent, (i, j)))
                indent += 1

    return coords_matrix

# -----------------------------------------------------------------------------
# Builds the matrix A
def build_matrix_a(M, G, cell):
    A = np.zeros((M.max(), M.max()))

    for i in range(0, M.max()):
        row = cell[i][1][0]
        column = cell[i][1][1]

        if G[row, column] == 1 or G[row, column] == 2:
            cell_up = cell[i][1][0] - 1
            cell_down = cell[i][1][0] + 1
            cell_left = cell[i][1][1] - 1
            cell_right = cell[i][1][1] + 1

            # Checks the walls around the cell
            # Cell above
            if G[cell_up, column] != 0 and cell_up >= 0:
                A[i, i] -= 1
                A[i, M[cell_up, column] - 1] = 1

            # Cell under
            if G[cell_down, column] != 0 and cell_down <= M.max():
                A[i, i] -= 1
                A[i, M[cell_down, column] - 1] = 1
        
            # Cell on the left
            if G[row, cell_left] != 0 and cell_left >= 0:
                A[i, i] -= 1
                A[i, M[row, cell_left] - 1] = 1

            # Cell on the right
            if G[row, cell_right] != 0 and cell_right <= M.max():
                A[i, i] -= 1
                A[i, M[row, cell_right] - 1] = 1

        elif G[row, column] == 3:
            A[i, i] = 1

    return A.astype(int)


# -----------------------------------------------------------------------------
# Builds vector b
def vector_b(G, M, cell, inlet=5, outlet=6):
    b = np.zeros((M.max(), 1))

    for i in range(M.max()):
        if G[cell[i][1][0], cell[i][1][1]] == 2:
            b[i, 0] = inlet

        elif G[cell[i][1][0], cell[i][1][1]] == 3:
            b[i, 0] = outlet
    
    return b.astype(int)


# -----------------------------------------------------------------------------
# Plots the matrix G and saves it
def plot_matrices(Nx=12, Ny=12, h=1, geometry='straight'):
    # Prevents unwanted input for Nx, Ny and h
    if Nx <= 0 or Ny <= 0 or h <= 0 or type(Nx) != int \
    or type(Ny) != int or type(h) != int:
        print("x, y and h must be positive integers")
        return 0

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # Builds all the needed objects
    G = build_box(Nx, Ny, h, geometry)
    M = build_cell_numbers(G, Nx, Ny, h)
    cell_link = cell_coords(G, Nx, Ny, h)
    A = build_matrix_a(M, G, cell_link)
    b = vector_b(G, M, cell_link)
    x_solution = solve(A, b)

    # Print commands for testing purposes
    print(A)
    print(b)
    print(x_solution)
    #print(y_derivative(f, Nx, Ny, h, 'forward'))
    # Plots the box
    ax.imshow(G, cmap='coolwarm')
    
    # Plots the walls
    build_walls(Nx, Ny, h, G, ax)
    
    # Saving the figure
    figname = "../figures/{}_x={}_y={}_h={}.pdf".format(geometry, Nx, Ny, h)
    plt.savefig(figname)



# -----------------------------------------------------------------------------
# When do we start physics ?!
# It is now or never !
# -----------------------------------------------------------------------------
# Solving the linear system of equation
def solve(A, b):
    return np.linalg.solve(A, b)


# -----------------------------------------------------------------------------
# Derivative functions in the x direction
def x_derivative(f, Nx, Ny, h, method='forward'):
    if method == 'forward':
        for j in range(Nx - 1):
            for i in range(Ny - 1):
                r[i] = (-1) * (f[Nx + 1, y] - f[x, y]) / h

    elif method == 'backward':
        for j in range(Nx - 1):
            for i in range(Ny - 1):
                return (-1) * (f[x, y] - f[Nx - 1, y]) / h

    elif method == 'centered_2h':
        for j in range(Nx - 1):
            for i in range(Ny - 1):
                return (-1) * (f[Nx + 1, y] - f[Nx - 1, y]) / (2 * h)

    elif method == 'centered_h':
        for j in range(Nx - 1):
            for i in range(Ny - 1):
                return (-1) * (f[x + (1 / 2), y] - f[x - (1 / 2), y]) / h
    
    elif method =='second':
        for j in range(Nx - 1):
            for i in range(Ny - 1):
                r[i, j] = (-1) * (f[x - 1, j] - 2 * f[x, y] + f[x + 1, j]) / h**2


# Derivative functions in the y directions
def y_derivative(f, x, y, h, method='forward'):
    r = np.zeros((x, y))

    if method == 'forward':
        for j in range(x - 1):
            for i in range(y - 1):
                r[i, j] =  (-1) * (f[i, j + 1] - f[i, j]) / h

    elif method == 'backward':
        for j in range(x - 1):
            for i in range(y - 1):
                 r[i, j] = (-1) * (f[i, j] - f[i, j - 1]) / h

    elif method == 'centered_2h':
        for j in range(x - 1):
            for i in range(y - 1):
                r[i, j] = (-1) * (f[i, j + 1] - f[i - 1, j]) / (2 * h)

    elif method == 'centered_h':
        for j in range(x - 1):
            for i in range(y - 1):
                r[i, j] = (-1) * (f[i, j + (1 /2)] - f[i, j - (1 / 2)]) / h
    
    elif method =='second':
        for j in range(x - 1):
            for i in range(y - 1):
                r[i, j] = (-1) * (f[i, j - 1] - 2 * f[i, j] + f[i, j + 1]) / h**2

    return r


# -----------------------------------------------------------------------------
# Laplace equation

def laplace(x, y):
    return (f[x - 1, y] + f[x + 1, y] + f[x, y - 1] + f[x, y + 1]) / 4

# -----------------------------------------------------------------------------
# Neumann's boundary condition
# THIS FUNCTION AND EVERY FUNCTIONS CALLED IN IT CAN BE SOURCE OF ERROR
def neumann(f, x, y, h, axis):
    if axis == 'x':
        condition = f[x + 1, y] + h * x_derivative(f, x, y, h, 'forward')

    elif axis == 'y':
        condition = f[x, y + 1] + h * y_derivative(f, x, y, h, 'backward')


# -----------------------------------------------------------------------------
# Dirichlet's boundary condition
def dirichlet(ref):
    return ref


# -----------------------------------------------------------------------------
# Build the matrix phi
#def matrix_phi():
    

