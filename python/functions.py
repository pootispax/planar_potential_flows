# All the functions used by the main program are in this file
# Python version used at the start of the project : Python 3.8.5

import numpy as np
import matplotlib.pyplot as plt

########## FIRST PART ##########
# The program starts by building all needed objects

# This function builds a squared matrix of size Nx * Ny
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
    offset = 0

    for j in range(0, Ny * h):
        if j < h:
            for i in range(Nx * h // 4 + 1, 3 * Nx * h // 4 - 1):
                G[i, j] = 2

        elif j >= (Ny - 1) * h:
            for i in range(h + 1, (Nx - 1) * h - 1):
                G[i, j] = 3

        else:
            if j <= Ny_quarter * h  - 1:
                for i in range(Nx_quarter * h + 1, 3 * Nx_quarter * h - 1):
                    G[i, j] = 1 

            elif j > 2 * Ny_quarter * h - 1:
                for i in range(h + 1, (Nx - 1) * h - 1):
                    G[i, j] = 1

            elif j >= Ny_quarter * h and j < 2 * Ny_quarter * h:
                for i in range(Nx_quarter * h - offset + 1,
                               3 * Nx_quarter * h + offset - 1):
                    G[i, j] = 1
                if (j + 1) % h == 0:
                    offset += h 

    return G


# -----------------------------------------------------------------------------
# Build a matrix according to the shrinkage geometry
def build_box_shrinkage(Nx, Ny, h, G, Nx_quarter, Ny_quarter):
    offset = 0

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
                for i in range(h + offset + 1, (Nx - 1) * h - offset - 1):
                    G[i, j] = 1
                if (j + 1) % h == 0:
                    offset += h 
    
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
def build_matrix_m(G, Nx, Ny, h):
    M = np.zeros((Nx * h, Ny * h))
    count = 1

    for j in range(0, Nx * h):
        for i in range(0, Ny * h):
            if G[i, j] != 0:
                M[i, j] = count
                count += 1

    M = M.astype(int)

    return M


# -----------------------------------------------------------------------------
# Builds array cell 
def build_cell_coords(G, Nx, Ny, h):
    coords_matrix = []
    count = 1

    for j in range(0, Ny * h):
        for i in range(0, Nx * h):
            if G[i, j] != 0:
                coords_matrix.append((count, (i, j)))
                count += 1

    return coords_matrix


# -----------------------------------------------------------------------------
# Builds the matrix A
def build_matrix_a(M, G, cell):
    A = np.zeros((M.max(), M.max()))

    for i in range(0, M.max()):
        row = cell[i][1][0]
        column = cell[i][1][1]

        if G[row, column] == 1:
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

        elif G[row, column] == 2:
            A[i, i] = -1
        elif G[row, column] == 3:
            A[i, i] = 1

    return A.astype(int)


# -----------------------------------------------------------------------------
# Builds vector b
def build_vector_b(G, M, cell, inlet, outlet):
    b = np.zeros((M.max(), 1))

    for i in range(M.max()):
        if G[cell[i][1][0], cell[i][1][1]] == 2:
            b[i, 0] = inlet

        elif G[cell[i][1][0], cell[i][1][1]] == 3:
            b[i, 0] = outlet

    return b.astype(int)


# -----------------------------------------------------------------------------
# Plots the different needed matrixes
def build_plot(Nx=12, Ny=12, h=1, geometry='straight'):
    # Prevents unwanted input for Nx, Ny and h
    if Nx <= 0 or Ny <= 0 or h <= 0 or type(Nx) != int \
    or type(Ny) != int or type(h) != int:
        print("Nx, Ny and h must be positive integers")
        return 0

    # Builds all the needed objects for the base plot
    G = build_box(Nx, Ny, h, geometry)
    M = build_matrix_m(G, Nx, Ny, h)
    cell_coords = build_cell_coords(G, Nx, Ny, h)

    # Starting the plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plots the box
    ax.imshow(G, cmap='coolwarm')

    # Plots the walls
    build_walls(Nx, Ny, h, G, ax)

    # Builds the elements for the linear solving
    A = build_matrix_a(M, G, cell_coords)
    b = build_vector_b(G, M, cell_coords, 5, 5)
    x = solve(A, b)


    grad = matrix_grad(Nx, Ny, h, x, cell_coords)
    plot_contour(Nx, Ny, h, grad, 'green')

    # Saving the figure
    # figname = "../figures/{}_x={}_y={}_h={}.pdf".format(geometry, Nx, Ny, h)
    figname = "../figures/{}_x={}_y={}.pdf".format(geometry, Nx, Ny)
    plt.savefig(figname)

    # Print commands for testing purposes
    # print(G)
    # print(A)
    # print(b)
    # print(x)
    # print(M)
    # print(grad)
    # print(np.gradient(r))
    # print(y_derivative(f, Nx, Ny, h, 'forward'))


    



# -----------------------------------------------------------------------------
# Plot potential contour
def plot_contour(Nx, Ny, h, grad, color):
    X = np.linspace(0, Nx * h - 1, Nx * h)
    Y = np.linspace(0, Ny * h - 1, Ny * h)
    plt.contour(X, Y, grad, colors=color, linewidths=1)


# -----------------------------------------------------------------------------
# When do we start physics ?!
# It is now or never !
# -----------------------------------------------------------------------------
# Solving the linear system of equation
def solve(A, b):
    x = np.linalg.solve(A, b)

    return x


# -----------------------------------------------------------------------------
# Build matrix grad
def matrix_grad(Nx, Ny, h, x, cell):
    grad = np.zeros((Nx * h, Ny * h))
    for i in range(len(cell)):
        x_coords = cell[i][1][0]
        y_coords = cell[i][1][1]
        grad[x_coords, y_coords] = x[i]

    return grad


# -----------------------------------------------------------------------------
# Gradient

# -----------------------------------------------------------------------------
# Derivative in x direction v2
# def nx_derivative():



# -----------------------------------------------------------------------------
# Derivative functions in the x direction
def gradient(x, M, G, cell, Nx, Ny, h, method='forward'):
    grad = []
    
    for i in range(M.max()):
        row = cell[i][1][0]
        column = cell[i][1][1]
        cell_up = cell[i][1][0] - 1
        cell_down = cell[i][1][0] + 1
        cell_left = cell[i][1][1] - 1
        cell_right = cell[i][1][1] + 1

        if G[row, column] == 1:
            if G[cell_up, column] != 0 \
            and G[cell_down, column] != 0 \
            and G[row, cell_left] != 0 \
            and G[row, cell_right] != 0:
                if method == 'forward':
                    temp_x = (-1) * (x[M[cell_down, column], 0] - \
                                        x[M[row, column]]) / h
                    temp_y = (-1) * (x[M[row, cell_down], 0] - \
                                        x[M[row, column]]) / h
                    grad.append([temp_x[0], temp_y[0]])

                elif method == 'backward':
                    temp_x = (-1) * (x[M[row, column], 0] - \
                                        x[M[cell_up, column]]) / h
                    temp_y = (-1) * (x[M[row, column], 0] - \
                                        x[M[row, cell_up]]) / h
                    grad.append([temp_x[0], temp_y[0]])

                elif method == 'centered_2h':
                    temp_x = (-1) * (x[M[cell_down, column], 0] - \
                                        x[M[cell_up, column]]) / (h * 2)
                    temp_y = (-1) * (x[M[row, cell_down], 0] - \
                                        x[M[row, cell_up]]) / (h * 2)
                    grad.append([temp_x[0], temp_y[0]])

                # elif method == 'centered_h':
                #     grad[i, 0] = (-1) * (x[M[cell_down - .5, column], 0] - \
                #                         x[M[cell_up + .5, column]]) / h
    return grad


# -----------------------------------------------------------------------------
# Derivative functions in the y direction
def ny_derivative(x, M, G, cell, Nx, Ny, h, method='forward'):
    grad = np.zeros((len(x), 1))
    
    for i in range(M.max()):
        row = cell[i][1][0]
        column = cell[i][1][1]
        cell_up = cell[i][1][0] - 1
        cell_down = cell[i][1][0] + 1
        cell_left = cell[i][1][1] - 1
        cell_right = cell[i][1][1] + 1

        if G[row, column] == 1:
            if G[cell_up, column] != 0 \
            and G[cell_down, column] != 0 \
            and G[row, cell_left] != 0 \
            and G[row, cell_right] != 0:
                if method == 'forward':
                    grad[i, 0] = (-1) * (x[M[row, cell_down], 0] - \
                                        x[M[row, column]]) / h

                elif method == 'backward':
                    grad[i, 0] = (-1) * (x[M[row, column], 0] - \
                                        x[M[row, cell_up]]) / h


                elif method == 'centered_2h':
                    grad[i, 0] = (-1) * (x[M[row, cell_down], 0] - \
                                        x[M[row, cell_up]]) / (h * 2)

                # elif method == 'centered_h':
                #     grad[i, 0] = (-1) * (x[M[row, cell_down - .5], 0] - \
                #                         x[M[row, cell_up - .5]]) / h

    return grad


# -----------------------------------------------------------------------------
# Laplace equation



# -----------------------------------------------------------------------------
# Neumann's boundary condition



# -----------------------------------------------------------------------------
# Dirichlet's boundary condition
