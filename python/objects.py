# This file contains the functions used to build the matrices that
# will be used throughout the execution of the program


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

# -----------------------------------------------------------------------------
# Builds the needed objects
def build_object(G, Nx, Ny, h, inlet, outlet):
    # Prevents unwanted input for Nx, Ny and h
    if Nx <= 0 or Ny <= 0 or h <= 0 or type(Nx) != int \
    or type(Ny) != int or type(h) != int:
        print("Nx, Ny and h must be positive integers")
        return 0

    M = build_matrix_m(G, Nx, Ny, h)
    cell_coords = build_cell_coords(G, Nx, Ny, h)

    # Builds the elements for the linear solving
    A = build_matrix_a(M, G, cell_coords)
    b = build_vector_b(G, M, cell_coords, inlet, outlet)
    x = solve(A, b)

    phi = matrix_grad(Nx, Ny, h, x, cell_coords)
    plot_contour(Nx, Ny, h, phi, 'green')
    
    
