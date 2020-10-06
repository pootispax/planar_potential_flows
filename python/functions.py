# All the functions used by the main program are in this file
# Python version used at the start of the project : Python 3.8.5

import numpy as np
import matplotlib.pyplot as plt


# This function builds a squared matrix of size x * y
# h represents the size of a cell
def build_box(x=12, y=12, h=1, geometry='straight'):
    G = np.zeros((x * h, y * h))
    xquarter = x // 4
    yquarter = y // 4

    if geometry == 'straight':
        return build_box_straight(x, y, h, G, xquarter, yquarter)

    elif geometry == 'widening':
        return build_box_widening(x, y, h, G, xquarter, yquarter)

    elif geometry == 'shrinkage':
        return build_box_shrinkage(x, y, h, G, xquarter, yquarter)


# -----------------------------------------------------------------------------
# Build a matrix according to the straight geometry
def build_box_straight(x, y, h, G, xquarter, yquarter):
    for i in range(0, y * h):
        if i < h:
            for j in range(xquarter * h + 1, 3 * xquarter * h - 1):
                G[j, i] = 2

        elif i >=(x - 1) * h:
            for j in range(xquarter * h + 1, 3 * xquarter * h  - 1):
                G[j, i] = 3
    
        else:
            for j in range(xquarter * h + 1, 3 * xquarter * h - 1):
                G[j, i] = 1
    return G

# -----------------------------------------------------------------------------
# Build a matrix according to the widening geometry
def build_box_widening(x, y, h, G, xquarter, yquarter):
    xoffset = 0
    for i in range(0, y * h):
        if i < h:
            for j in range(x * h // 4 + 1, 3 * x * h // 4 - 1):
                G[j, i] = 2
        elif i >= (y - 1) * h:
            for j in range(h + 1, (x - 1) * h - 1):
                G[j, i] = 3
        else:
            if i <= yquarter * h  - 1:
                for j in range(xquarter * h + 1, 3 * xquarter * h - 1):
                    G[j, i] = 1 
            elif i > 2 * yquarter * h - 1:
                for j in range(h + 1, (x - 1) * h - 1):
                    G[j, i] = 1
            elif i >= yquarter * h and i < 2 * yquarter * h:
                for j in range(xquarter * h - xoffset + 1,
                               3 * xquarter * h + xoffset - 1):
                    G[j, i] = 1
                if (i + 1) % h == 0:
                    xoffset += h 
    return G

# -----------------------------------------------------------------------------
# Build a matrix according to the shrinkage geometry
def build_box_shrinkage(x, y, h, G, xquarter, yquarter):
    xoffset = 0
    for i in range(0, y * h):
        if i < h:
            for j in range(h + 1, (x - 1) * h - 1):
                G[j, i] = 2
        elif i >= (y - 1) * h:
            for j in range(xquarter * h + 1, 3 * xquarter * h - 1):
                G[j, i] = 3
        else:
            if i <= yquarter * h - 1:
                for j in range(h + 1, (x - 1) * h - 1):
                    G[j, i] = 1 
            elif i > 2 * yquarter * h - 1:
                for j in range(xquarter * h + 1, 3 * xquarter * h - 1):
                    G[j, i] = 1
            elif i >= yquarter * h  and i < 2 * yquarter * h:
                for j in range(h + xoffset + 1, (x - 1) * h - xoffset - 1):
                    G[j, i] = 1
                if (i + 1) % h == 0:
                    xoffset += h 
    
    return G


# -----------------------------------------------------------------------------
# Builds and plots the borders between walls and fluid cells
def build_walls(x, y, h, G, ax):
    X = np.linspace(0, x * h - 1, x * h)
    Y = np.linspace(0, y * h - 1, y * h)
    XX, YY = np.meshgrid(X, Y)
    
    for i in range(0, y * h - 1):
        for j in range(0, x * h - 1):
            if G[i, j] == 0:
                # Builds a wall under the wall cells
                if G[i + 1, j] != 0:
                    ax.plot(XX[i][j:j + 2] - .5,
                            YY[i][i:i + 2] + .5,
                            ls='-', color='black')

                # Builds a wall above the wall cells
                if G[i - 1, j] != 0: 
                    ax.plot(XX[i][j:j + 2] - .5,
                            YY[i][i:i + 2] - .5,
                            ls='-', color='black')

                # Builds a wall to the right of the wall cells
                if G[i, j + 1] != 0:
                    if i == y * h - 2:
                        ax.plot(YY[j][j:j + 2] + .5,
                                XX[i][i:i + 2] + .5,
                                ls='-', color='black')

                    ax.plot(YY[j][j:j + 2] + .5,
                            XX[i][i:i + 2] - .5 ,
                            ls='-', color='black')
               
                # Builds a wall to the left of the wall cells
                if G[i, j - 1] != 0:
                    if i == y * h - 2:
                        ax.plot(YY[j][j:j + 2] - .5,
                                XX[i][i:i + 2] + .5,
                                ls='-', color='black')
 
                    ax.plot(YY[j][j:j + 2] - .5,
                            XX[i][i:i + 2] - .5,
                            ls='-', color='black')


# -----------------------------------------------------------------------------
# Builds the matrix M containing each cell number
def build_cell_numbers(boxMatrix, x=12, y=12, h=1):
    xsized = x * h
    ysized = y * h
    M = np.zeros((xsized, ysized))
    indent = 1

    for i in range(0, xsized):
        for j in range(0, ysized):
            if boxMatrix[i, j] != 0:
                M[i, j] = indent
                indent += 1

    M = M.astype(int) # Convert each cell of the matrix to an int

    return M


# -----------------------------------------------------------------------------
# Builds the matrix A



# -----------------------------------------------------------------------------
# Plots the matrix G and saves it
def plot_matrices(x=12, y=12, h=1, geometry='straight'):
    # Verification in case the input data is incorrect
    if x <= 0 or y <= 0 or h <= 0 or type(x) != int \
    or type(y) != int or type(h) != int:
        print("x, y and h must be positive integers")
        return 0

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    G = build_box(x, y, h, geometry)
    M = build_cell_numbers(G, x, y)
    # Plots the box
    ax.imshow(G, cmap='coolwarm')
    
    # Plots the walls
    build_walls(x, y, h, G, ax)

    figname = "../figures/{}_x={}_y={}_h={}.pdf".format(geometry, x, y, h)
    plt.savefig(figname)


# -----------------------------------------------------------------------------
# When do we start physics ?!
# It is now or never !
# -----------------------------------------------------------------------------
# Derivative functions in the x direction
def x_derivative(f, x, y, h, method='forward'):
    if method == 'forward':
        return (-1) * (f[x + 1, y] - f[x, y]) / h

    elif method == 'backward':
        return (-1) * (f[x, y] - f[x - 1, y]) / h

    elif method == 'centered_2h':
        return (-1) * (f[x + 1, y] - f[x - 1, y]) / (2 * h)

    elif method == 'centered_h':
        return (-1) * (f[x + (1 / 2), y] - f[x - (1 / 2), y]) / h
    
    elif method =='second':
        return (-1) * (f[x - 1, j] - 2 * f[x, y] + f[x + 1, j]) / h**2


# Derivative functions in the y directions
def y_derivative(f, x, y, h, method='forward'):
    if method == 'forward':
        return (-1) * (f[x, y + 1] - f[x, y]) / h

    elif method == 'backward':
        return (-1) * (f[x, y] - f[x, y - 1]) / h

    elif method == 'centered_2h':
        return (-1) * (f[x, y + 1] - f[x - 1, y]) / (2 * h)

    elif method == 'centered_h':
        return (-1) * (f[x, y + (1 /2)] - f[x, y - (1 / 2)]) / h
    
    elif method =='second':
        return (-1) * (f[x, j - 1] - 2 * f[x, y] + f[x, j + 1]) / h**2


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


