# All the functions used by the main program are in this file
# Python version used at the start of the project : Python 3.8.5

import numpy as np
import matplotlib.pyplot as plt


# This function builds a squared matrix of size x * y
# h represents the size of a cell (not working yet)
def buildBox(x=12, y=12, h=12, geometry='straight'):
    G = np.zeros((x, y))
    yquarter = y // 4
    xoffset = 0
    if geometry == 'straight':
        for i in range(0, y):
            if i == 0:
                for j in range(x // 4, 3 * x // 4):
                    G[i, j] = 2
            
            elif i == x - 1:
                for j in range(x // 4, 3 * x // 4):
                    G[i, j] = 3

            else:
                for j in range(x // 4, 3 * x // 4):
                    G[i, j] = 1

    elif geometry == 'widening':
        for i in range(0, y):
            if i == 0:
                for j in range(x // 4, 3 * x // 4):
                    G[i, j] = 2
            elif i == y - 1:
                for j in range(1, x - 1):
                    G[i, j] = 3
            else:
                if i <= yquarter:
                    for j in range(x // 4, 3 * x // 4):
                        G[i, j] = 1
                elif i > 2 * yquarter - 1:
                    for j in range(1, x - 1):
                        G[i, j] = 1
                elif i > yquarter and i < 2 * yquarter:
                    for j in range(x // 4 - xoffset, 3 * x // 4 + xoffset):
                        G[i, j] = 1
                    xoffset += 1

    elif geometry == 'shrinkage':
        for i in range(0, y):
            if i == 0:
                for j in range(1, x - 1):
                    G[i, j] = 2
            elif i == y - 1:
                for j in range(x // 4, 3 * x // 4):
                    G[i, j] = 3
            else:
                if i <= yquarter:
                    for j in range(1, x - 1):
                        G[i, j] = 1
                elif i > 2 * yquarter - 1:
                    for j in range(x // 4, 3 * x // 4):
                        G[i, j] = 1
                elif i > yquarter and i < 2 * yquarter:
                    for j in range(1 + xoffset, x - 1 - xoffset):
                        G[i, j] = 1
                    xoffset += 1


# -----------------------------------------------------------------------------
#   This part builds a grid the size of the matrix and plots it
def buildGrid(x=12, y=12):
    X = np.linspace(0, x, x + 1)
    Y = np.linspace(0, y, y + 1)
    XX, YY = np.meshgrid(X, Y)
    return (XX, YY)


# -----------------------------------------------------------------------------
    # The following builds the matrix M containing each cell number
def buildCellNumbers(boxMatrix, x=12, y=12):
    M = np.zeros((x, y))
    indent = 1

    for i in range(0, x):
        for j in range(0, y):
            if boxMatrix[i, j] != 0:
                M[i, j] = indent
                indent += 1

    M = M.astype(int) # Convert each cell of the matrix to an int
    return M


# -----------------------------------------------------------------------------
# The following plots the matrix G and saves it
def plotMatrices(x=12, y=12, h=12, geometry='straight'):
    plt.figure()
    
    G = buildBox(x, y, h, geometry)
    M = buildCellNumbers(G, x, y)
    XX = buildGrid(x, y)[0]
    YY = buildGrid(x, y)[1]

    # Plots the box
    plt.imshow(G, cmap='coolwarm')
    
    # Plots the grid
    plt.plot(XX - .5, YY - .5, ls='-', color='black')
    plt.plot(YY - .5, XX - .5, ls='-', color='black')
    
    # Prints the number of each cell containing fluid
    for i in range(0, x):
        for j in range(0, y):
            if G[i, j] != 0:
                plt.text(x=j, y=i + .05, s=M[i, j],
                        ha='center', va='center', size=h / 2)
    plt.savefig("fig.pdf")

