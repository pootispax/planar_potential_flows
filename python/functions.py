# All the functions used by the main program are in this file
# Python version used at the start of the project : Python 3.8.5

import numpy as np
import matplotlib.pyplot as plt


# This function builds a squared matrix of size x * y
# h represents the size of a cell (not working yet)
def buildBox(x=12, y=12, h=1, geometry='straight'):
    xsized = x * h
    ysized = y * h
    G = np.zeros((xsized, ysized))
    xquarter = x // 4
    yquarter = y // 4
    xoffset = 0

    # Build a matrix according to the straight geometry
    if geometry == 'straight':
        for i in range(0, ysized):
            if i == 0:
                for j in range(xsized // 4, 3 * xsized // 4):
                    G[i, j] = 2
            
            elif i == xsized - 1:
                for j in range(xsized // 4, 3 * xsized // 4):
                    G[i, j] = 3

            else:
                for j in range(xsized // 4, 3 * xsized // 4):
                    G[i, j] = 1

    # Build a matrix according to the widening geometry
    elif geometry == 'widening':
        for i in range(0, y * h):
            if i < h:
                for j in range(x * h // 4, 3 * x * h // 4):
                    G[i, j] = 2
            elif i >= (y - 1) * h:
                for j in range(h, (x - 1) * h):
                    G[i, j] = 3
            else:
                if i <= yquarter * h  - 1:
                    for j in range(xquarter * h, 3 * xquarter * h):
                        G[i, j] = 1 
                elif i > 2 * yquarter * h - 1:
                    for j in range(h, (x - 1) * h):
                        G[i, j] = 1
                elif i >= yquarter * h and i < 2 * yquarter * h:
                    for j in range(xquarter * h - xoffset,
                                   3 * xquarter * h + xoffset):
                        G[i, j] = 1
                    if (i + 1) % h == 0:
                        xoffset += h 

    # Build a matrix according to the shrinkage geometry
    elif geometry == 'shrinkage':
        for i in range(0, y * h):
            if i < h:
                for j in range(h, (x - 1) * h):
                    G[i, j] = 2
            elif i >= (y - 1) * h:
                for j in range(xquarter * h, 3 * xquarter * h):
                    G[i, j] = 3
            else:
                if i <= yquarter * h - 1:
                    for j in range(h, (x - 1) * h):
                        G[i, j] = 1 
                elif i > 2 * yquarter * h - 1:
                    for j in range(xquarter * h, 3 * xquarter * h):
                        G[i, j] = 1
                elif i >= yquarter * h  and i < 2 * yquarter * h:
                    for j in range(h + xoffset, (x - 1) * h - xoffset):
                        G[i, j] = 1
                    if (i + 1) % h == 0:
                        xoffset += h 
    
    return G


# -----------------------------------------------------------------------------
#   This part builds a grid the size of the matrix and plots it
def buildGrid(x=12, y=12, h=1):
    X = np.linspace(0, x, x + 1) * h 
    Y = np.linspace(0, y, y + 1) * h
    XX, YY = np.meshgrid(X, Y)

    return (XX, YY)


# -----------------------------------------------------------------------------
    # The following builds the matrix M containing each cell number
def buildCellNumbers(boxMatrix, x=12, y=12, h=1):
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
# The following plots the matrix G and saves it
def plotMatrices(x=12, y=12, h=1, geometry='straight'):
    plt.figure()
    
    G = buildBox(x, y, h, geometry)
    M = buildCellNumbers(G, x, y)
    XX = buildGrid(x, y, h)[0]
    YY = buildGrid(x, y, h)[1]

    # Plots the box
    plt.imshow(G, cmap='coolwarm')
    
    # Plots the grid
    plt.plot(XX - .5, YY - .5, ls='-', color='black')
    plt.plot(YY - .5, XX - .5, ls='-', color='black')
    
    # Prints the number of each cell containing fluid
#    for i in range(0, x):
#        for j in range(0, y):
#            if G[i, j] != 0:
#                plt.text(x=j, y=i + .05, s=M[i, j],
#                        ha='center', va='center', size=h * 8)

    figname = "../figures/{}_x={}_y={}_h={}.pdf".format(geometry, x, y, h)
    plt.savefig(figname)

