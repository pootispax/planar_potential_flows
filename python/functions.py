# All the functions used by the main program are in this file
# Python version used at the start of the project : Python 3.8.5

import numpy as np
import matplotlib.pyplot as plt


# This function builds a squared matrix of size x * y
# h represents the size of a cell
def buildBox(x=12, y=12, h=1, geometry='straight'):
    G = np.zeros((x * h, y * h))
    xquarter = x // 4
    yquarter = y // 4

    if geometry == 'straight':
        return buildBoxStraight(x, y, h, G, xquarter, yquarter)

    elif geometry == 'widening':
        return buildBoxWidening(x, y, h, G, xquarter, yquarter)

    elif geometry == 'shrinkage':
        return buildBoxShrinkage(x, y, h, G, xquarter, yquarter)


# -----------------------------------------------------------------------------
# Build a matrix according to the straight geometry
def buildBoxStraight(x, y, h, G, xquarter, yquarter):
    for i in range(0, y * h):
        if i == 0:
            for j in range(xquarter * h, 3 * xquarter * h):
                G[i, j] = 2
            
        elif i == x * h - 1:
            for j in range(xquarter * h, 3 * xquarter * h):
                G[i, j] = 3
    
        else:
            for j in range(xquarter * h, 3 * xquarter * h):
                G[i, j] = 1
    return G

# -----------------------------------------------------------------------------
# Build a matrix according to the widening geometry
def buildBoxWidening(x, y, h, G, xquarter, yquarter):
    xoffset = 0
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
    return G

# -----------------------------------------------------------------------------
# Build a matrix according to the shrinkage geometry
def buildBoxShrinkage(x, y, h, G, xquarter, yquarter):
    xoffset = 0
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


def buildWalls(x, y, h, G, ax):
    X = np.linspace(0, x * h - 1, x * h)
    Y = np.linspace(0, y * h - 1, y * h)

    XX, YY = np.meshgrid(X, Y)

    for i in range(0, y * h):
        for j in range(0, x * h):
            if G[i, j] == 0:
                # Builds a wall on the bottom of the wall cells
                if i == y * h - 1 or j == x * h - 1:
                    pass

                elif G[i + 1, j] != 0:
                    ax.plot(XX[i][j:j + 2] - .5,
                            YY[i][i:i + 2] + .5,
                            ls='-', color='black')
                
                # Builds a wall on the top of the wall cells
                if i == y * h - 1 or j == x * h - 1:
                    pass
                
                elif G[i - 1, j] != 0: 
                    ax.plot(XX[i][j:j + 2] - .5,
                            YY[i][i:i + 2] - .5,
                            ls='-', color='black')
                
                # Builds a wall on the right of the wall cells
                if i == y * h - 1 or j == x * h - 1:
                    pass

                elif G[i, j + 1] != 0:
                    ax.plot(YY[j][j:j + 2] + .5,
                            XX[i][i:i + 2] - .5 ,
                            ls='-', color='black')
               
                # Builds a wall on the left of the wall cells
                if G[i, j - 1] != 0:
                    if i == y * h - 1:
                        ax.plot(YY[j][j - 2:j] - .5,
                                XX[i][i - 2:i] - .5,
                                ls='-', color='black')
                        print("oui")

                    else:
                        ax.plot(YY[j][j:j + 2] - .5,
                                XX[i][i:i + 2] - .5,
                                ls='-', color='black')


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
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    
    G = buildBox(x, y, h, geometry)
    M = buildCellNumbers(G, x, y)
    XX = buildGrid(x, y, h)[0]
    YY = buildGrid(x, y, h)[1]

    # Plots the box
    ax.imshow(G, cmap='coolwarm')
    
    # Plots the grid
    #plt.plot(XX - .5, YY - .5, ls='-', color='black')
    #plt.plot(YY - .5, XX - .5, ls='-', color='black')
    
    # Plots the walls
    buildWalls(x, y, h, G, ax)
    # Prints the number of each cell containing fluid
#    for i in range(0, x):
#        for j in range(0, y):
#            if G[i, j] != 0:
#                plt.text(x=j, y=i + .05, s=M[i, j],
#                        ha='center', va='center', size=h * 8)

    figname = "../figures/{}_x={}_y={}_h={}.pdf".format(geometry, x, y, h)
    plt.savefig(figname)

