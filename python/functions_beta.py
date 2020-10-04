# All the functions used by the main program are in this file
# Python version used at the start of the project : Python 3.8.5

import numpy as np
import matplotlib.pyplot as plt
import main_beta as main

# This function builds a matrix x * y cells
# h represents the size of a cell
# To complete for gold version
def boxBuild(x=12, y=12, h=0):
    plt.figure()
    G = np.zeros((x, y))           # Create an array of size x * y
    
    if main.geometry == 1:
        for i in range(y // 3, 2 * y // 3):
            for j in range(1, x - 1):
                G[i, j] = 1
            for j in range(0, 1):
                G[i, j] = 2
            for j in range(x - 1, x):
                G[i, j] = 3

    elif main.geometry == 2:
        xtier = x // 3
        yoffset = 1
        print(xtier, 2 * xtier)
        for i in range(0, x):
            if i == 0:
                for j in range(1, y - 1):
                    G[j, i] = 2
            elif i == x - 1:
                for j in range(1, y - 1):
                    G[j, i] = 3
            elif i < xtier:
                for j in range(1 + yoffset, y - 1 - yoffset):
                    G[j, i] = 1
                yoffset += 1
            elif i >= 2 * xtier:
                yoffset -= 1
                for j in range(1 + yoffset, y - 1 - yoffset):
                    G[j, i] = 1
            else:
                for j in range(xtier, 2 * xtier):
                    G[j, i] = 1

# -------------------------------------------------------------
        M = np.zeros((x, y))

        indent = 1
        for i in range(0, 12):
            for j in range(0, 12):
                if G[i, j] != 0:
                    M[i, j] = indent
                    indent += 1
        M = M.astype(int) # Convert each cell of the matrix to an int
        N_fluid = np.max(M)
        A = np.zeros((N_fluid, N_fluid))
        
        X = np.linspace(0, 12, 13)
        Y = np.linspace(0, 12, 13)
        XX, YY = np.meshgrid(X, Y)
        
        plt.plot(XX - .5, YY - .5, ls='-', color='black')
        plt.plot(YY - .5, XX - .5, ls='-', color='black')
        
        for i in range(0, 12):
            for j in range(0, 12):
                if G[i, j] != 0:
                    plt.text(x=j, y=i + .05, s=M[i, j], ha='center', va='center')
        plt.imshow(G, cmap='coolwarm')
        print(G)
        plt.colorbar()
        plt.savefig("fig_beta.pdf")

boxBuild(12, 12)

