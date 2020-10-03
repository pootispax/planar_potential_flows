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
    matrix = np.zeros((x, y))           # Create an array of size x * y
    
    if main.geometry == 1:
        for i in range(y // 3, 2 * y // 3):
            for j in range(1, x - 1):
                matrix[i, j] = 1
            for j in range(0, 1):
                matrix[i, j] = 2
            for j in range(x - 1, x):
                matrix[i, j] = 3

    elif main.geometry == 2:
        xtier = x // 3
        yoffset = 0
        for i in range(1, y - 1):
            for j in range(1, x - 1):
                if i < xtier:
                    matrix[i, j + yoffset] = 1
                elif i > 2 * xtier:
                    matrix[i, j - yoffset] = 1
            for j in range(0, 1):
                matrix[i, j] = 2
            for j in range(x - 1, x):
                matrix[i, j] = 3
            if i < xtier:
                yoffset += 1
            elif i + 1 > 2 * xtier:
                yoffset -= 1
            print(yoffset)

#    plt.imshow(matrix, cmap='coolwarm')
    print(matrix)
    print(type(matrix[0, 0]))
#    plt.colorbar()
#    plt.show()

boxBuild()

