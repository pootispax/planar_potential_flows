# This program contains the functions for the alpha version

import numpy as np 
import matplotlib.pyplot as plt
import main_alpha as main

G = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
     [2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 3.],
     [2., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 3.],
     [2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.],
     [2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.],
     [2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.],
     [2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.],
     [2., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 3.],
     [2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 3.],
     [2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
G = np.asarray(G) # Convert the list G to a matrix

M = np.zeros((12, 12))

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

plt.figure()
plt.plot(XX - .5, YY - .5, ls='-', color='black')
plt.plot(YY - .5, XX - .5, ls='-', color='black')
plt.imshow(G, cmap='coolwarm')
plt.colorbar()

for i in range(0, 12):
    for j in range(0, 12):
        if G[i, j] != 0:
            plt.text(x=j, y=i + .05, s=M[i, j], ha='center', va='center')
plt.savefig("matrix_geo_2.pdf")
